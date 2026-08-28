using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real, weight-driven QwenTTS end-to-end pipeline: text -&gt; Talker semantic codes (with a real
/// per-frame Code Predictor acoustic-depth-expansion, using <see cref="ForwardPass.LastHidden"/>
/// -- see docs/audio-review-progress.md's QwenTTS entries for the full real derivation) -&gt; 16-
/// codebook frames -&gt; the real, independently golden-verified codec decode chain (RVQ -&gt; pre-
/// conv -&gt; transformer -&gt; ConvNeXt upsample -&gt; DAC) -&gt; 24kHz waveform.
///
/// <para>Real, deliberate simplification (documented, not silently dropped): stops generation
/// at the real codec EOS id from the Talker only (the Code Predictor's own real stop-token
/// convention from `code-predictor-forward.h` -- e.g. a maximum silent-frame count -- is not
/// replicated here; a fixed `maxFrames` cap is used instead).</para>
/// </summary>
public sealed class QwenTtsPipeline : ITextToSpeechPipeline
{
    public string Architecture => "QwenTTS";
    public int SampleRate => 24000;
    public int DefaultSampleRate => 24000;

    private readonly GgufModel _talkerModel;
    private readonly GgufModel _codecModel;

    private QwenTtsPipeline(GgufModel talkerModel, GgufModel codecModel)
    {
        _talkerModel = talkerModel;
        _codecModel = codecModel;
    }

    public static QwenTtsPipeline Load(string modelPath, string? codecGgufPath = null)
    {
        string dir = Path.GetDirectoryName(modelPath) ?? "models";
        codecGgufPath ??= Path.Combine(dir, "qwen-tokenizer-12hz-Q8_0.gguf");
        if (!File.Exists(codecGgufPath))
        {
            codecGgufPath = Path.Combine(dir, "qwen-tokenizer-12hz.gguf");
        }
        var talkerModel = GgufModel.Open(modelPath);
        var codecModel = GgufModel.Open(codecGgufPath);
        return new QwenTtsPipeline(talkerModel, codecModel);
    }

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        var pcm = Generate(request.Text);
        var result = new AudioGenerationResult(pcm, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }
        return result;
    }

    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, System.Threading.CancellationToken ct = default)
        => TtsStreamingHelper.SplitAndGenerateAsync(request, Generate, ct);

    /// <summary>Synthesizes real 24kHz PCM audio for the given text.</summary>
    /// <param name="seed">RNG seed for the talker/code-predictor sampling (real Qwen3-TTS defaults
    /// to <c>do_sample=True, temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05</c> --
    /// confirmed from the local reference source, `examples/qwen-tts-py/qwen_tts/core/models/
    /// modeling_qwen3_tts.py`. Greedy argmax was this pipeline's original decode strategy, same
    /// class of bug as Parler-TTS's "drill noise" collapse -- fixed 2026-08-28, see
    /// docs/audio-review-progress.md.</param>
    public float[] Generate(string text, int talkerNumLayers = 28, int codePredNumLayers = 5, int maxFrames = 50, string? language = null, int seed = 42)
    {
        var frames = GenerateFrames(text, talkerNumLayers, codePredNumLayers, maxFrames, language, seed);
        if (frames.Count == 0) return [];

        // Real codec decode chain, already independently golden-verified earlier this session:
        // codes[16][T] (semantic + 15 acoustic) -> RVQ decode -> pre-conv -> transformer ->
        // ConvNeXt upsample x2 -> DAC decoder chain -> waveform.
        int t = frames.Count;
        var codes = new int[16][];
        for (int g = 0; g < 16; g++)
        {
            codes[g] = new int[t];
            for (int i = 0; i < t; i++) codes[g][i] = frames[i][g];
        }

        var rvqWeights = new QwenTtsCodecRvqWeights(_codecModel);
        var preConvWeights = new QwenTtsCodecPreConvWeights(_codecModel);
        var transformerWeights = new QwenTtsCodecTransformerWeights(_codecModel);
        var upsampleWeights0 = new QwenTtsCodecUpsampleWeights(_codecModel, stage: 0);
        var upsampleWeights1 = new QwenTtsCodecUpsampleWeights(_codecModel, stage: 1);
        var dacWeights = new QwenTtsCodecDacWeights(_codecModel);

        var rvqOut = QwenTtsCodecRvq.Decode(rvqWeights, codes);
        var preConvOut = QwenTtsCodecPreConv.Forward(preConvWeights, rvqOut);
        var transformerOut = QwenTtsCodecTransformer.Forward(transformerWeights, preConvOut);
        var up0 = QwenTtsCodecUpsample.Forward(upsampleWeights0, transformerOut);
        var up1 = QwenTtsCodecUpsample.Forward(upsampleWeights1, up0);
        var wav = QwenTtsCodecDac.Forward(dacWeights, up1);

        return wav;
    }

    /// <summary>
    /// Generates real 16-codebook frames (`[c0, c1..c15]` per frame): the Talker's semantic
    /// decode loop, with a real Code Predictor acoustic-expansion pass run per frame using that
    /// frame's real `LastHidden`.
    /// </summary>
    private List<int[]> GenerateFrames(string text, int talkerNumLayers, int codePredNumLayers, int maxFrames, string? language, int seed)
    {
        var talkerWeights = QwenTtsTalkerPromptBuilder.Weights.Load(_talkerModel);
        var tokenizer = GgufTokenizer.FromGgufModel(_talkerModel);
        var languageTable = QwenTtsTalkerPromptBuilder.ReadLanguageTable(_talkerModel);
        var (promptEmbed, tRows) = QwenTtsTalkerPromptBuilder.BuildBasePrompt(talkerWeights, tokenizer, text, language, languageTable);

        using var talkerSource = new QwenTtsTalkerTensorSource(_talkerModel, talkerNumLayers);

        // Real fix for the "LastHidden is all-zero right after Prefill alone" constraint this
        // session found: prefill everything except the last prompt row, then a real Forward
        // step for that last row, so LastHidden is valid from the very first generated frame.
        var prefillRows = new float[(tRows - 1) * QwenTtsTalkerPromptBuilder.TalkerHiddenDim];
        Array.Copy(promptEmbed, prefillRows, prefillRows.Length);
        talkerSource.SetPromptEmbedding(prefillRows, tRows - 1);

        var hp = ModelHyperparams.FromGgufMetadata(talkerSource.Metadata);
        if (Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_GOLDEN_DUMP") is not null)
            Console.Error.WriteLine($"hp: HeadDim={hp.HeadDim} NumHeads={hp.NumHeads} NumKvHeads={hp.NumKvHeads} " +
                $"EmbeddingDim={hp.EmbeddingDim} RopeTheta={hp.RopeTheta} RmsNormEps={hp.RmsNormEps} NumLayers={hp.NumLayers}");
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(talkerSource, backend, hp);

        var prefillIds = new int[tRows - 1];
        for (int i = 0; i < prefillIds.Length; i++) prefillIds[i] = i;
        if (prefillIds.Length > 0) _ = fwd.Prefill(prefillIds);

        var lastRow = new float[QwenTtsTalkerPromptBuilder.TalkerHiddenDim];
        Array.Copy(promptEmbed, (tRows - 1) * QwenTtsTalkerPromptBuilder.TalkerHiddenDim, lastRow, 0, QwenTtsTalkerPromptBuilder.TalkerHiddenDim);
        talkerSource.SetPromptEmbedding(lastRow, 1);
        var logits = fwd.Forward(0, tRows - 1).ToArray();
        int pos = tRows;

        if (Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_GOLDEN_DUMP") is { } dumpDir)
        {
            System.IO.Directory.CreateDirectory(dumpDir);
            System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "prompt_embed.csv"),
                $"{tRows},{QwenTtsTalkerPromptBuilder.TalkerHiddenDim}\n" + string.Join(",", promptEmbed));
            var lastHidden = fwd.LastHidden.ToArray();
            System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "last_hidden.csv"),
                $"1,{lastHidden.Length}\n" + string.Join(",", lastHidden));
            System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "logits.csv"),
                $"1,{logits.Length}\n" + string.Join(",", logits));
        }

        var codePredWeights = QwenTtsCodePredictorGeneration.Weights.Load(_talkerModel);
        var specials = talkerWeights.Specials;
        var frames = new List<int[]>();
        var c0History = new List<int>();
        var rng = new Random(seed);

        for (int frame = 0; frame < maxFrames; frame++)
        {
            int c0 = SampleTopK(logits, c0History, temperature: 0.9f, topK: 50, repetitionPenalty: 1.05f, rng);
            if (c0 == specials.CodecEosId) break;
            c0History.Add(c0);

            var talkerLastHidden = fwd.LastHidden.ToArray();
            var acoustic = QwenTtsCodePredictorGeneration.GenerateAcousticCodes(_talkerModel, codePredWeights, codePredNumLayers, c0, talkerLastHidden, rng);

            var frameCodes = new int[16];
            frameCodes[0] = c0;
            Array.Copy(acoustic, 0, frameCodes, 1, 15);
            frames.Add(frameCodes);

            // Real next-step feedback sums ALL 16 codebook embeddings (semantic c0 via the
            // talker's own codec table, acoustic c1..c15 via the code predictor's 15 per-codebook
            // tables), confirmed from the real reference's `tts_engine_step` (`pipeline-tts.cpp`
            // ~line 1106-1119: `ids[g * N_dec + i] = s.prev_ids[g]` for ALL `num_code_groups`,
            // fed together with `pt->code_predictor.codec_embedding` into the next talker decode).
            // Found and fixed 2026-08-28: this pipeline previously only fed c0's embedding,
            // silently dropping the 15 acoustic codes the code predictor generates every single
            // frame -- real, per-frame signal loss that compounds across the whole autoregressive
            // sequence, consistent with the "garbled" (not clean) speech this produced even after
            // the greedy-decode/sampling fix. See docs/audio-review-progress.md.
            var stepRow = QwenTtsTalkerPromptBuilder.ProjectTextIds(talkerWeights, [specials.TtsPadId]);
            var codecVec = QwenTtsTalkerPromptBuilder.CodecEmbedRow(talkerWeights, c0);
            for (int d = 0; d < QwenTtsTalkerPromptBuilder.TalkerHiddenDim; d++) stepRow[d] += codecVec[d];
            for (int g = 0; g < acoustic.Length; g++)
            {
                var acVec = new float[QwenTtsTalkerPromptBuilder.TalkerHiddenDim];
                Array.Copy(codePredWeights.CodecEmbd[g], (long)acoustic[g] * QwenTtsTalkerPromptBuilder.TalkerHiddenDim,
                    acVec, 0, QwenTtsTalkerPromptBuilder.TalkerHiddenDim);
                for (int d = 0; d < QwenTtsTalkerPromptBuilder.TalkerHiddenDim; d++) stepRow[d] += acVec[d];
            }

            talkerSource.SetPromptEmbedding(stepRow, 1);
            logits = fwd.Forward(0, pos).ToArray();
            pos++;
        }

        return frames;
    }

    /// <summary>
    /// Real Qwen3-TTS talker sampling: repetition penalty (standard HF
    /// <c>RepetitionPenaltyLogitsProcessor</c> convention -- for every token id that appears
    /// ANYWHERE in <paramref name="history"/>, divide its logit by the penalty if positive, else
    /// multiply) over the FULL generated history (not a small window -- confirmed the real source
    /// just forwards <c>repetition_penalty</c> straight into HF's standard `generate()`, so it's
    /// the standard processor, not a bespoke windowed one like CosyVoice3's), then
    /// temperature-scaled top-k softmax sampling. <c>top_p=1.0</c> in the real default config makes
    /// nucleus filtering a no-op (keeps the whole top-k set), so it is not implemented here --
    /// would need adding if a caller ever wants a real <c>top_p &lt; 1.0</c>.
    /// </summary>
    private static int SampleTopK(float[] logits, List<int> history, float temperature, int topK, float repetitionPenalty, Random rng)
    {
        if (repetitionPenalty != 1.0f)
        {
            foreach (int tok in history)
            {
                if ((uint)tok >= (uint)logits.Length) continue;
                logits[tok] = logits[tok] > 0 ? logits[tok] / repetitionPenalty : logits[tok] * repetitionPenalty;
            }
        }

        int k = Math.Min(topK, logits.Length);
        Span<int> topIdx = stackalloc int[k];
        Span<float> topVal = stackalloc float[k];
        int filled = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i] / temperature;
            if (filled < k)
            {
                int pos = filled++;
                while (pos > 0 && topVal[pos - 1] > v) { topVal[pos] = topVal[pos - 1]; topIdx[pos] = topIdx[pos - 1]; pos--; }
                topVal[pos] = v; topIdx[pos] = i;
            }
            else if (v > topVal[0])
            {
                int pos = 0;
                while (pos < k - 1 && topVal[pos + 1] < v) { topVal[pos] = topVal[pos + 1]; topIdx[pos] = topIdx[pos + 1]; pos++; }
                topVal[pos] = v; topIdx[pos] = i;
            }
        }

        float max = topVal[k - 1];
        double sum = 0.0;
        for (int i = 0; i < k; i++) sum += Math.Exp(topVal[i] - max);

        double r = rng.NextDouble() * sum;
        double cumulative = 0.0;
        for (int i = 0; i < k; i++)
        {
            cumulative += Math.Exp(topVal[i] - max);
            if (r < cumulative) return topIdx[i];
        }
        return topIdx[k - 1];
    }

    public void Dispose()
    {
        _talkerModel.Dispose();
        _codecModel.Dispose();
    }
}
