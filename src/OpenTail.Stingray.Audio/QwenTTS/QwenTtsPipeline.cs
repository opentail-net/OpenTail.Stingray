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

    public async IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken ct = default)
    {
        var res = Generate(request);
        yield return res.Samples;
    }

    /// <summary>Synthesizes real 24kHz PCM audio for the given text.</summary>
    public float[] Generate(string text, int talkerNumLayers = 28, int codePredNumLayers = 5, int maxFrames = 50, string? language = null)
    {
        var frames = GenerateFrames(text, talkerNumLayers, codePredNumLayers, maxFrames, language);
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
    private List<int[]> GenerateFrames(string text, int talkerNumLayers, int codePredNumLayers, int maxFrames, string? language)
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

        var codePredWeights = QwenTtsCodePredictorGeneration.Weights.Load(_talkerModel);
        var specials = talkerWeights.Specials;
        var frames = new List<int[]>();

        for (int frame = 0; frame < maxFrames; frame++)
        {
            int c0 = ArgMax(logits);
            if (c0 == specials.CodecEosId) break;

            var talkerLastHidden = fwd.LastHidden.ToArray();
            var acoustic = QwenTtsCodePredictorGeneration.GenerateAcousticCodes(_talkerModel, codePredWeights, codePredNumLayers, c0, talkerLastHidden);

            var frameCodes = new int[16];
            frameCodes[0] = c0;
            Array.Copy(acoustic, 0, frameCodes, 1, 15);
            frames.Add(frameCodes);

            var stepRow = QwenTtsTalkerPromptBuilder.ProjectTextIds(talkerWeights, [specials.TtsPadId]);
            var codecVec = QwenTtsTalkerPromptBuilder.CodecEmbedRow(talkerWeights, c0);
            for (int d = 0; d < QwenTtsTalkerPromptBuilder.TalkerHiddenDim; d++) stepRow[d] += codecVec[d];

            talkerSource.SetPromptEmbedding(stepRow, 1);
            logits = fwd.Forward(0, pos).ToArray();
            pos++;
        }

        return frames;
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    public void Dispose()
    {
        _talkerModel.Dispose();
        _codecModel.Dispose();
    }
}
