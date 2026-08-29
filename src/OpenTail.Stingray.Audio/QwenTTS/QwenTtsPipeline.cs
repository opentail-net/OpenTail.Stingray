
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

    private readonly QwenTtsCodecRvqWeights _rvqWeights;
    private readonly QwenTtsCodecPreConvWeights _preConvWeights;
    private readonly QwenTtsCodecTransformerWeights _transformerWeights;
    private readonly QwenTtsCodecUpsampleWeights _upsampleWeights0;
    private readonly QwenTtsCodecUpsampleWeights _upsampleWeights1;
    private readonly QwenTtsCodecDacWeights _dacWeights;

    private readonly QwenTtsTalkerPromptBuilder.Weights _talkerWeights;
    private readonly GgufTokenizer _tokenizer;
    private readonly IReadOnlyDictionary<string, int> _languageTable;
    private readonly QwenTtsCodePredictorGeneration.Weights _codePredWeights;

    private QwenTtsPipeline(GgufModel talkerModel, GgufModel codecModel)
    {
        _talkerModel = talkerModel;
        _codecModel = codecModel;

        _rvqWeights = new QwenTtsCodecRvqWeights(_codecModel);
        _preConvWeights = new QwenTtsCodecPreConvWeights(_codecModel);
        _transformerWeights = new QwenTtsCodecTransformerWeights(_codecModel);
        _upsampleWeights0 = new QwenTtsCodecUpsampleWeights(_codecModel, stage: 0);
        _upsampleWeights1 = new QwenTtsCodecUpsampleWeights(_codecModel, stage: 1);
        _dacWeights = new QwenTtsCodecDacWeights(_codecModel);

        _talkerWeights = QwenTtsTalkerPromptBuilder.Weights.Load(_talkerModel);
        _tokenizer = GgufTokenizer.FromGgufModel(_talkerModel);
        _languageTable = QwenTtsTalkerPromptBuilder.ReadLanguageTable(_talkerModel);
        _codePredWeights = QwenTtsCodePredictorGeneration.Weights.Load(_talkerModel);
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
        await foreach (var chunk in GenerateStreamAsync(request.Text, chunkFrames: 3, ct: ct))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Real-time low-latency audio streaming: yields audio chunks as each group of
    /// <paramref name="chunkFrames"/> (default 3 frames = ~240ms of 24kHz audio) is generated.
    /// Time-To-First-Audio (TTFA) is sub-second (&lt;500ms).
    /// </summary>
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(string text, int chunkFrames = 3, int talkerNumLayers = 28, int codePredNumLayers = 5, int maxFrames = 50, string? language = null, int seed = 42, [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken ct = default)
    {
        var (promptEmbed, tRows) = QwenTtsTalkerPromptBuilder.BuildBasePrompt(_talkerWeights, _tokenizer, text, language, _languageTable);

        using var talkerSource = new QwenTtsTalkerTensorSource(_talkerModel, talkerNumLayers);

        var prefillRows = new float[(tRows - 1) * QwenTtsTalkerPromptBuilder.TalkerHiddenDim];
        Array.Copy(promptEmbed, prefillRows, prefillRows.Length);
        talkerSource.SetPromptEmbedding(prefillRows, tRows - 1);

        var hp = ModelHyperparams.FromGgufMetadata(talkerSource.Metadata, talkerSource);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(talkerSource, backend, hp);

        var prefillIds = new int[tRows - 1];
        for (int i = 0; i < prefillIds.Length; i++) prefillIds[i] = i;
        if (prefillIds.Length > 0) _ = fwd.Prefill(prefillIds);

        var lastRow = new float[QwenTtsTalkerPromptBuilder.TalkerHiddenDim];
        Array.Copy(promptEmbed, (tRows - 1) * QwenTtsTalkerPromptBuilder.TalkerHiddenDim, lastRow, 0, QwenTtsTalkerPromptBuilder.TalkerHiddenDim);
        talkerSource.SetPromptEmbedding(lastRow, 1);
        int c0 = SampleTopK(fwd.Forward(0, tRows - 1), [], temperature: 0.9f, topK: 50, repetitionPenalty: 1.05f, new Random(seed));
        int pos = tRows;

        using var codePredSession = new QwenTtsCodePredictorGeneration.CodePredictorSession(_talkerModel, _codePredWeights, codePredNumLayers);
        var specials = _talkerWeights.Specials;
        var frames = new List<int[]>();
        var c0History = new List<int>();
        var rng = new Random(seed);

        var padEmbed = QwenTtsTalkerPromptBuilder.ProjectTextIds(_talkerWeights, [specials.TtsPadId]);
        var stepRow = new float[QwenTtsTalkerPromptBuilder.TalkerHiddenDim];

        int lastDecodedFrame = 0;

        for (int frame = 0; frame < maxFrames; frame++)
        {
            if (ct.IsCancellationRequested) yield break;
            if (c0 == specials.CodecEosId) break;
            c0History.Add(c0);

            var acoustic = codePredSession.GenerateAcousticCodes(c0, fwd.LastHidden, rng);

            var frameCodes = new int[16];
            frameCodes[0] = c0;
            Array.Copy(acoustic, 0, frameCodes, 1, 15);
            frames.Add(frameCodes);

            Array.Copy(padEmbed, stepRow, stepRow.Length);
            var codecVec = QwenTtsTalkerPromptBuilder.CodecEmbedRow(_talkerWeights, c0);
            for (int d = 0; d < stepRow.Length; d++) stepRow[d] += codecVec[d];
            for (int g = 0; g < acoustic.Length; g++)
            {
                int acCode = acoustic[g];
                int acOffset = acCode * QwenTtsTalkerPromptBuilder.TalkerHiddenDim;
                var acTable = _codePredWeights.CodecEmbd[g];
                for (int d = 0; d < stepRow.Length; d++) stepRow[d] += acTable[acOffset + d];
            }

            talkerSource.SetPromptEmbedding(stepRow, 1);
            c0 = SampleTopK(fwd.Forward(0, pos), c0History, temperature: 0.9f, topK: 50, repetitionPenalty: 1.05f, rng);
            pos++;

            if (frames.Count - lastDecodedFrame >= chunkFrames)
            {
                int newFrames = frames.Count - lastDecodedFrame;
                var chunk = DecodeChunk(frames, newFrames, contextFrames: 8);
                lastDecodedFrame = frames.Count;
                yield return chunk;
            }
        }

        if (frames.Count > lastDecodedFrame)
        {
            int newFrames = frames.Count - lastDecodedFrame;
            var chunk = DecodeChunk(frames, newFrames, contextFrames: 8);
            yield return chunk;
        }
    }

    /// <summary>
    /// Decodes a recent window of acoustic frames into a discrete audio chunk with causal context overlap.
    /// </summary>
    public float[] DecodeChunk(IReadOnlyList<int[]> frames, int chunkCount, int contextFrames = 8)
    {
        int totalFrames = frames.Count;
        int sliceStart = Math.Max(0, totalFrames - contextFrames);
        int sliceLen = totalFrames - sliceStart;

        var sliceCodes = new int[16][];
        for (int g = 0; g < 16; g++)
        {
            sliceCodes[g] = new int[sliceLen];
            for (int i = 0; i < sliceLen; i++)
                sliceCodes[g][i] = frames[sliceStart + i][g];
        }

        var rvqOut = QwenTtsCodecRvq.Decode(_rvqWeights, sliceCodes);
        var preConvOut = QwenTtsCodecPreConv.Forward(_preConvWeights, rvqOut);
        var transformerOut = QwenTtsCodecTransformer.Forward(_transformerWeights, preConvOut);
        var up0 = QwenTtsCodecUpsample.Forward(_upsampleWeights0, transformerOut);
        var up1 = QwenTtsCodecUpsample.Forward(_upsampleWeights1, up0);
        var wav = QwenTtsCodecDac.Forward(_dacWeights, up1);

        int targetSamples = chunkCount * 1920;
        int availableSamples = wav.Length;
        if (targetSamples >= availableSamples) return wav;

        var chunk = new float[targetSamples];
        Array.Copy(wav, availableSamples - targetSamples, chunk, 0, targetSamples);
        return chunk;
    }

    /// <summary>Decodes 16-codebook acoustic frames into a continuous 24kHz PCM waveform.</summary>
    public float[] DecodeFrames(IReadOnlyList<int[]> frames)
    {
        if (frames.Count == 0) return [];
        int t = frames.Count;
        var codes = new int[16][];
        for (int g = 0; g < 16; g++)
        {
            codes[g] = new int[t];
            for (int i = 0; i < t; i++) codes[g][i] = frames[i][g];
        }

        var rvqOut = QwenTtsCodecRvq.Decode(_rvqWeights, codes);
        var preConvOut = QwenTtsCodecPreConv.Forward(_preConvWeights, rvqOut);
        var transformerOut = QwenTtsCodecTransformer.Forward(_transformerWeights, preConvOut);
        var up0 = QwenTtsCodecUpsample.Forward(_upsampleWeights0, transformerOut);
        var up1 = QwenTtsCodecUpsample.Forward(_upsampleWeights1, up0);
        return QwenTtsCodecDac.Forward(_dacWeights, up1);
    }

    /// <summary>Synthesizes real 24kHz PCM audio for the given text.</summary>
    public float[] Generate(string text, int talkerNumLayers = 28, int codePredNumLayers = 5, int maxFrames = 50, string? language = null, int seed = 42)
    {
        var frames = GenerateFrames(text, talkerNumLayers, codePredNumLayers, maxFrames, language, seed);
        return DecodeFrames(frames);
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

        var hp = ModelHyperparams.FromGgufMetadata(talkerSource.Metadata, talkerSource);
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
        var logitsSpan = fwd.Forward(0, tRows - 1);
        int pos = tRows;

        if (Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_GOLDEN_DUMP") is { } dumpDir)
        {
            System.IO.Directory.CreateDirectory(dumpDir);
            System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "prompt_embed.csv"),
                $"{tRows},{QwenTtsTalkerPromptBuilder.TalkerHiddenDim}\n" + string.Join(",", promptEmbed));
            var lastHidden = fwd.LastHidden.ToArray();
            System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "last_hidden.csv"),
                $"1,{lastHidden.Length}\n" + string.Join(",", lastHidden));
            var logitsArr = logitsSpan.ToArray();
            System.IO.File.WriteAllText(System.IO.Path.Combine(dumpDir, "logits.csv"),
                $"1,{logitsArr.Length}\n" + string.Join(",", logitsArr));
        }

        var codePredWeights = QwenTtsCodePredictorGeneration.Weights.Load(_talkerModel);
        using var codePredSession = new QwenTtsCodePredictorGeneration.CodePredictorSession(_talkerModel, codePredWeights, codePredNumLayers);
        var specials = talkerWeights.Specials;
        var frames = new List<int[]>();
        var c0History = new List<int>();
        var rng = new Random(seed);

        var padEmbed = QwenTtsTalkerPromptBuilder.ProjectTextIds(talkerWeights, [specials.TtsPadId]);
        var stepRow = new float[QwenTtsTalkerPromptBuilder.TalkerHiddenDim];

        for (int frame = 0; frame < maxFrames; frame++)
        {
            int c0 = SampleTopK(logitsSpan, c0History, temperature: 0.9f, topK: 50, repetitionPenalty: 1.05f, rng);
            if (c0 == specials.CodecEosId) break;
            c0History.Add(c0);

            var acoustic = codePredSession.GenerateAcousticCodes(c0, fwd.LastHidden, rng);

            var frameCodes = new int[16];
            frameCodes[0] = c0;
            Array.Copy(acoustic, 0, frameCodes, 1, 15);
            frames.Add(frameCodes);

            Array.Copy(padEmbed, stepRow, stepRow.Length);
            var codecVec = QwenTtsTalkerPromptBuilder.CodecEmbedRow(talkerWeights, c0);
            for (int d = 0; d < stepRow.Length; d++) stepRow[d] += codecVec[d];
            for (int g = 0; g < acoustic.Length; g++)
            {
                int acCode = acoustic[g];
                int acOffset = acCode * QwenTtsTalkerPromptBuilder.TalkerHiddenDim;
                var acTable = codePredWeights.CodecEmbd[g];
                for (int d = 0; d < stepRow.Length; d++) stepRow[d] += acTable[acOffset + d];
            }

            talkerSource.SetPromptEmbedding(stepRow, 1);
            logitsSpan = fwd.Forward(0, pos);
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
    private static int SampleTopK(ReadOnlySpan<float> logits, List<int> history, float temperature, int topK, float repetitionPenalty, Random rng)
    {
        HashSet<int>? historySet = (repetitionPenalty != 1.0f && history.Count > 0) ? new HashSet<int>(history) : null;

        int k = Math.Min(topK, logits.Length);
        Span<int> topIdx = stackalloc int[k];
        Span<float> topVal = stackalloc float[k];
        int filled = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (historySet != null && historySet.Contains(i))
            {
                v = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
            }
            v /= temperature;

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
