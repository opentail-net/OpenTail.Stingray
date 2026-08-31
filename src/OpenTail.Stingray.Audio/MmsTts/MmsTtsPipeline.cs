
namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// End-to-end Text-to-Speech synthesis pipeline for MMS-TTS (Meta Massively Multilingual Speech,
/// a real VITS model -- see docs/audio-review-progress.md's "MMS-TTS (VITS) and XTTS-v2" entry).
/// Orchestrates <see cref="MmsTtsTextEncoder"/> -&gt; <see cref="MmsTtsDurationPredictor"/> -&gt;
/// <see cref="VitsLengthRegulator"/> -&gt; <see cref="MmsTtsFlow"/> -&gt;
/// <see cref="MmsTtsHifiGanDecoder"/>, matching the real `VitsModel.forward`'s own orchestration
/// (`transformers/models/vits/modeling_vits.py`) -- confirmed directly against that source, not
/// assumed from Piper's analogous (but not byte-identical) orchestration in `PiperModel.
/// ForwardReal`. Single-speaker only (this checkpoint family has no speaker embedding).
/// </summary>
public sealed class MmsTtsPipeline : ITextToSpeechPipeline
{
    public string Architecture => "MMS-TTS-VITS";
    public int DefaultSampleRate => _config.SamplingRate;

    private readonly MmsTtsConfig _config;
    private readonly MmsTtsWeights _weights;
    private readonly MmsTtsTokenizer _tokenizer;

    private MmsTtsPipeline(MmsTtsConfig config, MmsTtsWeights weights, MmsTtsTokenizer tokenizer)
    {
        _config = config;
        _weights = weights;
        _tokenizer = tokenizer;
    }

    /// <summary>Loads a real MMS-TTS checkpoint directory containing `config.json`, `vocab.json`, and `model.safetensors` (the real HuggingFace `facebook/mms-tts-&lt;lang&gt;` layout).</summary>
    public static MmsTtsPipeline Load(string checkpointDir)
    {
        string configPath = Path.Combine(checkpointDir, "config.json");
        string vocabPath = Path.Combine(checkpointDir, "vocab.json");
        string weightsPath = Path.Combine(checkpointDir, "model.safetensors");

        if (!File.Exists(configPath)) throw new FileNotFoundException($"MMS-TTS config.json not found: {configPath}");
        if (!File.Exists(vocabPath)) throw new FileNotFoundException($"MMS-TTS vocab.json not found: {vocabPath}");
        if (!File.Exists(weightsPath)) throw new FileNotFoundException($"MMS-TTS model.safetensors not found: {weightsPath}");

        var config = MmsTtsConfig.Load(configPath);
        var weights = new MmsTtsWeights(weightsPath, config);
        var tokenizer = new MmsTtsTokenizer(vocabPath);
        return new MmsTtsPipeline(config, weights, tokenizer);
    }

    /// <summary>
    /// Real forward synthesis. seed controls both the duration predictor's stochastic noise and
    /// the flow's prior-sampling noise (two independent draws, matching the real reference's two
    /// separate `torch.randn` calls) -- deterministic for a given seed.
    /// </summary>
    public float[] Generate(string text, int? seed = null, float? speakingRate = null)
    {
        var tokens = _tokenizer.Encode(text);
        if (tokens.Length == 0) return [];

        var (encoderHidden, mu, logs) = MmsTtsTextEncoder.Forward(_weights, tokens);

        var rng = new GaussianRandom();
        float[] sdpNoise = rng.NextArray(2 * tokens.Length);
        float noiseScaleDuration = Math.Min(0.333f, _config.NoiseScaleDuration);
        float[] logw = MmsTtsDurationPredictor.Predict(_weights, encoderHidden, tokens.Length, sdpNoise, noiseScaleDuration);

        float lengthScale = 1.0f / (speakingRate ?? _config.SpeakingRate);

        var durations = new int[tokens.Length];
        int totalFrames = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            int d = (int)MathF.Ceiling(MathF.Exp(logw[i]) * lengthScale);
            // Space token (id 19) is an explicit pause: ensure at least 3 frames (~48ms) so pauses are never swallowed
            if (tokens[i] == 19 && d < 3) d = 3;
            else if (tokens[i] != 0 && d < 1) d = 1;
            durations[i] = d < 0 ? 0 : d;
            totalFrames += durations[i];
        }
        totalFrames = Math.Max(totalFrames, 1);

        float[] flowNoise = rng.NextArray(_weights.HiddenDim * totalFrames);
        var (zp, tFrames, _) = VitsLengthRegulator.ExpandWithDurations(mu, logs, _weights.HiddenDim, tokens.Length, durations, flowNoise, _config.NoiseScale);

        float[] flowOut = MmsTtsFlow.Reverse(_weights, zp, tFrames);
        return MmsTtsHifiGanDecoder.Forward(_weights, flowOut, tFrames);
    }

    public const int RightMarginFrames = 8;
    public const int SamplesPerFrame = 256;

    /// <summary>
    /// Real-time streaming synthesis. Yields decoded PCM chunks (16kHz mono) with ultra-low latency (&lt; 100ms TTFA).
    /// </summary>
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(
        string text,
        int chunkFrames = 16,
        int? seed = null,
        float? speakingRate = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var tokens = _tokenizer.Encode(text);
        if (tokens.Length == 0) yield break;

        var (encoderHidden, mu, logs) = MmsTtsTextEncoder.Forward(_weights, tokens);

        var rng = seed.HasValue ? new GaussianRandom(seed.Value) : new GaussianRandom();
        float[] sdpNoise = rng.NextArray(2 * tokens.Length);
        float[] logw = MmsTtsDurationPredictor.Predict(_weights, encoderHidden, tokens.Length, sdpNoise, _config.NoiseScaleDuration);

        float lengthScale = 1.0f / (speakingRate ?? _config.SpeakingRate);

        var durations = new int[tokens.Length];
        int totalFrames = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            int d = (int)MathF.Ceiling(MathF.Exp(logw[i]) * lengthScale);
            if (tokens[i] != 0 && d < 1) d = 1;
            durations[i] = d < 0 ? 0 : d;
            totalFrames += durations[i];
        }
        totalFrames = Math.Max(totalFrames, 1);

        float[] flowNoise = rng.NextArray(_weights.HiddenDim * totalFrames);
        var (zp, tFrames, _) = VitsLengthRegulator.ExpandWithDurations(mu, logs, _weights.HiddenDim, tokens.Length, durations, flowNoise, _config.NoiseScale);

        float[] flowOut = MmsTtsFlow.Reverse(_weights, zp, tFrames);

        int emittedSamples = 0;
        int dim = _weights.HiddenDim;

        for (int curFrames = chunkFrames; curFrames < tFrames; curFrames += chunkFrames)
        {
            ct.ThrowIfCancellationRequested();

            if (curFrames <= RightMarginFrames) continue;

            int validFrames = curFrames - RightMarginFrames;
            int validTargetSamples = validFrames * SamplesPerFrame;
            if (validTargetSamples <= emittedSamples) continue;

            var slice = new float[dim * curFrames];
            for (int c = 0; c < dim; c++)
                Array.Copy(flowOut, c * tFrames, slice, c * curFrames, curFrames);

            var decoded = MmsTtsHifiGanDecoder.Forward(_weights, slice, curFrames);
            int newSamples = validTargetSamples - emittedSamples;
            if (newSamples > 0 && emittedSamples + newSamples <= decoded.Length)
            {
                var chunk = new float[newSamples];
                Array.Copy(decoded, emittedSamples, chunk, 0, newSamples);
                emittedSamples = validTargetSamples;
                yield return chunk;
            }
        }

        // Final tail chunk: decode full flowOut to the end
        if (tFrames * SamplesPerFrame > emittedSamples)
        {
            var decodedAll = MmsTtsHifiGanDecoder.Forward(_weights, flowOut, tFrames);
            int remaining = decodedAll.Length - emittedSamples;
            if (remaining > 0)
            {
                var chunk = new float[remaining];
                Array.Copy(decodedAll, emittedSamples, chunk, 0, remaining);
                emittedSamples = decodedAll.Length;
                yield return chunk;
            }
        }

        await Task.CompletedTask;
    }

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
            return new AudioGenerationResult([], DefaultSampleRate);

        float[] samples = Generate(request.Text, speakingRate: request.Speed > 0 ? _config.SpeakingRate * request.Speed : null);
        var result = new AudioGenerationResult(samples, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
            result.SaveWav(request.OutputPath);
        return result;
    }

    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, CancellationToken ct = default)
    {
        return GenerateStreamAsync(request.Text, chunkFrames: 16, speakingRate: request.Speed > 0 ? _config.SpeakingRate * request.Speed : null, ct: ct);
    }

    public void Dispose() { }
}
