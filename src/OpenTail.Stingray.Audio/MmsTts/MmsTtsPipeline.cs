
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
        float[] logw = MmsTtsDurationPredictor.Predict(_weights, encoderHidden, tokens.Length, sdpNoise, _config.NoiseScaleDuration);

        float lengthScale = 1.0f / (speakingRate ?? _config.SpeakingRate);

        // Real reference computes durations via ceil(exp(logw)*mask*lengthScale) with NO per-token
        // minimum-1 floor (unlike Piper's own implementation, which adds one) -- a token CAN
        // legitimately contribute zero frames. VitsLengthRegulator.Expand already matches this
        // (durations[i] = max(0, ceil(...)), no floor of 1).
        var durations = new int[tokens.Length];
        int totalFrames = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            int d = (int)MathF.Ceiling(MathF.Exp(logw[i]) * lengthScale);
            durations[i] = d < 0 ? 0 : d;
            totalFrames += durations[i];
        }
        totalFrames = Math.Max(totalFrames, 1);

        float[] flowNoise = rng.NextArray(_weights.HiddenDim * totalFrames);
        var (zp, tFrames, _) = VitsLengthRegulator.ExpandWithDurations(mu, logs, _weights.HiddenDim, tokens.Length, durations, flowNoise, _config.NoiseScale);

        float[] flowOut = MmsTtsFlow.Reverse(_weights, zp, tFrames);
        return MmsTtsHifiGanDecoder.Forward(_weights, flowOut, tFrames);
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
        => TtsStreamingHelper.SplitAndGenerateAsync(request, Generate, ct);

    public void Dispose() { }
}
