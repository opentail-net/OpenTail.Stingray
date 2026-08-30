namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// End-to-end Text-to-Speech synthesis pipeline for Piper (VITS).
/// </summary>
public sealed class PiperPipeline : ITextToSpeechPipeline
{
    public string Architecture => "Piper-VITS";
    public int DefaultSampleRate => _config.Audio.SampleRate;

    private readonly PiperConfig _config;
    private readonly PiperPhonemizer _phonemizer;
    private readonly PiperModel _model;

    public PiperPipeline(PiperConfig? config = null, PiperModel? model = null, PiperPhonemizer? phonemizer = null)
    {
        _config = config ?? new PiperConfig();
        _model = model ?? new PiperModel(sampleRate: _config.Audio.SampleRate);
        _phonemizer = phonemizer ?? new PiperPhonemizer(_config);
    }

    /// <summary>
    /// Loads a Piper pipeline from a model config JSON file (.onnx.json).
    /// </summary>
    public static PiperPipeline FromConfigFile(string configPath)
    {
        var config = PiperConfig.Load(configPath);

        // Piper's convention: "{voice}.onnx.json" alongside "{voice}.onnx". Falls back to the
        // procedural placeholder model if the weights file isn't actually present (matches the
        // "keep a fake fallback" precedent from Chatterbox/Kokoro).
        string onnxPath = configPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? configPath[..^".json".Length]
            : configPath + ".onnx";

        var model = File.Exists(onnxPath)
            ? new PiperModel(onnxPath, sampleRate: config.Audio.SampleRate)
            : new PiperModel(sampleRate: config.Audio.SampleRate);

        return new PiperPipeline(config, model);
    }

    /// <summary>
    /// Synthesizes input text to 22050Hz audio waveform samples.
    /// </summary>
    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return new AudioGenerationResult([], DefaultSampleRate);
        }

        // 1. Phonemize & Intersperse Pad Tokens
        int[] tokens = _phonemizer.Tokenize(request.Text, interspersePad: true);

        // 2. Resolve Speaker ID
        int speakerId = 0;
        if (_config.SpeakerIdMap != null && _config.SpeakerIdMap.TryGetValue(request.Voice, out int id))
        {
            speakerId = id;
        }

        // 3. Length Scale (inverse of speed)
        float lengthScale = _config.Inference.LengthScale / Math.Max(0.1f, request.Speed);

        // 4. Model Forward Pass
        float[] samples = _model.Forward(
            tokens: tokens,
            speakerId: speakerId,
            noiseScale: _config.Inference.NoiseScale,
            lengthScale: lengthScale,
            noiseW: _config.Inference.NoiseW);

        var result = new AudioGenerationResult(samples, DefaultSampleRate);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    /// <summary>
    /// Synthesizes text in streaming fashion, yielding real-time progressive audio chunks (&lt; 100ms TTFA).
    /// </summary>
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
            yield break;

        int[] tokens = _phonemizer.Tokenize(request.Text, interspersePad: true);
        int speakerId = 0;
        if (_config.SpeakerIdMap != null && _config.SpeakerIdMap.TryGetValue(request.Voice, out int id))
            speakerId = id;

        float lengthScale = _config.Inference.LengthScale / Math.Max(0.1f, request.Speed);

        await foreach (var chunk in _model.ForwardStreamAsync(
            tokens: tokens,
            chunkFrames: 16,
            speakerId: speakerId,
            noiseScale: _config.Inference.NoiseScale,
            lengthScale: lengthScale,
            noiseW: _config.Inference.NoiseW,
            ct: ct))
        {
            yield return chunk;
        }
    }

    public void Dispose()
    {
        _model.Dispose();
    }
}
