
namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// End-to-end Multilingual VITS Text-to-Speech pipeline for MeloTTS.
/// </summary>
public sealed class MeloPipeline : ITextToSpeechPipeline
{
    public string Architecture => "MeloTTS";
    public int DefaultSampleRate => 44100;

    private readonly MeloPhonemizer _phonemizer;
    private readonly MeloBertEncoder _bertEncoder;
    private readonly MeloModel _model;

    public MeloPipeline(
        MeloPhonemizer? phonemizer = null,
        MeloBertEncoder? bertEncoder = null,
        MeloModel? model = null)
    {
        _phonemizer = phonemizer ?? new MeloPhonemizer();
        _bertEncoder = bertEncoder ?? new MeloBertEncoder();
        _model = model ?? new MeloModel(sampleRate: DefaultSampleRate);
    }

    /// <summary>
    /// Loads a MeloTTS pipeline from an ONNX or GGUF model file.
    /// </summary>
    public static MeloPipeline Load(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            throw new FileNotFoundException($"MeloTTS model file not found: {modelPath}");

        var phonemizer = new MeloPhonemizer();
        var bertEncoder = new MeloBertEncoder();
        // NOTE: previously always constructed the fake/placeholder MeloModel here regardless of
        // modelPath -- Load() validated the file exists but never actually wired it into the
        // model, so every "real weights" pipeline run silently used procedural/sinusoidal fake
        // synthesis. Fixed to load real weights the same way PiperPipeline.Load does.
        var model = new MeloModel(modelPath, sampleRate: 44100);

        return new MeloPipeline(phonemizer, bertEncoder, model);
    }

    /// <summary>
    /// Synthesizes multilingual speech to 44.1kHz audio samples.
    /// </summary>
    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return new AudioGenerationResult([], DefaultSampleRate);
        }

        // 1. Multilingual Phonemization (Phones, Tones, Language IDs)
        string voice = string.IsNullOrWhiteSpace(request.Voice) ? "EN-US" : request.Voice;
        var phonemeRes = _phonemizer.Phonemize(request.Text, voice);

        // 2. Phone-Level Context BERT Feature Extraction
        float[] bertFeatures = _bertEncoder.Encode(phonemeRes.Phones, phonemeRes.Tones, phonemeRes.LangIds);

        // 3. Speaker / Accent ID
        int speakerId = MeloVoices.GetSpeakerId(voice);

        // 4. Multilingual VITS Model Forward
        float[] samples = _model.Forward(
            phones: phonemeRes.Phones,
            tones: phonemeRes.Tones,
            langIds: phonemeRes.LangIds,
            bertFeatures: bertFeatures,
            speakerId: speakerId,
            speed: request.Speed,
            sdpRatio: 0.2f,
            noiseScale: 0.333f,
            noiseScaleW: 0.6f);

        // 5. Volume / Peak Normalization (to 0.75 full scale)
        if (samples.Length > 0)
        {
            float maxVal = 0f;
            for (int i = 0; i < samples.Length; i++)
            {
                float a = MathF.Abs(samples[i]);
                if (a > maxVal) maxVal = a;
            }
            if (maxVal > 1e-4f && maxVal < 0.75f)
            {
                float gain = 0.75f / maxVal;
                for (int i = 0; i < samples.Length; i++) samples[i] *= gain;
            }
        }

        var result = new AudioGenerationResult(samples, DefaultSampleRate);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    /// <summary>
    /// Synthesizes text in streaming fashion, yielding clause/sentence audio waveforms as they are generated.
    /// </summary>
    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, CancellationToken ct = default)
        => TtsStreamingHelper.SplitAndGenerateAsync(request, Generate, ct);

    public void Dispose()
    {
        _model.Dispose();
    }
}
