namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// End-to-end Text-to-Speech synthesis pipeline for Kokoro-82M.
/// </summary>
public sealed class KokoroPipeline : ITextToSpeechPipeline
{
    public string Architecture => "Kokoro-82M";
    public int DefaultSampleRate => 24000;

    private readonly KokoroPhonemizer _phonemizer;
    private readonly KokoroModel _model;

    public KokoroPipeline(KokoroModel? model = null, KokoroPhonemizer? phonemizer = null)
    {
        _model = model ?? new KokoroModel();
        _phonemizer = phonemizer ?? new KokoroPhonemizer();
    }

    /// <summary>
    /// Synthesizes input text to 24kHz audio waveform samples.
    /// </summary>
    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return new AudioGenerationResult([], DefaultSampleRate);
        }

        // 1. Text to Phonemes
        string phonemes = _phonemizer.TextToPhonemes(request.Text);

        // 2. Phonemes to Token IDs
        int[] tokens = _phonemizer.Tokenize(phonemes);

        // 3. Get Speaker Style Vector
        float[] style = KokoroVoices.GetVoiceStyle(request.Voice);

        // 4. Model Forward Pass
        float[] samples = _model.Forward(tokens, style, request.Speed);

        var result = new AudioGenerationResult(samples, DefaultSampleRate);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    public void Dispose()
    {
        _model.Dispose();
    }
}
