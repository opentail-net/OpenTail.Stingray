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

        // 3. Get Speaker Style Vector. A real trained style vector loaded from a voice GGUF
        // (KokoroModel.Load(modelPath, voicePath)) always wins over KokoroVoices' procedural
        // placeholder presets -- those are seeded-random "calibrated initial style vectors", not
        // real speaker embeddings, and produce garbled/unintelligible speech if ever synthesized
        // against real model weights (the model itself was never trained against them).
        float[] style = _model.HasRealVoiceStyle ? [] : KokoroVoices.GetVoiceStyle(request.Voice);

        // 4. Model Forward Pass
        float[] samples = _model.Forward(tokens, style, request.Speed);

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
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(
        AudioGenerationRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text)) yield break;

        var sentences = System.Text.RegularExpressions.Regex.Split(request.Text, @"(?<=[.!?,
])\s+");
        foreach (var s in sentences)
        {
            var trimmed = s.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            ct.ThrowIfCancellationRequested();

            var req = request with { Text = trimmed, OutputPath = null };
            var res = Generate(req);
            if (res.Samples.Length > 0)
            {
                yield return res.Samples;
            }
            await Task.Yield();
        }
    }

    public void Dispose()
    {
        _model.Dispose();
    }
}
