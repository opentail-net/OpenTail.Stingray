namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// End-to-end Text-to-Speech synthesis pipeline for Chatterbox-Turbo.
/// </summary>
public sealed class ChatterboxPipeline : ITextToSpeechPipeline
{
    public string Architecture => "Chatterbox-Turbo";
    public int DefaultSampleRate => 24000;

    private readonly ChatterboxTokenizer _tokenizer;
    private readonly ChatterboxAcousticLm _acousticLm;
    private readonly ChatterboxDecoder _decoder;

    public ChatterboxPipeline(
        ChatterboxTokenizer? tokenizer = null,
        ChatterboxAcousticLm? acousticLm = null,
        ChatterboxDecoder? decoder = null)
    {
        _tokenizer = tokenizer ?? new ChatterboxTokenizer();
        _acousticLm = acousticLm ?? new ChatterboxAcousticLm();
        _decoder = decoder ?? new ChatterboxDecoder();
    }

    /// <summary>
    /// Synthesizes text to 24kHz audio samples using Chatterbox-Turbo.
    /// </summary>
    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return new AudioGenerationResult([], DefaultSampleRate);
        }

        // 1. Text Tokenization
        int[] textTokens = _tokenizer.Encode(request.Text);

        // 2. Speaker Feature Bank
        float[] speakerFeatures = ChatterboxVoices.GetSpeakerFeatures(request.Voice);

        // 3. Autoregressive Acoustic LM -> Speech Tokens
        var speechTokens = _acousticLm.GenerateSpeechTokens(
            textTokens: textTokens,
            speakerFeatures: speakerFeatures,
            temperature: 0.7f);

        // 4. Conditional Neural Decoder -> 24kHz PCM Audio
        float[] samples = _decoder.Decode(speechTokens, speakerFeatures);

        var result = new AudioGenerationResult(samples, DefaultSampleRate);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    public void Dispose()
    {
        _acousticLm.Dispose();
    }
}
