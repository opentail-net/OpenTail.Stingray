
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
    private readonly ChatterboxWeights? _weights;
    private readonly ChatterboxS3GenWeights? _s3GenWeights;

    public ChatterboxPipeline(
        ChatterboxTokenizer? tokenizer = null,
        ChatterboxAcousticLm? acousticLm = null,
        ChatterboxDecoder? decoder = null,
        ChatterboxWeights? weights = null,
        ChatterboxS3GenWeights? s3GenWeights = null)
    {
        _weights = weights;
        _s3GenWeights = s3GenWeights;
        _tokenizer = tokenizer ?? new ChatterboxTokenizer();
        _acousticLm = acousticLm ?? new ChatterboxAcousticLm(weights);
        _decoder = decoder ?? new ChatterboxDecoder(s3GenWeights, weights);
    }

    /// <summary>
    /// Loads a real Chatterbox pipeline from GGUF model files (T3 Acoustic LM and optional S3Gen vocoder).
    /// </summary>
    public static ChatterboxPipeline Load(string t3GgufPath, string? s3GenGgufPath = null)
    {
        if (string.IsNullOrWhiteSpace(t3GgufPath) || !File.Exists(t3GgufPath))
            throw new FileNotFoundException($"Chatterbox T3 GGUF model not found: {t3GgufPath}");

        if (string.IsNullOrEmpty(s3GenGgufPath))
        {
            var dir = Path.GetDirectoryName(t3GgufPath);
            if (!string.IsNullOrEmpty(dir))
            {
                var candidate = Path.Combine(dir, "chatterbox-turbo-s3gen-q4_k.gguf");
                if (File.Exists(candidate)) s3GenGgufPath = candidate;
            }
            if (string.IsNullOrEmpty(s3GenGgufPath) && File.Exists("models/chatterbox-turbo-s3gen-q4_k.gguf"))
            {
                s3GenGgufPath = "models/chatterbox-turbo-s3gen-q4_k.gguf";
            }
        }

        var weights = new ChatterboxWeights(t3GgufPath, s3GenGgufPath);
        var s3GenWeights = (s3GenGgufPath != null && File.Exists(s3GenGgufPath))
            ? new ChatterboxS3GenWeights(s3GenGgufPath)
            : null;
        var tokenizer = new ChatterboxTokenizer(weights);
        var acousticLm = new ChatterboxAcousticLm(weights);
        var decoder = new ChatterboxDecoder(s3GenWeights, weights);

        return new ChatterboxPipeline(tokenizer, acousticLm, decoder, weights, s3GenWeights);
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

        // 2. Speaker conditioning: real GGUF speaker embedding (the built-in default voice) takes
        // priority when real weights are loaded; the synthetic voice bank is only a fallback for
        // the no-model path (ChatterboxAcousticLm's real T3 inference reads the GGUF embedding
        // directly and ignores this value, but the fake placeholder generator needs it).
        float[] speakerFeatures = _weights?.SpeakerEmbedding ?? ChatterboxVoices.GetSpeakerFeatures(request.Voice);

        bool diag = Environment.GetEnvironmentVariable("STINGRAY_AUDIO_DIAGNOSTIC_DUMP") == "1";
        var sw = diag ? System.Diagnostics.Stopwatch.StartNew() : null;

        // 3. Autoregressive Acoustic LM -> Speech Tokens
        var speechTokens = _acousticLm.GenerateSpeechTokens(
            textTokens: textTokens,
            speakerFeatures: speakerFeatures ?? [],
            temperature: 0.7f,
            maxTokens: Math.Clamp(textTokens.Length * 12, 32, 512));

        if (diag) DiagLog($"T3 speech-token generation: {sw!.ElapsedMilliseconds}ms, {speechTokens.Count} tokens");
        sw?.Restart();

        // 4. Conditional Neural Decoder -> 24kHz PCM Audio
        float[] samples = _decoder.Decode(speechTokens, speakerFeatures ?? []);

        if (diag) DiagLog($"S3Gen decode (encoder+CFM+vocoder): {sw!.ElapsedMilliseconds}ms, {samples.Length} samples");

        // 5. Volume / Peak Normalization
        if (samples.Length > 0)
        {
            float maxVal = 0f;
            for (int i = 0; i < samples.Length; i++)
            {
                float a = MathF.Abs(samples[i]);
                if (a > maxVal) maxVal = a;
            }
            if (maxVal > 1e-4f)
            {
                float targetPeak = 0.85f;
                float gain = MathF.Min(targetPeak / maxVal, 5.0f);
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
        _weights?.Dispose();
        _s3GenWeights?.Dispose();
        _acousticLm.Dispose();
        _decoder.Dispose();
    }

    /// <summary>
    /// Diagnostic logging for Generate() timing breakdowns (STINGRAY_AUDIO_DIAGNOSTIC_DUMP=1).
    /// Written to both stderr and a file, since xUnit/Microsoft.Testing.Platform only surfaces
    /// captured console output for failing tests -- a file is the only way to see this for a
    /// passing run without deliberately breaking the test.
    /// </summary>
    internal static void DiagLog(string message)
    {
        string line = $"[ChatterboxDiag] {message}";
        Console.Error.WriteLine(line);
        try
        {
            File.AppendAllText(Path.Combine(Path.GetTempPath(), "stingray-chatterbox-diag.log"), line + Environment.NewLine);
        }
        catch
        {
            // Best-effort only -- never let diagnostic logging break real synthesis.
        }
    }
}
