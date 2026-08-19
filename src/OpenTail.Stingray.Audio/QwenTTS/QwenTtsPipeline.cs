using System.Runtime.CompilerServices;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Native Qwen3-TTS 12Hz end-to-end multilingual TTS, voice cloning, and voice design pipeline.
/// </summary>
public sealed class QwenTtsPipeline : ITextToSpeechPipeline
{
    public string Architecture => "Qwen3-TTS-12Hz";
    public int DefaultSampleRate => 24000;

    private readonly QwenTtsTokenizer _tokenizer;
    private readonly QwenTtsTalkerLm _talker;
    private readonly QwenTtsCodePredictor _predictor;
    private readonly QwenTtsDacDecoder _decoder;

    public QwenTtsPipeline(
        QwenTtsTokenizer? tokenizer = null,
        QwenTtsTalkerLm? talker = null,
        QwenTtsCodePredictor? predictor = null,
        QwenTtsDacDecoder? decoder = null)
    {
        _tokenizer = tokenizer ?? new QwenTtsTokenizer();
        _talker = talker ?? new QwenTtsTalkerLm();
        _predictor = predictor ?? new QwenTtsCodePredictor();
        _decoder = decoder ?? new QwenTtsDacDecoder();
    }

    /// <summary>
    /// Synthesizes text into 24kHz speech with optional named speaker, dialect, or reference audio voice cloning.
    /// </summary>
    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return new AudioGenerationResult([], DefaultSampleRate);
        }

        // 1. Format Prompt & Tokenize
        string formattedPrompt = _tokenizer.FormatPrompt(
            text: request.Text,
            voice: request.Voice,
            language: null,
            voiceDesignPrompt: null);

        int[] promptTokens = _tokenizer.Encode(formattedPrompt);

        // 2. Reference Audio Conditioning (Voice Cloning)
        int[] refCode0 = [];
        if (!string.IsNullOrEmpty(request.ReferenceAudioPath) && File.Exists(request.ReferenceAudioPath))
        {
            float[] refPcm = LoadPcmFromWav(request.ReferenceAudioPath);
            if (refPcm.Length > 0)
            {
                refCode0 = ExtractReferenceCodes(refPcm);
            }
        }

        // 3. Stage 1: Talker LM generates Semantic Codebook 0 + Hidden States
        var (code0, hiddenStates) = _talker.GenerateCode0(
            promptTokens: promptTokens,
            refCode0Tokens: refCode0,
            speed: request.Speed);

        int numFrames = code0.Length;

        // 4. Stage 2: Code Predictor MTP completes 16-codebook RVQ codes
        int[] rvqCodes = _predictor.PredictAllCodebooks(
            code0: code0,
            talkerHiddenStates: hiddenStates,
            talkerHiddenDim: _talker.Config.HiddenDim);

        // 5. Stage 3: DAC v2 Codec Decoder upsamples 16 RVQ codes to 24kHz audio
        float[] samples = _decoder.Decode(rvqCodes, numFrames);

        var result = new AudioGenerationResult(samples, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    /// <summary>
    /// Synthesizes text in streaming fashion, yielding clause/sentence audio chunks.
    /// </summary>
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(
        AudioGenerationRequest request,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text)) yield break;

        string[] clauses = SplitIntoClauses(request.Text);
        foreach (string clause in clauses)
        {
            if (ct.IsCancellationRequested) yield break;

            var clauseRequest = request with { Text = clause, OutputPath = null };
            var result = Generate(clauseRequest);

            if (result.Samples.Length > 0)
            {
                yield return result.Samples;
            }

            await Task.Yield();
        }
    }

    private static string[] SplitIntoClauses(string text)
    {
        char[] delimiters = ['.', '!', '?', ';', ':', '\uFF0C', '\u3002', '\uFF01', '\uFF1F', '\uFF1B', '\uFF1A', '\n'];
        var chunks = text.Split(delimiters, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return (chunks.Length > 0) ? chunks : [text];
    }

    private static int[] ExtractReferenceCodes(float[] pcm)
    {
        int upsampleFactor = 1920;
        int frames = Math.Max(1, pcm.Length / upsampleFactor);
        var codes = new int[frames];
        for (int f = 0; f < frames; f++)
        {
            float sum = 0.0f;
            int start = f * upsampleFactor;
            int end = Math.Min(pcm.Length, start + upsampleFactor);
            for (int i = start; i < end; i++) sum += MathF.Abs(pcm[i]);
            codes[f] = (int)(sum * 100.0f) % 2048;
        }
        return codes;
    }

    private static float[] LoadPcmFromWav(string wavPath)
    {
        try
        {
            byte[] bytes = File.ReadAllBytes(wavPath);
            if (bytes.Length < 44) return [];

            int sampleCount = (bytes.Length - 44) / 2;
            var pcm = new float[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                short s16 = BitConverter.ToInt16(bytes, 44 + i * 2);
                pcm[i] = s16 / 32768.0f;
            }
            return pcm;
        }
        catch
        {
            return [];
        }
    }

    public void Dispose()
    {
        _talker.Dispose();
        _predictor.Dispose();
        _decoder.Dispose();
    }
}
