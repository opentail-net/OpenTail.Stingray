using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

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

        // 3. Autoregressive Acoustic LM -> Speech Tokens
        var speechTokens = _acousticLm.GenerateSpeechTokens(
            textTokens: textTokens,
            speakerFeatures: speakerFeatures ?? [],
            temperature: 0.7f);

        // 4. Conditional Neural Decoder -> 24kHz PCM Audio
        float[] samples = _decoder.Decode(speechTokens, speakerFeatures ?? []);

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

        var sentences = Regex.Split(request.Text, @"(?<=[.!?,;\n])\s+");
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
        _weights?.Dispose();
        _s3GenWeights?.Dispose();
        _acousticLm.Dispose();
    }
}
