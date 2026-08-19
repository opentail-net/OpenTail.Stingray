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

    public ChatterboxPipeline(
        ChatterboxTokenizer? tokenizer = null,
        ChatterboxAcousticLm? acousticLm = null,
        ChatterboxDecoder? decoder = null,
        ChatterboxWeights? weights = null)
    {
        _weights = weights;
        _tokenizer = tokenizer ?? new ChatterboxTokenizer();
        _acousticLm = acousticLm ?? new ChatterboxAcousticLm(weights);
        _decoder = decoder ?? new ChatterboxDecoder();
    }

    /// <summary>
    /// Loads a real Chatterbox pipeline from GGUF model files (T3 Acoustic LM and optional S3Gen vocoder).
    /// </summary>
    public static ChatterboxPipeline Load(string t3GgufPath, string? s3GenGgufPath = null)
    {
        if (string.IsNullOrWhiteSpace(t3GgufPath) || !File.Exists(t3GgufPath))
            throw new FileNotFoundException($"Chatterbox T3 GGUF model not found: {t3GgufPath}");

        var weights = new ChatterboxWeights(t3GgufPath, s3GenGgufPath);
        var tokenizer = new ChatterboxTokenizer();
        var acousticLm = new ChatterboxAcousticLm(weights);
        var decoder = new ChatterboxDecoder();

        return new ChatterboxPipeline(tokenizer, acousticLm, decoder, weights);
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
        if ((speakerFeatures == null || speakerFeatures.Length == 0) && _weights?.SpeakerEmbedding is { } spk)
        {
            speakerFeatures = spk;
        }

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
        _acousticLm.Dispose();
    }
}
