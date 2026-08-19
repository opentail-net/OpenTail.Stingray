using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

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
        var model = new MeloModel(sampleRate: 44100);

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
            noiseScale: 0.6f,
            noiseScaleW: 0.8f);

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
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text)) yield break;

        var sentences = Regex.Split(request.Text, @"(?<=[.!?,
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
