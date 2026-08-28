using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.CosyVoice;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies <see cref="CamPlusSpeakerEncoder"/> produces a real, non-degenerate 192-dim x-vector
/// from real audio via the checkpoint's own `campplus.onnx` -- not just "runs without throwing."
/// Part of unblocking CosyVoice3's zero-shot voice conditioning (see
/// docs/audio-review-progress.md's CosyVoice3 entries; previously an all-zero placeholder).
/// </summary>
public sealed class CamPlusSpeakerEncoderTests : HeavyTestBase
{
    private static string? FindModelPath(string relative)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relative);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Extract_RealAudio_ProducesNonDegenerateEmbedding()
    {
        string? onnxPath = FindModelPath("models/campplus.onnx");
        string? wavPath = FindModelPath("docs/audio-samples/fishspeech-s2pro-fixed.wav");
        Assert.SkipUnless(onnxPath != null && wavPath != null, "campplus.onnx or a real sample WAV not found");

        var (samples, sr, _) = WavReader.ReadWav(wavPath!);
        if (sr != CamPlusSpeakerEncoder.SampleRate)
            samples = AudioResampler.Resample(samples, sr, CamPlusSpeakerEncoder.SampleRate);

        var emb = CamPlusSpeakerEncoder.Extract(onnxPath!, samples);

        Assert.NotNull(emb);
        Assert.Equal(192, emb!.Length);
        foreach (var v in emb) Assert.True(float.IsFinite(v), "embedding value was not finite");

        double sumSq = emb.Sum(v => (double)v * v);
        Assert.True(sumSq > 1e-6, $"embedding appears degenerate (near-zero), sumSq={sumSq}");
    }

    [Fact]
    public void ExtractFbank_RealAudio_MatchesCamPlusInputShape()
    {
        string? wavPath = FindModelPath("docs/audio-samples/fishspeech-s2pro-fixed.wav");
        Assert.SkipUnless(wavPath != null, "sample WAV not found");

        var (samples, sr, _) = WavReader.ReadWav(wavPath!);
        if (sr != CamPlusSpeakerEncoder.SampleRate)
            samples = AudioResampler.Resample(samples, sr, CamPlusSpeakerEncoder.SampleRate);

        var feat = CamPlusSpeakerEncoder.ExtractFbank(samples);

        Assert.NotEmpty(feat);
        Assert.Equal(0, feat.Length % CamPlusSpeakerEncoder.NumMelBins);
        foreach (var v in feat) Assert.True(float.IsFinite(v), "fbank feature value was not finite");
    }
}
