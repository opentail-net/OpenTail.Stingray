using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Parakeet;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for the FastConformer encoder port (see
/// docs/audio-review-progress.md's Parakeet section). NOT yet golden-verified against a real
/// oracle (no crispasr build/dump done yet) -- these checks confirm the real GGUF weights load,
/// the forward pass runs end-to-end without NaN/Inf, and shapes match the checkpoint's own
/// metadata, which is the same bar every other pipeline's "does it even run for real" checkpoint
/// passed before golden verification followed in a later iteration.
/// </summary>
public sealed class ParakeetConformerEncoderTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ParakeetWeights_LoadsRealGguf_AllTensorsPresent()
    {
        string? path = FindRepoFile("models/parakeet-ctc-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/parakeet-ctc-0.6b-q4_k.gguf not found");

        using var w = new ParakeetWeights(path!);
        Assert.Equal(24, w.NumLayers);
        Assert.Equal(1024, w.HiddenDim);
        Assert.Equal(8, w.NumHeads);
        Assert.Equal(128, w.HeadDim);
        Assert.Equal(80, w.NMels);
        Assert.Equal(24, w.Layers.Length);

        foreach (var l in w.Layers)
        {
            Assert.Equal(9, w.ConvKernel);
            foreach (var v in l.ConvDwBias) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
            foreach (var v in l.ConvDwWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
    }

    [Fact]
    public void ParakeetConformerEncoder_Forward_RealWeights_ProducesFiniteOutput()
    {
        string? path = FindRepoFile("models/parakeet-ctc-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/parakeet-ctc-0.6b-q4_k.gguf not found");

        using var w = new ParakeetWeights(path!);

        // Small synthetic mel input (real ParakeetMelExtractor output would also work, this
        // isolates encoder correctness from mel-extraction correctness -- same bisection
        // philosophy used throughout this doc's other pipelines).
        int tMel = 64; // -> ~8 encoder frames after 8x subsampling
        var mel = new float[tMel * w.NMels];
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.1f * MathF.Sin(i * 0.05f);

        var (hidden, ctcLogits, tEnc) = ParakeetConformerEncoder.Forward(w, mel, tMel);

        Assert.True(tEnc > 0);
        Assert.Equal(tEnc, hidden.Length);
        Assert.Equal(tEnc, ctcLogits.Length);

        foreach (var row in hidden)
        {
            Assert.Equal(w.HiddenDim, row.Length);
            foreach (var v in row) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
        foreach (var row in ctcLogits)
        {
            Assert.Equal(w.VocabSize + 1, row.Length);
            foreach (var v in row) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
    }

    [Fact]
    public void ParakeetPipeline_Load_RealWeights_TranscribesWithoutCrashing()
    {
        string? path = FindRepoFile("models/parakeet-ctc-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/parakeet-ctc-0.6b-q4_k.gguf not found");

        using var pipeline = ParakeetPipeline.Load(path!);

        int sampleRate = 16000;
        int numSamples = sampleRate; // 1 second
        var pcm = new float[numSamples];
        for (int i = 0; i < numSamples; i++)
            pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * 300.0f * i / sampleRate);

        var request = new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = sampleRate,
            Language = "en"
        };

        var result = pipeline.Transcribe(request);

        Assert.NotNull(result);
        Assert.NotNull(result.Segments);
    }
}
