
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Structural forward-pass test for RvcSynthesizerEncoder against real weights
/// (voice_v2_default_checkpoint) with synthetic content/pitch input -- not yet golden-verified
/// against the real C++ reference (that needs a real end-to-end run through
/// hubert->rmvpe->synthesizer with a real audio clip, a bigger follow-on step). This test only
/// confirms the forward pass runs to completion and produces finite, non-degenerate audio for a
/// shape/wiring sanity check, the same first step used for RvcRmvpeEncoderForwardTests.</summary>
public sealed class RvcSynthesizerEncoderForwardTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Forward_SyntheticInput_RealWeights_ProducesFiniteNonDegenerateAudio()
    {
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcSynthesizerWeights(source, "voice_v2_default_checkpoint", sampleRate: 40000, v1: false, hasF0: true);

        const int frames = 40;
        const int featureDim = 768; // v2 HuBERT content width
        var rng = new Random(1234);
        var features = new float[frames][];
        var pitchIds = new int[frames];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[featureDim];
            for (int c = 0; c < featureDim; c++) row[c] = (float)(rng.NextDouble() * 0.2 - 0.1);
            features[t] = row;
            pitchIds[t] = 128; // mid-range coarse pitch bin
        }
        var sine = new float[frames * w.HopSamples];
        for (int i = 0; i < sine.Length; i++) sine[i] = MathF.Sin(2 * MathF.PI * 220f * i / 16000f) * 0.1f;

        var audio = OpenTail.Stingray.Audio.Rvc.RvcSynthesizerEncoder.Forward(w, features, pitchIds, sine, speakerId: 0, noiseRng: rng);

        Assert.True(audio.Length > 0);
        double sum = 0, sumSq = 0;
        int nonZero = 0;
        foreach (var v in audio)
        {
            Assert.True(float.IsFinite(v), "generator output contains a non-finite value");
            sum += v;
            sumSq += (double)v * v;
            if (MathF.Abs(v) > 1e-6f) nonZero++;
        }
        double mean = sum / audio.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / audio.Length - mean * mean));
        Console.Error.WriteLine($"[RvcSynthesizer] samples={audio.Length} mean={mean:F5} std={std:F5} nonZero={nonZero}");
        Assert.True(std > 1e-4, "generator output has collapsed to near-silence");
        Assert.True(nonZero > audio.Length / 2, "generator output is mostly zero");
    }
}
