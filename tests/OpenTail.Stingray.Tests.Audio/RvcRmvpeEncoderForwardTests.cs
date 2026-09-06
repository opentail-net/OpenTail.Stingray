
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real forward-pass sanity test for RvcRmvpeEncoder against real weights, with a
/// synthetic mel input (the real n_fft=1024 log-mel frontend isn't ported yet -- see
/// docs/audio-review-progress.md's RMVPE section -- so this validates the U-Net+GRU network
/// itself runs correctly and produces well-formed output, independent of the frontend).</summary>
public sealed class RvcRmvpeEncoderForwardTests : HeavyTestBase
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
    public void Forward_RealWeights_SyntheticMel_ProducesWellFormedSigmoidOutput()
    {
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcRmvpeWeights(source);

        // Frame count must be a multiple of 32 (2^5, matching the 5 avg-pool-2x2 levels) so the
        // U-Net's spatial dimensions divide evenly at every level.
        const int frames = 64;
        var mel = new float[frames][];
        var rng = new Random(42);
        for (int f = 0; f < frames; f++)
        {
            mel[f] = new float[OpenTail.Stingray.Audio.Rvc.RvcRmvpeWeights.MelBins];
            for (int m = 0; m < mel[f].Length; m++)
                mel[f][m] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        }

        var output = OpenTail.Stingray.Audio.Rvc.RvcRmvpeEncoder.Forward(w, mel);

        Assert.Equal(frames, output.Length);
        Assert.Equal(OpenTail.Stingray.Audio.Rvc.RvcRmvpeWeights.NumPitchClasses, output[0].Length);

        double sum = 0;
        int n = 0;
        double maxAcrossFrames = 0;
        var perFrameVariance = new bool[frames];
        foreach (var (row, idx) in output.Select((r, i) => (r, i)))
        {
            float rowMax = 0f, rowMin = float.MaxValue;
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v), "non-finite value in RMVPE output");
                Assert.InRange(v, 0.0f, 1.0f); // real sigmoid output
                sum += v;
                n++;
                if (v > rowMax) rowMax = v;
                if (v < rowMin) rowMin = v;
            }
            maxAcrossFrames = Math.Max(maxAcrossFrames, rowMax);
            // A real, non-degenerate sigmoid layer varies its output across the 360 classes for
            // a given frame (the classify head is a real Linear projection, not a constant) --
            // rowMax==rowMin would mean every class got IDENTICAL output, a real degeneracy.
            perFrameVariance[idx] = rowMax > rowMin + 1e-6f;
        }
        double mean = sum / n;
        Console.Error.WriteLine($"[RvcRmvpeFwd] frames={output.Length} mean={mean:F4} maxAcrossFrames={maxAcrossFrames:F4}");
        // RMVPE's 360-way sigmoid head is naturally sparse (real pitch training only activates a
        // few adjacent bins), so a near-zero global mean AND a small peak are both EXPECTED,
        // correct behavior for structurally-random, out-of-distribution synthetic input (a real
        // trained network legitimately responds weakly to nonsense audio it never saw in
        // training) -- neither is a reliable signal of a bug with this input, so not asserted
        // numerically. What IS a reliable, input-independent structural check: real per-frame
        // variation across the 360 classes (rules out a degenerate "every class identical"
        // collapse, e.g. from a broken Linear/GRU producing a constant).
        Assert.All(perFrameVariance, v => Assert.True(v, "a frame's 360 class outputs were all identical -- degenerate"));
    }
}
