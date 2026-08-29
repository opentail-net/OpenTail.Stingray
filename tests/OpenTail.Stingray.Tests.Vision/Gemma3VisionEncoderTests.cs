
namespace OpenTail.Stingray.Tests.Vision;

public class Gemma3VisionEncoderTests
{
    /// <summary>
    /// Structural sanity check, NOT a parity test -- same caveat as
    /// <see cref="Gemma4VVisionEncoderTests"/>: no working oracle exists on this machine yet for
    /// gemma3 end-to-end numerics. This only asserts the encoder runs against the real mmproj
    /// without crashing, produces the expected shape, and produces finite, non-degenerate output.
    /// At 4096 patches x 27 blocks this is a genuinely heavier computation than gemma4v's 196
    /// patches -- expect this test to take noticeably longer.
    /// </summary>
    // Timeout so a genuinely stuck run fails fast with a clear message instead of hanging
    // indefinitely (found 2026-08-20: a full-suite run needed a manual kill after 30+ minutes with
    // no signal on which test, or whether it was stuck vs. just slow). 4096 patches x 27 blocks is
    // real, heavy, unbounded CPU compute -- 10 minutes is generous headroom over any expected
    // runtime, not a tight bound.
    [Fact(Timeout = 600_000)]
    public void Forward_OnRealGemma3Mmproj_ProducesFiniteExpectedShapeOutput()
    {
        var path = VisionTestPaths.FindGemma3Mmproj();
        Assert.SkipUnless(path is not null,
            "mmproj-gemma-3-4b-it-f16.gguf is required for the Gemma 3 SigLIP encoder sanity check.");

        using var model = Gemma3VisionModel.Open(path!);
        var encoder = new Gemma3VisionEncoder(model);

        // n_merge=4 (verified against the real mmproj), 896/14 = 64x64 patches, exactly divisible
        // -> 16x16 = 256 soft tokens, none dropped (unlike gemma4v's non-divisible 14/3).
        Assert.Equal(4, model.NMerge);
        Assert.Equal(256, encoder.TokenCount);
        // head_dim must be 72 for the per-head attention split to line up.
        Assert.Equal(72, model.EmbeddingLength / model.HeadCount);

        var size = model.ImageSize;
        var rgb = new byte[size * size * 3];
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var i = (y * size + x) * 3;
                var v = (byte)(((x / 32 + y / 32) % 2 == 0) ? 200 : 40);
                rgb[i] = v;
                rgb[i + 1] = (byte)(v / 2);
                rgb[i + 2] = (byte)(255 - v);
            }
        }
        var pre = Gemma3ImagePreprocessor.Preprocess(rgb, size, size, model);

        var output = encoder.Forward(pre.Chw);

        Assert.Equal(encoder.TokenCount * model.ProjectionDim, output.Length);

        var hasNonZero = false;
        var maxAbs = 0f;
        foreach (var value in output)
        {
            Assert.False(float.IsNaN(value), "encoder output contains NaN");
            Assert.False(float.IsInfinity(value), "encoder output contains Inf");
            if (value != 0f) hasNonZero = true;
            maxAbs = MathF.Max(maxAbs, MathF.Abs(value));
        }
        Assert.True(hasNonZero, "encoder output is degenerately all-zero");
        Assert.InRange(maxAbs, 1e-4f, 1e4f);
    }
}
