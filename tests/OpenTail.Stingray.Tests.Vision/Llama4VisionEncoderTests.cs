using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

public class Llama4VisionEncoderTests
{
    /// <summary>
    /// Structural sanity check, NOT a parity test -- same caveat as
    /// <see cref="Gemma3VisionEncoderTests"/>/<see cref="Gemma4VVisionEncoderTests"/>: no working
    /// oracle exists on this machine yet for llama4 end-to-end numerics, and llama.cpp's own code
    /// carries a runtime warning that this exact projector is known to have degraded quality
    /// regardless of implementation correctness. This only asserts the encoder runs against the
    /// real mmproj (one fixed-square tile, no multi-tile orchestration) without crashing, produces
    /// the expected shape, and produces finite, non-degenerate output.
    /// </summary>
    [Fact]
    public void Forward_OnRealLlama4Mmproj_ProducesFiniteExpectedShapeOutput()
    {
        var path = VisionTestPaths.FindLlama4Mmproj();
        Assert.SkipUnless(path is not null,
            "mmproj-llama-4-scout-17b-16e-instruct-f16.gguf is required for the Llama 4 ViT encoder sanity check.");

        using var model = Llama4VisionModel.Open(path!);
        var encoder = new Llama4VisionEncoder(model);

        // Verified against the real mmproj: 336/14 = 24x24 patches, scale_factor(n_merge)=2 ->
        // 12x12 = 144 soft tokens, exactly divisible (unlike gemma4v's non-divisible 14/3).
        Assert.Equal(2, model.NMerge);
        Assert.Equal(144, encoder.TokenCount);
        Assert.Equal(88, model.EmbeddingLength / model.HeadCount);
        Assert.Equal(5120, model.ProjectionDim);

        var size = model.ImageSize;
        var rgb = new byte[size * size * 3];
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var i = (y * size + x) * 3;
                var v = (byte)(((x / 24 + y / 24) % 2 == 0) ? 200 : 40);
                rgb[i] = v;
                rgb[i + 1] = (byte)(v / 2);
                rgb[i + 2] = (byte)(255 - v);
            }
        }
        var pre = Llama4ImagePreprocessor.Preprocess(rgb, size, size, model);

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
