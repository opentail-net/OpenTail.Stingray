
namespace OpenTail.Stingray.Tests.Vision;

public class Gemma4VVisionEncoderTests
{
    /// <summary>
    /// Structural sanity check, NOT a parity test: there is still no working oracle on this
    /// machine for gemma4v end-to-end numerics (the paired gemma4 text architecture isn't admitted
    /// by the local llama.cpp build -- see docs/03-gemma4-e4b-vision-plan.md). This only asserts
    /// the encoder runs end-to-end against the real mmproj without crashing, produces the expected
    /// shape, and produces finite, non-degenerate output -- the same class of check this repo has
    /// used elsewhere (e.g. the DeepSeek2 MLA work) when a real reference isn't available yet.
    /// </summary>
    // See Gemma3VisionEncoderTests' Timeout comment -- same reasoning, lighter grid (196 patches)
    // so a shorter bound is still generous headroom.
    [Fact(Timeout = 300_000)]
    public void Forward_OnRealE4BMmproj_ProducesFiniteExpectedShapeOutput()
    {
        var path = VisionTestPaths.FindE4BMmproj();
        Assert.SkipUnless(path is not null,
            "gemma-4-E4B-it-mmproj.gguf is required for the E4B ViT encoder sanity check.");

        using var model = Gemma4VVisionModel.Open(path!);
        var encoder = new Gemma4VVisionEncoder(model);

        // n_merge=3 (verified against the real mmproj -- see docs/03), 224/16 = 14x14 patches ->
        // floor((14-3)/3)+1 = 4 per side -> 16 soft tokens.
        Assert.Equal(3, model.NMerge);
        Assert.Equal(16, encoder.TokenCount);
        // 2D-RoPE frequency base: gemma4v-specific hardcoded constant (100), not the paired text
        // model's own rope theta. Locked here so an edit can't silently drift it.
        Assert.Equal(100f, Gemma4VVisionModel.RopeTheta);
        // head_dim must be 64 for the per-head QK/V-norm split and the 2D-RoPE half-split (32+32)
        // to line up; both are already implicit in EmbeddingLength/HeadCount, asserted explicitly
        // here since the encoder's correctness depends on it directly.
        Assert.Equal(64, model.EmbeddingLength / model.HeadCount);

        // Synthetic RGB checkerboard, already at the encoder's declared fixed input size --
        // exercises the real preprocessor's resize/normalise path with a deterministic, non-trivial
        // (not solid-color) input.
        var rgb = new byte[224 * 224 * 3];
        for (var y = 0; y < 224; y++)
        {
            for (var x = 0; x < 224; x++)
            {
                var i = (y * 224 + x) * 3;
                var v = (byte)(((x / 16 + y / 16) % 2 == 0) ? 200 : 40);
                rgb[i] = v;
                rgb[i + 1] = (byte)(v / 2);
                rgb[i + 2] = (byte)(255 - v);
            }
        }
        var pre = Gemma4VImagePreprocessor.Preprocess(rgb, 224, 224, model);

        var output = encoder.Forward(pre.Chw);

        Assert.Equal(encoder.TokenCount * model.ProjectionDim, output.Length);

        var hasNonZero = false;
        foreach (var value in output)
        {
            Assert.False(float.IsNaN(value), "encoder output contains NaN");
            Assert.False(float.IsInfinity(value), "encoder output contains Inf");
            if (value != 0f) hasNonZero = true;
        }
        Assert.True(hasNonZero, "encoder output is degenerately all-zero");

        // Coarse magnitude sanity beyond "not all zero": catches the class of bug (a missing or
        // doubled clamp, a wrong RoPE base blowing up attention) that produces finite, non-zero,
        // but wildly-scaled output. Not a parity bound -- just rules out an obviously exploded or
        // collapsed pipeline.
        var maxAbs = 0f;
        foreach (var value in output) maxAbs = MathF.Max(maxAbs, MathF.Abs(value));
        Assert.InRange(maxAbs, 1e-4f, 1e4f);
    }
}
