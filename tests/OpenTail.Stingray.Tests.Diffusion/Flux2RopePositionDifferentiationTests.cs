using OpenTail.Stingray.Diffusion.Flux2;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Cheap, no-real-weights numeric check on <see cref="Flux2RoPE"/>: does it actually produce
/// distinguishable cos/sin values for different (h,w) image-token positions? A periodic
/// grid/tiling artifact (found 2026-09-18 after the timestep-blindness fix, docs/087/088) is
/// consistent with a RoPE bug that collapses position information -- this test rules that
/// candidate in or out without needing another expensive real-weight run.
/// </summary>
public sealed class Flux2RopePositionDifferentiationTests
{
    [Fact]
    public void BuildContextFreqs_DifferentHwPositions_ProduceDifferentFrequencies()
    {
        int[] axesDim = [32, 32, 32, 32];
        int headDim = 128;

        // 4 image tokens at distinct (h,w): (0,0), (0,1), (1,0), (3,3) -- t=0, l=0 for all (real
        // image-token convention per prc_img).
        int[] positions =
        [
            0, 0, 0, 0,
            0, 0, 1, 0,
            0, 1, 0, 0,
            0, 3, 3, 0,
        ];
        var (cos, sin) = Flux2RoPE.BuildContextFreqs(positions, nTokens: 4, axesDim: axesDim);

        Assert.Equal(4 * headDim, cos.Length);
        Assert.Equal(4 * headDim, sin.Length);
        Assert.All(cos, v => Assert.True(float.IsFinite(v)));
        Assert.All(sin, v => Assert.True(float.IsFinite(v)));

        // Token 0 is the origin (0,0,0,0) -- its rotation should be identity (cos=1, sin=0)
        // everywhere, a real, checkable invariant of RoPE at position 0.
        for (int d = 0; d < headDim; d++)
        {
            Assert.True(MathF.Abs(cos[d] - 1f) < 1e-5f, $"cos at origin token, dim {d}: expected ~1, got {cos[d]}");
            Assert.True(MathF.Abs(sin[d]) < 1e-5f, $"sin at origin token, dim {d}: expected ~0, got {sin[d]}");
        }

        // Tokens 1 (w=1) and 2 (h=1) must differ from token 0 AND from each other -- if the h/w
        // axes were swapped or collapsed, tokens 1 and 2 (which vary on DIFFERENT axes) would be
        // identical to each other despite differing image positions.
        float diff01 = 0f, diff02 = 0f, diff12 = 0f;
        for (int d = 0; d < headDim; d++)
        {
            diff01 += MathF.Abs(cos[headDim + d] - cos[d]) + MathF.Abs(sin[headDim + d] - sin[d]);
            diff02 += MathF.Abs(cos[2 * headDim + d] - cos[d]) + MathF.Abs(sin[2 * headDim + d] - sin[d]);
            diff12 += MathF.Abs(cos[2 * headDim + d] - cos[headDim + d]) + MathF.Abs(sin[2 * headDim + d] - sin[headDim + d]);
        }

        Assert.True(diff01 > 1e-2f, $"Token 0 (origin) vs token 1 (w=1) must differ, got diff={diff01}");
        Assert.True(diff02 > 1e-2f, $"Token 0 (origin) vs token 2 (h=1) must differ, got diff={diff02}");
        Assert.True(diff12 > 1e-2f, $"Token 1 (w=1) vs token 2 (h=1) must differ (different axes), got diff={diff12}");

        // The w-axis and h-axis frequency BLOCKS must be genuinely different sub-ranges (per the
        // real axesDim=[32,32,32,32] split: dims [0,32)=t, [32,64)=h, [64,96)=w, [96,128)=l).
        // Token 1 varies only on w (dims [64,96)) -- it should be IDENTICAL to token 0 on the
        // h-block [32,64) since h=0 for both, but DIFFERENT on the w-block.
        float hBlockDiff01 = 0f, wBlockDiff01 = 0f;
        for (int d = 32; d < 64; d++) hBlockDiff01 += MathF.Abs(cos[headDim + d] - cos[d]) + MathF.Abs(sin[headDim + d] - sin[d]);
        for (int d = 64; d < 96; d++) wBlockDiff01 += MathF.Abs(cos[headDim + d] - cos[d]) + MathF.Abs(sin[headDim + d] - sin[d]);

        Assert.True(hBlockDiff01 < 1e-5f, $"Token 0 vs token 1: h-block (both h=0) must be IDENTICAL, got diff={hBlockDiff01}");
        Assert.True(wBlockDiff01 > 1e-2f, $"Token 0 vs token 1: w-block (w=0 vs w=1) must DIFFER, got diff={wBlockDiff01}");
    }
}
