using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness checks for the 4 new GPU primitives added for docs/067's Stage 3a
/// (SpatialTransformer residency): LayerNormGpu, GeGlu, PermuteChwToHwc, PermuteHwcToChw.
/// </summary>
public sealed class SpatialTransformerGpuPrimitivesParityTests
{
    [Theory]
    [InlineData(1, 8)]
    [InlineData(4096, 320)] // real SDXL self-attention shape
    [InlineData(77, 2048)]  // real SDXL cross-attention context shape
    public void LayerNormGpu_MatchesCpuReference(int n, int c)
    {
        var rng = new Random(3);
        var x = new float[n * c];
        var w = new float[c];
        var b = new float[c];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);
        foreach (ref var v in w.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);
        foreach (ref var v in b.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        var expected = (float[])x.Clone();
        DiffusionOps.LayerNorm(expected, w, b, c);

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        var wGpu = backend.Upload(w, TensorShape.D1(w.Length));
        var bGpu = backend.Upload(b, TensorShape.D1(b.Length));
        Tensor? outGpu = null;
        var result = new float[n * c];
        try
        {
            outGpu = backend.LayerNormGpu(xGpu, wGpu, bGpu, n, c);
            backend.Download(outGpu, result);
        }
        finally
        {
            backend.Free(xGpu);
            backend.Free(wGpu);
            backend.Free(bGpu);
            if (outGpu is not null) backend.Free(outGpu);
        }

        float maxAbsDiff = 0f;
        for (int i = 0; i < result.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(result[i] - expected[i]));
        Assert.True(maxAbsDiff < 1e-3f, $"Max abs diff {maxAbsDiff} (n={n}, c={c})");
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(4096, 1280)] // real SDXL FFN shape: mlpDim = c*4 at c=320
    public void GeGlu_MatchesCpuReference(int n, int d)
    {
        var rng = new Random(5);
        var x = new float[n * 2 * d];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[n * d];
        for (int row = 0; row < n; row++)
        {
            int srcOff = row * 2 * d;
            int dstOff = row * d;
            for (int m = 0; m < d; m++)
            {
                float val = x[srcOff + m];
                float gate = x[srcOff + d + m];
                float geluGate = 0.5f * gate * (1.0f + MathF.Tanh(0.79788456f * (gate + 0.044715f * gate * gate * gate)));
                expected[dstOff + m] = val * geluGate;
            }
        }

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        Tensor? outGpu = null;
        var result = new float[n * d];
        try
        {
            outGpu = backend.GeGlu(xGpu, n, d);
            backend.Download(outGpu, result);
        }
        finally
        {
            backend.Free(xGpu);
            if (outGpu is not null) backend.Free(outGpu);
        }

        float maxAbsDiff = 0f;
        for (int i = 0; i < result.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(result[i] - expected[i]));
        Assert.True(maxAbsDiff < 1e-3f, $"Max abs diff {maxAbsDiff} (n={n}, d={d})");
    }

    [Theory]
    [InlineData(4, 8)]
    [InlineData(320, 4096)] // real SDXL shape
    public void PermuteChwToHwcAndBack_RoundTrips(int c, int hw)
    {
        var rng = new Random(9);
        var x = new float[c * hw];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        var expectedHwc = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
            for (int s = 0; s < hw; s++)
                expectedHwc[s * c + ch] = x[ch * hw + s];

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        Tensor? hwcGpu = null, chwGpu = null;
        var hwcResult = new float[hw * c];
        var chwResult = new float[c * hw];
        try
        {
            hwcGpu = backend.PermuteChwToHwc(xGpu, c, hw);
            backend.Download(hwcGpu, hwcResult);
            chwGpu = backend.PermuteHwcToChw(hwcGpu, c, hw);
            backend.Download(chwGpu, chwResult);
        }
        finally
        {
            backend.Free(xGpu);
            if (hwcGpu is not null) backend.Free(hwcGpu);
            if (chwGpu is not null) backend.Free(chwGpu);
        }

        float maxAbsDiff1 = 0f;
        for (int i = 0; i < hwcResult.Length; i++)
            maxAbsDiff1 = Math.Max(maxAbsDiff1, Math.Abs(hwcResult[i] - expectedHwc[i]));
        Assert.True(maxAbsDiff1 < 1e-6f, $"CHW->HWC max abs diff {maxAbsDiff1}");

        float maxAbsDiff2 = 0f;
        for (int i = 0; i < chwResult.Length; i++)
            maxAbsDiff2 = Math.Max(maxAbsDiff2, Math.Abs(chwResult[i] - x[i]));
        Assert.True(maxAbsDiff2 < 1e-6f, $"HWC->CHW round-trip max abs diff {maxAbsDiff2}");
    }
}
