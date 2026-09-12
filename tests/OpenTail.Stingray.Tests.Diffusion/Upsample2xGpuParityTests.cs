using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// docs/068 Stage B0: isolated numerical parity test for DiffusionOps.Upsample2x (CPU) vs
/// IImageOpsBackend.Upsample2xGpu, verifying identical semantics (coordinate mapping, channel
/// layout, edge behavior) before treating the swap as "just wiring" in VaeDecoder's residency work.
/// </summary>
public sealed class Upsample2xGpuParityTests
{
    [Theory]
    [InlineData(4, 8, 8)]     // small
    [InlineData(512, 64, 64)] // real VAE mid-block resolution/channel count
    [InlineData(128, 256, 256)] // real VAE up.0 resolution/channel count
    public void GpuUpsample2x_MatchesCpuReference(int c, int h, int w)
    {
        var rng = new Random(13);
        var x = new float[c * h * w];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        var expected = DiffusionOps.Upsample2x(x, 1, c, h, w);

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        Tensor? outGpu = null;
        var result = new float[c * h * 2 * w * 2];
        try
        {
            outGpu = backend.Upsample2xGpu(xGpu, c, h, w);
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
        Assert.True(maxAbsDiff < 1e-6f, $"Max abs diff {maxAbsDiff} (c={c}, h={h}, w={w})");
    }
}
