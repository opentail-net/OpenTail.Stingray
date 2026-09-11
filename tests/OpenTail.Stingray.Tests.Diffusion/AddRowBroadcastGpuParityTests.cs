using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness check for AddRowBroadcastInPlace (docs/067's Stage 1 primitive layer) -- the
/// standard Linear-layer bias-add for a GPU-resident [N,D] Lin()/GEMM output, distinct from
/// AddChannelBroadcastInPlace's transposed [C,H,W] channel broadcast.
/// </summary>
public sealed class AddRowBroadcastGpuParityTests
{
    [Theory]
    [InlineData(4, 8)]      // small
    [InlineData(4096, 320)] // real SDXL self-attention shape: hw=4096 tokens, c=320
    [InlineData(77, 2048)]  // real SDXL cross-attention context shape
    public void GpuRowBroadcastAdd_MatchesCpuReference(int n, int d)
    {
        var rng = new Random(11);
        var x = new float[n * d];
        var bias = new float[d];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);
        foreach (ref var v in bias.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        var expected = (float[])x.Clone();
        for (int row = 0; row < n; row++)
        {
            var slice = expected.AsSpan(row * d, d);
            for (int col = 0; col < d; col++) slice[col] += bias[col];
        }

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        var biasGpu = backend.Upload(bias, TensorShape.D1(bias.Length));
        var result = new float[n * d];
        try
        {
            backend.AddRowBroadcastInPlace(xGpu, biasGpu, n, d);
            backend.Download(xGpu, result);
        }
        finally
        {
            backend.Free(xGpu);
            backend.Free(biasGpu);
        }

        float maxAbsDiff = 0f;
        for (int i = 0; i < result.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(result[i] - expected[i]));

        Assert.True(maxAbsDiff < 1e-4f, $"Max abs diff {maxAbsDiff} (n={n}, d={d})");
    }
}
