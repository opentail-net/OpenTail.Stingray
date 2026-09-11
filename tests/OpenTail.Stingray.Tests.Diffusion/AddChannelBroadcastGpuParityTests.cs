using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness check for the new AddChannelBroadcastInPlace GPU op (docs/067's Stage 1 primitive
/// layer) -- added specifically for SDXL ResBlock's timestep-embedding injection, which broadcasts
/// a per-channel [C] vector additively across every spatial position of a [C,H,W] tensor.
/// </summary>
public sealed class AddChannelBroadcastGpuParityTests
{
    [Theory]
    [InlineData(4, 16)]     // small
    [InlineData(320, 4096)] // real SDXL shape: 320 channels at 64x64 spatial
    [InlineData(1280, 64)]  // real SDXL shape: 1280 channels at 8x8 spatial
    public void GpuBroadcastAdd_MatchesCpuReference(int c, int hw)
    {
        var rng = new Random(7);
        var x = new float[c * hw];
        var bias = new float[c];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);
        foreach (ref var v in bias.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        var expected = (float[])x.Clone();
        for (int ch = 0; ch < c; ch++)
        {
            var slice = expected.AsSpan(ch * hw, hw);
            for (int i = 0; i < hw; i++) slice[i] += bias[ch];
        }

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        var biasGpu = backend.Upload(bias, TensorShape.D1(bias.Length));
        var result = new float[c * hw];
        try
        {
            backend.AddChannelBroadcastInPlace(xGpu, biasGpu, c, hw);
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

        Assert.True(maxAbsDiff < 1e-4f, $"Max abs diff {maxAbsDiff} (c={c}, hw={hw})");
    }
}
