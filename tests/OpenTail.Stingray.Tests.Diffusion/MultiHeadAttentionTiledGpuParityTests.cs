using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness check for the new tiled ("flash-attention"-style) GPU MultiHeadAttentionTiled
/// shader against the existing, already-trusted CPU reference (DiffusionOps.MultiHeadAttention)
/// -- added 2026-09-11, before wiring into SpatialTransformer, following the same discipline as
/// MultiHeadAttentionGpuParityTests (the naive shader's sibling test). The tiled kernel reorders
/// floating-point accumulation (16-lane shuffle reduction for QK dot products, tile-wise online
/// softmax) versus the CPU's strictly sequential sum, so a looser tolerance than the naive
/// shader's 1e-3f is used here deliberately -- this was called out explicitly by the external
/// design review this kernel is based on. Shapes match this codebase's actual usage (self-
/// attention: qSeq==kvSeq; cross-attention: kvSeq=77, the fixed CLIP context length), plus a
/// qSeq=4096 case exercising multiple 32-row K/V tiles (SDXL's largest self-attention hw).
/// </summary>
public sealed class MultiHeadAttentionTiledGpuParityTests
{
    private static float[] RandomTensor(int n, int seed)
    {
        var rng = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    [Theory]
    [InlineData(16, 16, 4, 64)]     // small self-attention, single tile
    [InlineData(64, 64, 5, 64)]     // SDXL c=320 self-attention shape (nHeads=320/64=5)
    [InlineData(64, 77, 5, 64)]     // SDXL cross-attention shape (kvSeq=77 fixed context length)
    [InlineData(1, 77, 10, 64)]     // single query token, larger nHeads
    [InlineData(4096, 77, 5, 64)]   // SDXL's largest cross-attention query count (64x64 latent)
    [InlineData(1024, 1024, 10, 64)] // large self-attention: multiple Br AND Bc tiles
    public void GpuAttentionTiled_MatchesCpuReference(int qSeq, int kvSeq, int numHeads, int headDim)
    {
        int dim = numHeads * headDim;
        var q = RandomTensor(qSeq * dim, seed: 1);
        var k = RandomTensor(kvSeq * dim, seed: 2);
        var v = RandomTensor(kvSeq * dim, seed: 3);

        var cpuOut = new float[qSeq * dim];
        DiffusionOps.MultiHeadAttention(q, k, v, cpuOut.AsSpan(), qSeq, kvSeq, numHeads, headDim);

        using var backend = new VulkanBackend();
        var qGpu = backend.Upload(q, TensorShape.D1(q.Length));
        var kGpu = backend.Upload(k, TensorShape.D1(k.Length));
        var vGpu = backend.Upload(v, TensorShape.D1(v.Length));
        Tensor? outGpu = null;
        var gpuOut = new float[qSeq * dim];
        try
        {
            outGpu = backend.MultiHeadAttentionTiled(qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
            backend.Download(outGpu, gpuOut);
        }
        finally
        {
            backend.Free(qGpu);
            backend.Free(kGpu);
            backend.Free(vGpu);
            if (outGpu is not null) backend.Free(outGpu);
        }

        float maxAbsDiff = 0f;
        int worstIdx = -1;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float diff = Math.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxAbsDiff) { maxAbsDiff = diff; worstIdx = i; }
        }

        // Tiled reduction reorders float accumulation vs. the CPU's sequential sum -- looser than
        // the naive shader's 1e-3f, but still tight enough to catch a structurally wrong kernel
        // (a real algorithm bug shows up as O(1) error, not O(1e-4)).
        Assert.True(maxAbsDiff < 5e-3f,
            $"Max abs diff {maxAbsDiff} at index {worstIdx} exceeds tolerance (qSeq={qSeq}, kvSeq={kvSeq}, numHeads={numHeads}, headDim={headDim}); cpu={cpuOut[Math.Max(worstIdx,0)]}, gpu={gpuOut[Math.Max(worstIdx,0)]}");
    }
}
