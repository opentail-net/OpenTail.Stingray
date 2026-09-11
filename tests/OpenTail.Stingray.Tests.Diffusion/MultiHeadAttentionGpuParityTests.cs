using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness check for the new GPU MultiHeadAttention shader (online-softmax, one thread per
/// query/head pair) against the existing, already-trusted CPU reference (DiffusionOps.
/// MultiHeadAttention) -- added 2026-09-11 before wiring this kernel into SpatialTransformer,
/// specifically because a subtly wrong attention kernel would be silently-plausible-looking, the
/// exact failure class already found twice this session (the CFG-embedding bug, the Z-Image-Turbo
/// regression). Real random data, several shapes matching this codebase's actual usage
/// (self-attention: qSeq==kvSeq; cross-attention: kvSeq=77, the fixed CLIP context length).
/// </summary>
public sealed class MultiHeadAttentionGpuParityTests
{
    private static float[] RandomTensor(int n, int seed)
    {
        var rng = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    [Theory]
    [InlineData(16, 16, 4, 64)]   // small self-attention
    [InlineData(64, 64, 5, 64)]   // SDXL c=320 self-attention shape (nHeads=320/64=5)
    [InlineData(64, 77, 5, 64)]   // SDXL cross-attention shape (kvSeq=77 fixed context length)
    [InlineData(1, 77, 10, 64)]   // single query token, larger nHeads
    public void GpuAttention_MatchesCpuReference(int qSeq, int kvSeq, int numHeads, int headDim)
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
            outGpu = backend.MultiHeadAttention(qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
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
        for (int i = 0; i < cpuOut.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(cpuOut[i] - gpuOut[i]));

        // fp32 GPU vs fp32 CPU, same math (stable softmax either way) -- expect near-exact
        // agreement, not just "same ballpark". A real bug in the online-softmax algorithm would
        // show up as a large, structural difference, not a small float-rounding one.
        Assert.True(maxAbsDiff < 1e-3f, $"Max abs diff {maxAbsDiff} exceeds tolerance (qSeq={qSeq}, kvSeq={kvSeq}, numHeads={numHeads}, headDim={headDim})");
    }
}
