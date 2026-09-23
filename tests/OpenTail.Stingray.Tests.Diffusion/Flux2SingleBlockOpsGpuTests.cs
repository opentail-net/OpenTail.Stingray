using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// GPU parity tests for FLUX.2 SingleStreamBlock custom kernels:
///   1. <see cref="VulkanBackend.Flux2SingleUnpackNormRope"/>
///   2. <see cref="VulkanBackend.Flux2SingleConcatAttnMlp"/>
/// </summary>
public sealed class Flux2SingleBlockOpsGpuTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Theory]
    [InlineData(4, 4, 8)]    // small synthetic: nSeq=4, numHeads=4, mlpHidden=8*4*128
    [InlineData(8, 48, 18432)] // production dimensions: 48 heads, dim=6144, mlpHidden=18432
    public void Flux2SingleUnpackNormRope_MatchesReference(int nSeq, int numHeads, int mlpHidden)
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int headDim = 128;
        int dim = numHeads * headDim;
        int rowStride = 3 * dim + 2 * mlpHidden;
        int nPairs = headDim / 2;

        var rng = new Random(101 + nSeq * 37 + numHeads * 19);

        var lin1 = new float[nSeq * rowStride];
        for (int i = 0; i < lin1.Length; i++) lin1[i] = (float)(rng.NextDouble() * 2 - 1);

        var qScale = new float[headDim];
        var kScale = new float[headDim];
        for (int i = 0; i < headDim; i++)
        {
            qScale[i] = (float)(rng.NextDouble() * 1.5 + 0.5);
            kScale[i] = (float)(rng.NextDouble() * 1.5 + 0.5);
        }

        var cos = new float[nSeq * nPairs];
        var sin = new float[nSeq * nPairs];
        for (int i = 0; i < cos.Length; i++)
        {
            float theta = (float)(rng.NextDouble() * Math.PI * 2);
            cos[i] = MathF.Cos(theta);
            sin[i] = MathF.Sin(theta);
        }

        // --- CPU Reference ---
        var qRef = new float[nSeq * dim];
        var kRef = new float[nSeq * dim];
        var vRef = new float[nSeq * dim];

        for (int t = 0; t < nSeq; t++)
        {
            int srcRow = t * rowStride;
            int dstRow = t * dim;

            // Unpack Q, K, V
            Array.Copy(lin1, srcRow, qRef, dstRow, dim);
            Array.Copy(lin1, srcRow + dim, kRef, dstRow, dim);
            Array.Copy(lin1, srcRow + dim * 2, vRef, dstRow, dim);

            // Per-head RMSNorm
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = dstRow + h * headDim;

                float sqQ = 0f, sqK = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    sqQ += qRef[hOff + d] * qRef[hOff + d];
                    sqK += kRef[hOff + d] * kRef[hOff + d];
                }
                float invStdQ = 1.0f / MathF.Sqrt(sqQ / headDim + 1e-6f);
                float invStdK = 1.0f / MathF.Sqrt(sqK / headDim + 1e-6f);

                for (int d = 0; d < headDim; d++)
                {
                    qRef[hOff + d] = qRef[hOff + d] * invStdQ * qScale[d];
                    kRef[hOff + d] = kRef[hOff + d] * invStdK * kScale[d];
                }

                // RoPE
                for (int p = 0; p < nPairs; p++)
                {
                    int d0 = p * 2;
                    int d1 = d0 + 1;
                    float c = cos[t * nPairs + p];
                    float s = sin[t * nPairs + p];

                    float q0 = qRef[hOff + d0];
                    float q1 = qRef[hOff + d1];
                    qRef[hOff + d0] = q0 * c - q1 * s;
                    qRef[hOff + d1] = q0 * s + q1 * c;

                    float k0 = kRef[hOff + d0];
                    float k1 = kRef[hOff + d1];
                    kRef[hOff + d0] = k0 * c - k1 * s;
                    kRef[hOff + d1] = k0 * s + k1 * c;
                }
            }
        }

        // --- GPU Execution ---
        using var lin1Gpu = backend.Upload(lin1, TensorShape.D2(nSeq, rowStride), exact: true);
        using var qGpu = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var kGpu = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var vGpu = backend.Allocate(TensorShape.D2(nSeq, dim));
        using var qScaleGpu = backend.Upload(qScale, TensorShape.D1(headDim), exact: true);
        using var kScaleGpu = backend.Upload(kScale, TensorShape.D1(headDim), exact: true);
        using var cosGpu = backend.Upload(cos, TensorShape.D2(nSeq, nPairs), exact: true);
        using var sinGpu = backend.Upload(sin, TensorShape.D2(nSeq, nPairs), exact: true);

        backend.Flux2SingleUnpackNormRope(lin1Gpu, qGpu, kGpu, vGpu, cosGpu, sinGpu,
            qScaleGpu, kScaleGpu, nSeq, numHeads, headDim, rowStride, eps: 1e-6f);
        backend.Synchronize();

        var qOut = new float[nSeq * dim];
        var kOut = new float[nSeq * dim];
        var vOut = new float[nSeq * dim];
        backend.Download(qGpu, qOut);
        backend.Download(kGpu, kOut);
        backend.Download(vGpu, vOut);

        // V should be bit-exact
        for (int i = 0; i < vRef.Length; i++)
            Assert.Equal(vRef[i], vOut[i]);

        // Q and K should match within float precision tolerance (< 2e-5)
        for (int i = 0; i < qRef.Length; i++)
        {
            float diff = MathF.Abs(qRef[i] - qOut[i]);
            Assert.True(diff < 2e-5f, $"Q mismatch at {i}: ref={qRef[i]} gpu={qOut[i]} diff={diff}");
        }
        for (int i = 0; i < kRef.Length; i++)
        {
            float diff = MathF.Abs(kRef[i] - kOut[i]);
            Assert.True(diff < 2e-5f, $"K mismatch at {i}: ref={kRef[i]} gpu={kOut[i]} diff={diff}");
        }
    }

    [Theory]
    [InlineData(4, 64, 128)]
    [InlineData(8, 6144, 18432)] // production scale
    public void Flux2SingleConcatAttnMlp_MatchesReference(int nSeq, int dim, int mlpHidden)
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        int rowStride = 3 * dim + 2 * mlpHidden;
        int outRowWidth = dim + mlpHidden;

        var rng = new Random(202 + nSeq * 41 + dim);

        var attnOut = new float[nSeq * dim];
        for (int i = 0; i < attnOut.Length; i++) attnOut[i] = (float)(rng.NextDouble() * 2 - 1);

        var lin1 = new float[nSeq * rowStride];
        for (int i = 0; i < lin1.Length; i++) lin1[i] = (float)(rng.NextDouble() * 2 - 1);

        // --- CPU Reference ---
        var expected = new float[nSeq * outRowWidth];
        for (int t = 0; t < nSeq; t++)
        {
            // 1. Copy AttnOut
            Array.Copy(attnOut, t * dim, expected, t * outRowWidth, dim);

            // 2. Gated MLP
            int mlpBase = t * rowStride + 3 * dim;
            for (int m = 0; m < mlpHidden; m++)
            {
                float gate = lin1[mlpBase + m];
                float val = lin1[mlpBase + mlpHidden + m];
                float silu = gate / (1.0f + MathF.Exp(-gate));
                expected[t * outRowWidth + dim + m] = silu * val;
            }
        }

        // --- GPU Execution ---
        using var attnGpu = backend.Upload(attnOut, TensorShape.D2(nSeq, dim), exact: true);
        using var lin1Gpu = backend.Upload(lin1, TensorShape.D2(nSeq, rowStride), exact: true);
        using var lin2InGpu = backend.Allocate(TensorShape.D2(nSeq, outRowWidth));

        backend.Flux2SingleConcatAttnMlp(attnGpu, lin1Gpu, lin2InGpu, nSeq, dim, mlpHidden, rowStride);
        backend.Synchronize();

        var actual = new float[nSeq * outRowWidth];
        backend.Download(lin2InGpu, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff < 2e-5f, $"Mismatch at {i}: expected={expected[i]} actual={actual[i]} diff={diff}");
        }
    }
}
