using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Correctness and parity checks for FLUX GPU compute primitives:
/// Flux2DRoPE, FluxUnpackQkv, FluxUnpackSingleLin1, FluxConcatAttnMlp, FluxConcatTxtImg, FluxSliceImg, FluxEulerStep.
/// </summary>
public sealed class FluxGpuParityTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void FluxUnpackQkv_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int n = 64;
        const int d = 128;
        var rng = new Random(42);
        var qkv = new float[n * 3 * d];
        for (int i = 0; i < qkv.Length; i++) qkv[i] = (float)(rng.NextDouble() * 2 - 1);

        var qExp = new float[n * d];
        var kExp = new float[n * d];
        var vExp = new float[n * d];
        for (int t = 0; t < n; t++)
        {
            int srcOff = t * 3 * d;
            int dstOff = t * d;
            Array.Copy(qkv, srcOff, qExp, dstOff, d);
            Array.Copy(qkv, srcOff + d, kExp, dstOff, d);
            Array.Copy(qkv, srcOff + 2 * d, vExp, dstOff, d);
        }

        var qkvGpu = backend.Upload(qkv, TensorShape.D1(qkv.Length));
        var qGpu = backend.Allocate(TensorShape.D1(n * d));
        var kGpu = backend.Allocate(TensorShape.D1(n * d));
        var vGpu = backend.Allocate(TensorShape.D1(n * d));

        try
        {
            backend.FluxUnpackQkv(qkvGpu, qGpu, kGpu, vGpu, n, d, dstTokenOffset: 0);

            var qAct = new float[n * d];
            var kAct = new float[n * d];
            var vAct = new float[n * d];
            backend.Download(qGpu, qAct);
            backend.Download(kGpu, kAct);
            backend.Download(vGpu, vAct);

            for (int i = 0; i < n * d; i++)
            {
                Assert.Equal(qExp[i], qAct[i], 5);
                Assert.Equal(kExp[i], kAct[i], 5);
                Assert.Equal(vExp[i], vAct[i], 5);
            }
        }
        finally
        {
            backend.Free(qkvGpu);
            backend.Free(qGpu);
            backend.Free(kGpu);
            backend.Free(vGpu);
        }
    }

    [Fact]
    public void FluxConcatAttnMlp_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int n = 32;
        const int d = 64;
        var rng = new Random(123);
        var attn = new float[n * d];
        var mlp = new float[n * 4 * d];
        for (int i = 0; i < attn.Length; i++) attn[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < mlp.Length; i++) mlp[i] = (float)(rng.NextDouble() * 2 - 1);

        var exp = new float[n * 5 * d];
        for (int t = 0; t < n; t++)
        {
            int dst = t * 5 * d;
            Array.Copy(attn, t * d, exp, dst, d);
            Array.Copy(mlp, t * 4 * d, exp, dst + d, 4 * d);
        }

        var attnGpu = backend.Upload(attn, TensorShape.D1(attn.Length));
        var mlpGpu = backend.Upload(mlp, TensorShape.D1(mlp.Length));
        var outGpu = backend.Allocate(TensorShape.D1(n * 5 * d));

        try
        {
            backend.FluxConcatAttnMlp(attnGpu, mlpGpu, outGpu, n, d);

            var act = new float[n * 5 * d];
            backend.Download(outGpu, act);

            for (int i = 0; i < act.Length; i++)
            {
                Assert.Equal(exp[i], act[i], 5);
            }
        }
        finally
        {
            backend.Free(attnGpu);
            backend.Free(mlpGpu);
            backend.Free(outGpu);
        }
    }

    [Fact]
    public void FluxEulerStep_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int len = 1024;
        const float dt = 0.25f;
        var rng = new Random(789);
        var latent = new float[len];
        var v = new float[len];
        for (int i = 0; i < len; i++)
        {
            latent[i] = (float)(rng.NextDouble() * 2 - 1);
            v[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        var exp = new float[len];
        for (int i = 0; i < len; i++) exp[i] = latent[i] + dt * v[i];

        var latGpu = backend.Upload(latent, TensorShape.D1(len));
        var vGpu = backend.Upload(v, TensorShape.D1(len));

        try
        {
            backend.FluxEulerStep(latGpu, vGpu, dt, len);

            var act = new float[len];
            backend.Download(latGpu, act);

            for (int i = 0; i < len; i++)
            {
                Assert.True(MathF.Abs(exp[i] - act[i]) < 1e-4f, $"Diff at {i}: exp={exp[i]}, act={act[i]}");
            }
        }
        finally
        {
            backend.Free(latGpu);
            backend.Free(vGpu);
        }
    }

    [Fact]
    public void MultiHeadAttentionTiled128_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int qSeq = 32;
        const int kvSeq = 32;
        const int numHeads = 4;
        const int headDim = 128;
        var rng = new Random(456);

        var q = new float[qSeq * numHeads * headDim];
        var k = new float[kvSeq * numHeads * headDim];
        var v = new float[kvSeq * numHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);

        var cpuOut = new float[qSeq * numHeads * headDim];
        DiffusionOps.MultiHeadAttention(q, k, v, cpuOut.AsSpan(), qSeq, kvSeq, numHeads, headDim);

        var qGpu = backend.Upload(q, TensorShape.D1(q.Length));
        var kGpu = backend.Upload(k, TensorShape.D1(k.Length));
        var vGpu = backend.Upload(v, TensorShape.D1(v.Length));
        var outGpu = backend.Allocate(TensorShape.D1(cpuOut.Length));

        try
        {
            backend.MultiHeadAttentionTiled(outGpu, qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);

            var gpuOut = new float[cpuOut.Length];
            backend.Download(outGpu, gpuOut);

            for (int i = 0; i < cpuOut.Length; i++)
            {
                Assert.True(MathF.Abs(cpuOut[i] - gpuOut[i]) < 1e-3f, $"Mismatch at {i}: cpu={cpuOut[i]}, gpu={gpuOut[i]}");
            }
        }
        finally
        {
            backend.Free(qGpu);
            backend.Free(kGpu);
            backend.Free(vGpu);
            backend.Free(outGpu);
        }
    }

    [Fact]
    public void AdaLNModulate_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int nTokens = 64;
        const int dim = 128;
        var rng = new Random(101);

        var input = new float[nTokens * dim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var mod = new float[6 * dim];
        for (int i = 0; i < mod.Length; i++) mod[i] = (float)(rng.NextDouble() * 2 - 1);

        var cpuOut = new float[nTokens * dim];
        DiffusionOps.AdaLNModulate(cpuOut, input, mod.AsSpan(0, dim), mod.AsSpan(dim, dim), nTokens, dim, isRmsNorm: true, eps: 1e-6f);

        var inGpu = backend.Upload(input, TensorShape.D2(nTokens, dim));
        var modGpu = backend.Upload(mod, TensorShape.D1(mod.Length));
        var outGpu = backend.Allocate(TensorShape.D2(nTokens, dim));

        try
        {
            backend.AdaLNModulate(outGpu, inGpu, modGpu, nTokens, dim, shiftOffset: 0, scaleOffset: dim, isRmsNorm: true, eps: 1e-6f);

            var gpuOut = new float[nTokens * dim];
            backend.Download(outGpu, gpuOut);

            for (int i = 0; i < cpuOut.Length; i++)
            {
                Assert.True(MathF.Abs(cpuOut[i] - gpuOut[i]) < 1e-3f, $"Mismatch at {i}: cpu={cpuOut[i]}, gpu={gpuOut[i]}");
            }
        }
        finally
        {
            backend.Free(inGpu);
            backend.Free(modGpu);
            backend.Free(outGpu);
        }
    }

    [Fact]
    public void SgemmF16_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int M = 48;
        const int K = 96;
        const int N = 64;
        var rng = new Random(202);

        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        var bF16 = new Half[N * K];
        var bF32 = new float[N * K];
        for (int i = 0; i < bF16.Length; i++)
        {
            float val = (float)(rng.NextDouble() * 2 - 1);
            bF16[i] = (Half)val;
            bF32[i] = (float)bF16[i];
        }

        var cpuOut = new float[M * N];
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float sum = 0f;
                for (int k = 0; k < K; k++)
                {
                    sum += a[m * K + k] * bF32[n * K + k];
                }
                cpuOut[m * N + n] = sum;
            }
        }

        var aGpu = backend.Upload(a, TensorShape.D2(M, K));
        var bGpu = backend.UploadHalf(bF16, TensorShape.D2(N, K));
        var cGpu = backend.Allocate(TensorShape.D2(M, N));

        try
        {
            backend.Sgemm(cGpu, aGpu, bGpu, M, K, N);

            var gpuOut = new float[M * N];
            backend.Download(cGpu, gpuOut);

            for (int i = 0; i < cpuOut.Length; i++)
            {
                Assert.True(MathF.Abs(cpuOut[i] - gpuOut[i]) < 1e-2f, $"Mismatch at {i}: cpu={cpuOut[i]}, gpu={gpuOut[i]}");
            }
        }
        finally
        {
            backend.Free(aGpu);
            backend.Free(bGpu);
            backend.Free(cGpu);
        }
    }

    [Fact]
    public void SgemmF16_M1_MatchesCpuReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int M = 1;
        const int K = 3072;
        const int N = 9216;
        var rng = new Random(42);

        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        var bF16 = new Half[N * K];
        var bF32 = new float[N * K];
        for (int i = 0; i < bF16.Length; i++)
        {
            float val = (float)(rng.NextDouble() * 2 - 1);
            bF16[i] = (Half)val;
            bF32[i] = (float)bF16[i];
        }

        var cpuOut = new float[M * N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++)
            {
                sum += a[k] * bF32[n * K + k];
            }
            cpuOut[n] = sum;
        }

        var aGpu = backend.Upload(a, TensorShape.D2(M, K));
        var bGpu = backend.UploadHalf(bF16, TensorShape.D2(N, K));
        var cGpu = backend.Allocate(TensorShape.D2(M, N));

        try
        {
            backend.Sgemm(cGpu, aGpu, bGpu, M, K, N);

            var gpuOut = new float[M * N];
            backend.Download(cGpu, gpuOut);

            for (int i = 0; i < cpuOut.Length; i++)
            {
                Assert.True(MathF.Abs(cpuOut[i] - gpuOut[i]) < 1e-2f, $"Mismatch at {i}: cpu={cpuOut[i]}, gpu={gpuOut[i]}");
            }
        }
        finally
        {
            backend.Free(aGpu);
            backend.Free(bGpu);
            backend.Free(cGpu);
        }
    }
}
