using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Parity and numerical-property tests for the Gemma-4 CPU helper kernels added
/// in Phase 2 of the Gemma 4 E4B plan: <see cref="SimdKernels.GeluTanhMul"/>,
/// <see cref="SimdKernels.ScaleInPlace"/> and <see cref="SimdKernels.SoftcapInPlace"/>.
///
/// <para><b>GeluTanhMul</b> implements the tanh approximation of GELU fused with
/// the up-projection multiply, i.e.
/// <c>out[i] = 0.5 * g * (1 + tanh(sqrt(2/π) * (g + 0.044715 * g^3))) * up[i]</c>.
/// We cross-check the AVX2 dispatcher against an internal scalar reference using
/// <see cref="MathF.Tanh"/> (no exp approximation) at a tight max-abs-diff bound,
/// plus two numerical edge cases (zero input ⇒ zero, large positive ⇒ gate·up).</para>
///
/// <para><b>SoftcapInPlace</b> clips logits via <c>x = tanh(x/cap) * cap</c>; for
/// |x| ≫ cap the output must have magnitude ≤ cap, and for |x| ≪ cap the output
/// must pass through with negligible error.</para>
///
/// AVX2-gated cases follow the existing <c>SimdKernelsQ8KSTests</c> guard:
/// <c>if (!Avx2.IsSupported || !Fma.IsSupported) return;</c>. The scalar
/// fallbacks are exercised indirectly on hosts without AVX2.
/// </summary>
public sealed unsafe class SimdKernelsGemma4Tests
{
    [Fact]
    public void GeluTanhMul_MatchesScalar()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(unchecked((int)0xBEEFCAFE));
        foreach (int n in new[] { 64, 257, 1024, 4096 })
        {
            var gate = new float[n];
            var up = new float[n];
            var avxOut = new float[n];
            var scalarOut = new float[n];

            for (int i = 0; i < n; i++)
            {
                // GELU is exercised in the range where it matters (~|x| < 5);
                // pick inputs in [-3, 3] for activations and [-2, 2] for up.
                gate[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
                up[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            }

            fixed (float* g = gate)
            fixed (float* u = up)
            fixed (float* oa = avxOut)
            fixed (float* os = scalarOut)
            {
                SimdKernels.GeluTanhMul(g, u, oa, n);
                SimdKernels.GeluTanhMul_Scalar(g, u, os, n);
            }

            float maxAbs = 0f;
            int worstIdx = -1;
            for (int i = 0; i < n; i++)
            {
                float d = MathF.Abs(avxOut[i] - scalarOut[i]);
                if (d > maxAbs) { maxAbs = d; worstIdx = i; }
            }
            Console.WriteLine(
                $"GeluTanhMul avx-vs-scalar n={n}: maxAbs={maxAbs:E3} (idx={worstIdx})");
            Assert.True(maxAbs < 1e-5f,
                $"GeluTanhMul AVX2 vs scalar diff too large at n={n}: maxAbs={maxAbs:E3}");
        }
    }

    [Fact]
    public void GeluTanhMul_ZeroInput_ReturnsZero()
    {
        const int n = 64;
        var gate = new float[n];   // all zeros
        var up = new float[n];
        for (int i = 0; i < n; i++) up[i] = (i + 1) * 0.5f;
        var outp = new float[n];

        fixed (float* g = gate)
        fixed (float* u = up)
        fixed (float* o = outp)
            SimdKernels.GeluTanhMul(g, u, o, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(0f, outp[i]);
    }

    [Fact]
    public void GeluTanhMul_LargePositive_ApproachesGateTimesUp()
    {
        // For large positive g, tanh(...) → 1, so gelu_tanh(g) → g, and
        // out[i] → gate[i] * up[i].
        const int n = 64;
        var gate = new float[n];
        var up = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = 20f + (i % 8);   // very large positive
            up[i] = 0.25f * (i + 1);
        }
        var outp = new float[n];

        fixed (float* g = gate)
        fixed (float* u = up)
        fixed (float* o = outp)
            SimdKernels.GeluTanhMul(g, u, o, n);

        float maxRel = 0f;
        for (int i = 0; i < n; i++)
        {
            float expected = gate[i] * up[i];
            float rel = MathF.Abs(outp[i] - expected) / MathF.Abs(expected);
            if (rel > maxRel) maxRel = rel;
        }
        Console.WriteLine($"GeluTanhMul large-positive max rel diff = {maxRel:E3}");
        Assert.True(maxRel < 1e-4f,
            $"GeluTanhMul large-positive should approach gate*up; maxRel={maxRel:E3}");
    }

    [Fact]
    public void ScaleInPlace_MultipliesEveryElement()
    {
        const int n = 257;   // odd size to exercise the AVX2 tail
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = 1.0f;

        fixed (float* p = x)
            SimdKernels.ScaleInPlace(p, 3.5f, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(3.5f, x[i]);
    }

    [Fact]
    public void SoftcapInPlace_LargeMagnitudeClamps()
    {
        const int n = 128;
        const float cap = 30f;
        var x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = (i % 2 == 0) ? 100f + i : -100f - i;

        fixed (float* p = x)
            SimdKernels.SoftcapInPlace(p, n, cap);

        for (int i = 0; i < n; i++)
        {
            Assert.True(MathF.Abs(x[i]) <= cap + 1e-3f,
                $"SoftcapInPlace did not clamp idx={i}: x={x[i]} (cap={cap})");
        }
    }

    [Fact]
    public void SoftcapInPlace_SmallMagnitudePassesThrough()
    {
        // For |x| ≪ cap, tanh(x/cap)*cap ≈ x with relative error < (x/cap)^2 / 3.
        // With cap=30 and |x|=0.1, x/cap ≈ 3.3e-3, so error ≈ 3.7e-6 — well under
        // the 1e-3 absolute tolerance asked for in the task spec.
        const int n = 128;
        const float cap = 30f;
        var x = new float[n];
        var orig = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (i % 2 == 0) ? 0.1f : -0.1f;
            orig[i] = x[i];
        }

        fixed (float* p = x)
            SimdKernels.SoftcapInPlace(p, n, cap);

        float maxAbs = 0f;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(x[i] - orig[i]);
            if (d > maxAbs) maxAbs = d;
        }
        Console.WriteLine($"SoftcapInPlace small-magnitude max abs diff = {maxAbs:E3}");
        Assert.True(maxAbs < 1e-3f,
            $"SoftcapInPlace small-magnitude must pass through; maxAbs={maxAbs:E3}");
    }

    [Fact]
    public void CpuBackend_Gemma4Ops_Parity()
    {
        using var backend = new CpuBackend();

        // 1. GeluTanhMul - test full spectrum against formula
        const int n = 128;
        var gate = backend.Allocate(TensorShape.D1(n));
        var up = backend.Allocate(TensorShape.D1(n));
        var gData = new float[n];
        var uData = new float[n];
        for (int i = 0; i < n; i++) { gData[i] = (i - 64) * 0.1f; uData[i] = 1.5f; }
        fixed (float* pg = gData) fixed (float* pu = uData)
        {
            new ReadOnlySpan<float>(pg, n).CopyTo(new Span<float>((void*)gate.Handle, n));
            new ReadOnlySpan<float>(pu, n).CopyTo(new Span<float>((void*)up.Handle, n));
        }

        backend.GeluTanhMul(gate, up);

        var gateResult = new float[n];
        backend.Download(gate, gateResult);
        const float alpha = 0.7978845608f;
        const float beta = 0.044715f;
        for (int i = 0; i < n; i++)
        {
            float g = gData[i];
            float expected = 0.5f * g * (1.0f + MathF.Tanh(alpha * (g + beta * g * g * g))) * uData[i];
            Assert.True(MathF.Abs(gateResult[i] - expected) < 1e-4f, $"GeluTanhMul mismatch at idx {i}: got {gateResult[i]}, expected {expected}");
        }

        // 2. SoftcapInPlace - test capped & sub-cap values
        var x = backend.Allocate(TensorShape.D1(n));
        var xData = new float[n];
        for (int i = 0; i < n; i++) xData[i] = (i % 2 == 0) ? 100f : 0.5f;
        fixed (float* px = xData) new ReadOnlySpan<float>(px, n).CopyTo(new Span<float>((void*)x.Handle, n));

        const float cap = 20f;
        backend.SoftcapInPlace(x, cap);

        var xResult = new float[n];
        backend.Download(x, xResult);
        for (int i = 0; i < n; i++)
        {
            float expected = MathF.Tanh(xData[i] / cap) * cap;
            Assert.True(MathF.Abs(xResult[i] - expected) < 1e-3f, $"SoftcapInPlace mismatch at idx {i}: got {xResult[i]}, expected {expected}");
        }

        // 3. AttentionSwa - test sliding-window attention with GQA
        int headDim = 16, numHeads = 4, numKvHeads = 2, maxSeqLen = 8, pos = 3, win = 4;
        var q = backend.Allocate(TensorShape.D1(numHeads * headDim));
        var kCache = backend.Allocate(TensorShape.D1(maxSeqLen * numKvHeads * headDim));
        var vCache = backend.Allocate(TensorShape.D1(maxSeqLen * numKvHeads * headDim));
        var out1 = backend.Allocate(TensorShape.D1(numHeads * headDim));
        var out2 = backend.Allocate(TensorShape.D1(numHeads * headDim));
        var scratch = backend.Allocate(TensorShape.D1(numHeads * win));

        var qData = new float[numHeads * headDim];
        var kData = new float[maxSeqLen * numKvHeads * headDim];
        var vData = new float[maxSeqLen * numKvHeads * headDim];
        for (int i = 0; i < qData.Length; i++) qData[i] = (i + 1) * 0.1f;
        for (int i = 0; i < kData.Length; i++) kData[i] = (i % 7 + 1) * 0.2f;
        for (int i = 0; i < vData.Length; i++) vData[i] = (i % 5 + 1) * 0.3f;

        fixed (float* pq = qData) new ReadOnlySpan<float>(pq, qData.Length).CopyTo(new Span<float>((void*)q.Handle, qData.Length));
        fixed (float* pk = kData) new ReadOnlySpan<float>(pk, kData.Length).CopyTo(new Span<float>((void*)kCache.Handle, kData.Length));
        fixed (float* pv = vData) new ReadOnlySpan<float>(pv, vData.Length).CopyTo(new Span<float>((void*)vCache.Handle, vData.Length));

        // Execute AttentionSwa with scratch buffer and without scratch buffer
        backend.AttentionSwa(q, kCache, vCache, out1, scratch, pos, win, headDim, numHeads, numKvHeads, maxSeqLen);
        backend.AttentionSwa(q, kCache, vCache, out2, null, pos, win, headDim, numHeads, numKvHeads, maxSeqLen);

        var res1 = new float[numHeads * headDim];
        var res2 = new float[numHeads * headDim];
        backend.Download(out1, res1);
        backend.Download(out2, res2);

        // Verify non-zero attention output and scratch vs no-scratch parity
        Assert.NotEqual(0f, res1[0]);
        for (int i = 0; i < res1.Length; i++)
        {
            Assert.True(MathF.Abs(res1[i] - res2[i]) < 1e-5f, $"AttentionSwa scratch vs no-scratch parity mismatch at idx {i}");
        }

        gate.Dispose();
        up.Dispose();
        x.Dispose();
        q.Dispose();
        kCache.Dispose();
        vCache.Dispose();
        out1.Dispose();
        out2.Dispose();
        scratch.Dispose();
    }

    [Fact]
    public void Sampler_ArrayPool_Boundary_Parity()
    {
        // Tests Sampler.Sample across stackalloc (<= 256) and ArrayPool (> 256) boundaries
        int vocabSize = 300; // > 256 boundary
        var logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            logits[i] = i * 0.01f;

        var spSmall = new SamplingParams { Temperature = 0.8f, TopK = 10, TopP = 0.9f };
        var spLarge = new SamplingParams { Temperature = 0.8f, TopK = 280, TopP = 0.9f };

        int tokenSmall = Sampler.Sample(logits, spSmall, new Random(42));
        int tokenLarge = Sampler.Sample(logits, spLarge, new Random(42));

        Assert.True(tokenSmall >= 0 && tokenSmall < vocabSize);
        Assert.True(tokenLarge >= 0 && tokenLarge < vocabSize);
    }
}



