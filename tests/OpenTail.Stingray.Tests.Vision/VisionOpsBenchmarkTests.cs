using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed unsafe class VisionOpsBenchmarkTests
{
    /// <summary>
    /// Pre-vectorization VisionOps.Attention, kept verbatim as the comparison baseline for
    /// <see cref="Benchmark_Attention_ScalarVsVectorized"/> -- see
    /// docs/done/vision-attention-vectorization-2026-08-20.md.
    /// </summary>
    private static void Attention_Scalar(
        float[] q, float[] k, float[] v, int nTokens, int heads, int headDim, float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int embd = heads * headDim;

        Parallel.For(0, heads, h =>
        {
            int headOff = h * headDim;
            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int qOff = i * embd + headOff;

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    int kOff = j * embd + headOff;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qOff + d] * k[kOff + d];
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float expSum = 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;

                int outOff = i * embd + headOff;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < nTokens; j++)
                    {
                        int vOff = j * embd + headOff;
                        acc += v[vOff + d] * scores[j];
                    }
                    output[outOff + d] = acc * invSum;
                }
            }
        });
    }

    /// <summary>
    /// Confirms VisionOps.Attention's TensorPrimitives-vectorized rewrite is actually faster than
    /// the scalar version it replaced, at a scale representative of a real ViT (1024 tokens, 16
    /// heads, head_dim 64 -- comparable to Pixtral's patch grid). Also checks the two
    /// implementations agree numerically (same algorithm, different accumulation order --
    /// tolerance accounts for floating-point reassociation, not a correctness bug).
    /// </summary>
    [Fact]
    public void Benchmark_Attention_ScalarVsVectorized()
    {
        int nTokens = 1024, heads = 16, headDim = 64;
        int total = nTokens * heads * headDim;

        var q = new float[total];
        var k = new float[total];
        var v = new float[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++)
        {
            q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var outScalar = new float[total];
        var outVectorized = new float[total];

        // Warmup
        for (int w = 0; w < 2; w++)
        {
            Attention_Scalar(q, k, v, nTokens, heads, headDim, outScalar);
            VisionOps.Attention(q, k, v, nTokens, heads, headDim, outVectorized);
        }

        // Numerical agreement check (float reassociation tolerance, not a correctness bound).
        float maxDiff = 0f;
        for (int i = 0; i < total; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(outScalar[i] - outVectorized[i]));
        Assert.True(maxDiff < 1e-3f, $"Scalar vs vectorized Attention diverged: maxDiff={maxDiff}");

        int iterations = 5;
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iterations; it++)
            Attention_Scalar(q, k, v, nTokens, heads, headDim, outScalar);
        sw.Stop();
        double scalarMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int it = 0; it < iterations; it++)
            VisionOps.Attention(q, k, v, nTokens, heads, headDim, outVectorized);
        sw.Stop();
        double vectorizedMs = sw.Elapsed.TotalMilliseconds / iterations;

        double speedup = scalarMs / vectorizedMs;
        Assert.True(speedup > 1.2, $"Speedup was only {speedup:F2}x (scalar={scalarMs:F2}ms, vectorized={vectorizedMs:F2}ms)");
    }

    /// <summary>
    /// Pre-vectorization VisionOps.AttentionGqa, kept verbatim as the comparison baseline for
    /// <see cref="Benchmark_AttentionGqa_ScalarVsVectorized"/> -- see
    /// docs/done/vision-attention-vectorization-2026-08-20.md.
    /// </summary>
    private static void AttentionGqa_Scalar(
        float[] q, float[] k, float[] v, int nTokens, int qHeads, int kvHeads, int headDim,
        float[] output, float* attnSinks = null)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int groupSize = qHeads / kvHeads;
        int qEmbd = qHeads * headDim;
        int kvEmbd = kvHeads * headDim;

        Parallel.For(0, qHeads, qh =>
        {
            int kvHead = qh / groupSize;
            int qOffHead = qh * headDim;
            int kvOffHead = kvHead * headDim;
            float sink = (attnSinks != null) ? attnSinks[qh] : 0f;

            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int qOff = i * qEmbd + qOffHead;

                float maxScore = sink != 0f ? sink : float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    int kOff = j * kvEmbd + kvOffHead;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qOff + d] * k[kOff + d];
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float expSum = sink != 0f ? MathF.Exp(sink - maxScore) : 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;

                int outOff = i * qEmbd + qOffHead;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < nTokens; j++)
                    {
                        int vOff = j * kvEmbd + kvOffHead;
                        acc += v[vOff + d] * scores[j];
                    }
                    output[outOff + d] = acc * invSum;
                }
            }
        });
    }

    /// <summary>
    /// Same shape/tolerance/speedup checks as <see cref="Benchmark_Attention_ScalarVsVectorized"/>,
    /// for the GQA sibling (2:1 query:kv head ratio, matching a typical GQA vision model).
    /// </summary>
    [Fact]
    public void Benchmark_AttentionGqa_ScalarVsVectorized()
    {
        int nTokens = 1024, qHeads = 16, kvHeads = 8, headDim = 64;
        int qTotal = nTokens * qHeads * headDim;
        int kvTotal = nTokens * kvHeads * headDim;

        var q = new float[qTotal];
        var k = new float[kvTotal];
        var v = new float[kvTotal];
        var rng = new Random(43);
        for (int i = 0; i < qTotal; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < kvTotal; i++)
        {
            k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var outScalar = new float[qTotal];
        var outVectorized = new float[qTotal];

        for (int w = 0; w < 2; w++)
        {
            AttentionGqa_Scalar(q, k, v, nTokens, qHeads, kvHeads, headDim, outScalar);
            VisionOps.AttentionGqa(q, k, v, nTokens, qHeads, kvHeads, headDim, outVectorized);
        }

        float maxDiff = 0f;
        for (int i = 0; i < qTotal; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(outScalar[i] - outVectorized[i]));
        Assert.True(maxDiff < 1e-3f, $"Scalar vs vectorized AttentionGqa diverged: maxDiff={maxDiff}");

        int iterations = 5;
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iterations; it++)
            AttentionGqa_Scalar(q, k, v, nTokens, qHeads, kvHeads, headDim, outScalar);
        sw.Stop();
        double scalarMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int it = 0; it < iterations; it++)
            VisionOps.AttentionGqa(q, k, v, nTokens, qHeads, kvHeads, headDim, outVectorized);
        sw.Stop();
        double vectorizedMs = sw.Elapsed.TotalMilliseconds / iterations;

        double speedup = scalarMs / vectorizedMs;
        Assert.True(speedup > 1.2, $"Speedup was only {speedup:F2}x (scalar={scalarMs:F2}ms, vectorized={vectorizedMs:F2}ms)");
    }

    private static void MatVecF16_Scalar(
        float[] input,
        Half* weights,
        float* bias,
        int nTokens,
        int inDim,
        int outDim,
        float[] output)
    {
        if (weights == null) return;

        Parallel.For(0, nTokens, t =>
        {
            int inOff = t * inDim;
            int outOff = t * outDim;

            for (int o = 0; o < outDim; o++)
            {
                float sum = bias != null ? bias[o] : 0f;
                int rowOff = o * inDim;

                for (int i = 0; i < inDim; i++)
                {
                    sum += input[inOff + i] * (float)weights[rowOff + i];
                }
                output[outOff + o] = sum;
            }
        });
    }

    private static void MatVecF32_SimdFma(
        float[] input,
        float* weightsF32,
        float* bias,
        int nTokens,
        int inDim,
        int outDim,
        float[] output)
    {
        if (weightsF32 == null) return;

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            var inPtr = pIn;
            var outPtr = pOut;
            var wPtr = weightsF32;
            var bPtr = bias;

            Parallel.For(0, nTokens, t =>
            {
                float* rowIn = inPtr + (long)t * inDim;
                float* rowOut = outPtr + (long)t * outDim;

                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr != null ? bPtr[o] : 0f;
                    float* rowW = wPtr + (long)o * inDim;

                    int i = 0;

                    if (Vector256.IsHardwareAccelerated && inDim >= 32)
                    {
                        var acc0 = Vector256<float>.Zero;
                        var acc1 = Vector256<float>.Zero;
                        var acc2 = Vector256<float>.Zero;
                        var acc3 = Vector256<float>.Zero;

                        int vecLimit = inDim - 31;

                        for (; i < vecLimit; i += 32)
                        {
                            var vIn0 = Vector256.Load(rowIn + i + 0);
                            var vW0 = Vector256.Load(rowW + i + 0);
                            acc0 = Vector256.MultiplyAddEstimate(vIn0, vW0, acc0);

                            var vIn1 = Vector256.Load(rowIn + i + 8);
                            var vW1 = Vector256.Load(rowW + i + 8);
                            acc1 = Vector256.MultiplyAddEstimate(vIn1, vW1, acc1);

                            var vIn2 = Vector256.Load(rowIn + i + 16);
                            var vW2 = Vector256.Load(rowW + i + 16);
                            acc2 = Vector256.MultiplyAddEstimate(vIn2, vW2, acc2);

                            var vIn3 = Vector256.Load(rowIn + i + 24);
                            var vW3 = Vector256.Load(rowW + i + 24);
                            acc3 = Vector256.MultiplyAddEstimate(vIn3, vW3, acc3);
                        }

                        var totalAcc = (acc0 + acc1) + (acc2 + acc3);
                        sum += Vector256.Sum(totalAcc);
                    }

                    for (; i < inDim; i++)
                    {
                        sum += rowIn[i] * rowW[i];
                    }

                    rowOut[o] = sum;
                }
            });
        }
    }

    [Fact]
    public void Benchmark_MatVecF16_ScalarVsSimd()
    {
        int nTokens = 256;
        int inDim = 1024;
        int outDim = 2048;

        var input = new float[nTokens * inDim];
        var weights = new Half[outDim * inDim];
        var weightsF32 = new float[outDim * inDim];
        var bias = new float[outDim];

        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < weights.Length; i++)
        {
            float val = (float)(rng.NextDouble() * 2.0 - 1.0);
            weights[i] = (Half)val;
            weightsF32[i] = (float)(Half)val;
        }
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 0.1);

        var outScalar = new float[nTokens * outDim];
        var outSimd = new float[nTokens * outDim];

        fixed (Half* pW = weights)
        fixed (float* pWF32 = weightsF32)
        fixed (float* pB = bias)
        {
            // Warmup
            for (int w = 0; w < 3; w++)
            {
                MatVecF16_Scalar(input, pW, pB, nTokens, inDim, outDim, outScalar);
                MatVecF32_SimdFma(input, pWF32, pB, nTokens, inDim, outDim, outSimd);
            }

            int iterations = 5;

            var sw = Stopwatch.StartNew();
            for (int it = 0; it < iterations; it++)
            {
                MatVecF16_Scalar(input, pW, pB, nTokens, inDim, outDim, outScalar);
            }
            sw.Stop();
            double scalarMs = sw.Elapsed.TotalMilliseconds / iterations;

            sw.Restart();
            for (int it = 0; it < iterations; it++)
            {
                MatVecF32_SimdFma(input, pWF32, pB, nTokens, inDim, outDim, outSimd);
            }
            sw.Stop();
            double simdMs = sw.Elapsed.TotalMilliseconds / iterations;

            double speedup = scalarMs / simdMs;
            Assert.True(speedup > 2.0, $"Speedup was {speedup:F2}x");
        }
    }
}
