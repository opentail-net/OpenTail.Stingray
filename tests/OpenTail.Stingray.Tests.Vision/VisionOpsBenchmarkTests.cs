using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed unsafe class VisionOpsBenchmarkTests
{
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
