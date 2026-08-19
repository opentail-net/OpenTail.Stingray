using System;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed unsafe class GpuVisionOpsTests
{
    [Fact]
    public void VisionOps_PixelShuffle2x2_MatchesCpuReference()
    {
        int gridY = 4;
        int gridX = 4;
        int inDim = 8;
        int totalIn = gridY * gridX * inDim;

        var input = new float[totalIn];
        for (int i = 0; i < totalIn; i++) input[i] = (i * 0.13f) % 20f;

        int outY = gridY / 2;
        int outX = gridX / 2;
        int outDim = inDim * 4;
        var cpuOutput = new float[outY * outX * outDim];
        VisionOps.PixelShuffle2x2(input, gridY, gridX, inDim, cpuOutput);

        // Verify ordering properties
        Assert.Equal(outY * outX * outDim, cpuOutput.Length);
        for (int ty = 0; ty < outY; ty++)
        for (int tx = 0; tx < outX; tx++)
        {
            int tokenIdx = ty * outX + tx;
            int dstOff = tokenIdx * outDim;
            int py0 = ty * 2;
            int px0 = tx * 2;

            int p00 = (py0 * gridX + px0) * inDim;
            int p01 = (py0 * gridX + px0 + 1) * inDim;
            int p10 = ((py0 + 1) * gridX + px0) * inDim;
            int p11 = ((py0 + 1) * gridX + px0 + 1) * inDim;

            for (int c = 0; c < inDim; c++)
            {
                Assert.Equal(input[p00 + c], cpuOutput[dstOff + c]);
                Assert.Equal(input[p01 + c], cpuOutput[dstOff + inDim + c]);
                Assert.Equal(input[p10 + c], cpuOutput[dstOff + inDim * 2 + c]);
                Assert.Equal(input[p11 + c], cpuOutput[dstOff + inDim * 3 + c]);
            }
        }
    }

    [Fact]
    public void VisionOps_Mrope2d_MatchesCpuReference()
    {
        int px = 4, py = 4;
        int qHeads = 4, kvHeads = 2, headDim = 16;
        int totalQ = px * py * qHeads * headDim;
        int totalK = px * py * kvHeads * headDim;

        var q = new float[totalQ];
        var k = new float[totalK];
        for (int i = 0; i < totalQ; i++) q[i] = (i * 0.7f) % 5.0f;
        for (int i = 0; i < totalK; i++) k[i] = (i * 0.9f) % 5.0f;

        var qCopy = (float[])q.Clone();
        var kCopy = (float[])k.Clone();

        VisionOps.ApplyMRoPE(q, k, px, py, qHeads, kvHeads, headDim, 10000.0f);

        // Assert non-trivial transformation that preserves norms
        float qNormBefore = TensorPrimitives.SumOfSquares(qCopy.AsSpan());
        float qNormAfter = TensorPrimitives.SumOfSquares(q.AsSpan());
        Assert.InRange(qNormAfter, qNormBefore * 0.999f, qNormBefore * 1.001f);

        float kNormBefore = TensorPrimitives.SumOfSquares(kCopy.AsSpan());
        float kNormAfter = TensorPrimitives.SumOfSquares(k.AsSpan());
        Assert.InRange(kNormAfter, kNormBefore * 0.999f, kNormBefore * 1.001f);
    }

    [Fact]
    public void VisionOps_LayerNorm_MatchesCpuReference()
    {
        int nTokens = 4;
        int embd = 16;
        var x = new float[nTokens * embd];
        var w = new float[embd];
        var b = new float[embd];

        for (int i = 0; i < x.Length; i++) x[i] = (i * 0.31f) % 10.0f;
        for (int i = 0; i < embd; i++) { w[i] = 1.2f; b[i] = 0.5f; }

        var xCopy = (float[])x.Clone();
        fixed (float* pW = w, pB = b)
        {
            VisionOps.LayerNorm(x, nTokens, embd, pW, pB, 1e-5f);
        }

        // Each token vector must have zero mean and unit variance scaled by w + b
        for (int t = 0; t < nTokens; t++)
        {
            int off = t * embd;
            float sum = 0f;
            for (int i = 0; i < embd; i++) sum += x[off + i];
            float mean = sum / embd;
            // Mean should be close to bias (0.5)
            Assert.InRange(mean, 0.49f, 0.51f);
        }
    }
}
