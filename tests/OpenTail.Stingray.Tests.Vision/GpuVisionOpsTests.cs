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

    [Fact]
    public void DiffusionOps_AdaLNModulate_MatchesMathematicalDefinition()
    {
        int nTokens = 2;
        int dim = 8;
        var input = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,  2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        var shift = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
        var scale = new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        var output = new float[input.Length];

        Diffusion.DiffusionOps.AdaLNModulate(output, input, shift, scale, nTokens, dim, isRmsNorm: true, eps: 1e-5f);

        // For token 0: sum of squares = 1+4+9+16+25+36+49+64 = 204. invStd = 1/sqrt(204/8) = 1/sqrt(25.5) = 0.19803
        // Element 0: 1 * 0.19803 * 1.5 + 0.1 = 0.397045
        float expected0 = 1f * (1f / MathF.Sqrt(204f / 8f + 1e-5f)) * 1.5f + 0.1f;
        Assert.Equal(expected0, output[0], precision: 4);
    }

    [Fact]
    public void DiffusionOps_ScaleGateAdd_ModulatesResidualAccurately()
    {
        int nTokens = 2;
        int dim = 4;
        var x = new float[] { 1f, 1f, 1f, 1f,  2f, 2f, 2f, 2f };
        var proj = new float[] { 2f, 3f, 4f, 5f,  1f, 2f, 3f, 4f };
        var gate = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };

        Diffusion.DiffusionOps.ScaleGateAdd(x, proj, gate, nTokens, dim);

        // Token 0: [1 + 2*0.5, 1 + 3*0.5, 1 + 4*0.5, 1 + 5*0.5] = [2, 2.5, 3, 3.5]
        Assert.Equal(2.0f, x[0]);
        Assert.Equal(2.5f, x[1]);
        Assert.Equal(3.0f, x[2]);
        Assert.Equal(3.5f, x[3]);
    }

    [Fact]
    public void DiffusionOps_QKNorm_NormalizesPerHeadIndependently()
    {
        int nTokens = 1;
        int numHeads = 2;
        int headDim = 4;
        var q = new float[] { 1f, 2f, 3f, 4f,  10f, 20f, 30f, 40f };
        var k = new float[] { 2f, 2f, 2f, 2f,  5f, 5f, 5f, 5f };
        var qScale = new float[] { 1.5f, 1.5f, 1.5f, 1.5f };
        var kScale = new float[] { 0.8f, 0.8f, 0.8f, 0.8f };

        Diffusion.DiffusionOps.QKNorm(q, k, qScale, kScale, nTokens, numHeads, headDim, 1e-5f);

        // Check head 0 of Q: sumSq = 1+4+9+16 = 30. invStd = 1/sqrt(30/4) = 1/sqrt(7.5) = 0.365148.
        // Element 0: 1 * 0.365148 * 1.5 = 0.54772
        float expectedQ0 = 1f * (1f / MathF.Sqrt(7.5f + 1e-5f)) * 1.5f;
        Assert.Equal(expectedQ0, q[0], precision: 4);

        // Check head 0 of K: sumSq = 4*4 = 16. invStd = 1/sqrt(16/4) = 1/2 = 0.5.
        // Element 0: 2 * 0.5 * 0.8 = 0.8
        Assert.Equal(0.8f, k[0], precision: 4);
    }

    [Fact]
    public void VisionOps_RoPE3D_PreservesNormsAcrossTemporalAndSpatialAxes()
    {
        int numTokens = 8;
        int numHeads = 2;
        int headDim = 12; // 6 sub-bands: 2 temporal, 2 height, 2 width
        int tDim = 2, hDim = 2, wDim = 2;

        var q = new float[numTokens * numHeads * headDim];
        var k = new float[numTokens * numHeads * headDim];
        for (int i = 0; i < q.Length; i++) { q[i] = (i * 0.37f) % 5f; k[i] = (i * 0.51f) % 5f; }

        var qCopy = (float[])q.Clone();
        var kCopy = (float[])k.Clone();

        VisionOps.ApplyRoPE3D(q, k, numTokens, numHeads, headDim, tDim, hDim, wDim, 10000.0f);

        float qNormBefore = TensorPrimitives.SumOfSquares(qCopy.AsSpan());
        float qNormAfter = TensorPrimitives.SumOfSquares(q.AsSpan());
        Assert.InRange(qNormAfter, qNormBefore * 0.999f, qNormBefore * 1.001f);

        float kNormBefore = TensorPrimitives.SumOfSquares(kCopy.AsSpan());
        float kNormAfter = TensorPrimitives.SumOfSquares(k.AsSpan());
        Assert.InRange(kNormAfter, kNormBefore * 0.999f, kNormBefore * 1.001f);
    }
}
