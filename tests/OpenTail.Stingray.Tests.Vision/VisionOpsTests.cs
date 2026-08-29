
namespace OpenTail.Stingray.Tests.Vision;

public sealed unsafe class VisionOpsTests
{
    [Fact]
    public void Attention_ComputesCorrectSoftmaxAndWeightedSum()
    {
        int nTokens = 2;
        int heads = 1;
        int headDim = 2;

        // Q = [[1, 0], [0, 1]]
        // K = [[1, 0], [0, 1]]
        // V = [[2, 4], [6, 8]]
        var q = new float[] { 1f, 0f, 0f, 1f };
        var k = new float[] { 1f, 0f, 0f, 1f };
        var v = new float[] { 2f, 4f, 6f, 8f };
        var output = new float[nTokens * heads * headDim];

        VisionOps.Attention(q, k, v, nTokens, heads, headDim, output);

        // Scale = 1 / sqrt(2) ≈ 0.70710678
        // For token 0:
        // dot(Q0, K0) = 1 -> score0 = 0.70710678
        // dot(Q0, K1) = 0 -> score1 = 0
        // Softmax(0.7071, 0) = [0.670, 0.330]
        // Output0 = 0.670 * [2, 4] + 0.330 * [6, 8] = [1.34 + 1.98, 2.68 + 2.64] = [3.32, 5.32]
        Assert.True(output[0] > 2f && output[0] < 6f);
        Assert.True(output[1] > 4f && output[1] < 8f);
        // For token 1, symmetric:
        Assert.True(output[2] > 2f && output[2] < 6f);
        Assert.True(output[3] > 4f && output[3] < 8f);
        Assert.True(output[2] > output[0]); // token 1 attends more to V1
    }

    [Fact]
    public void AttentionGqa_WithAttentionSinks_AttenuatesAttentionMass()
    {
        int nTokens = 2;
        int qHeads = 2;
        int kvHeads = 1;
        int headDim = 2;

        var q = new float[] { 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f };
        var k = new float[] { 1f, 0f, 0f, 1f };
        var v = new float[] { 2f, 4f, 6f, 8f };
        var outputNoSink = new float[nTokens * qHeads * headDim];
        var outputWithSink = new float[nTokens * qHeads * headDim];

        VisionOps.AttentionGqa(q, k, v, nTokens, qHeads, kvHeads, headDim, outputNoSink);

        var sinks = new float[] { 10.0f, 10.0f };
        fixed (float* pSinks = sinks)
        {
            VisionOps.AttentionGqa(q, k, v, nTokens, qHeads, kvHeads, headDim, outputWithSink, pSinks);
        }

        // With huge sink logit (+10.0), sink consumes almost all softmax probability mass,
        // so output vectors shrink toward 0.
        float normNoSink = TensorPrimitives.SumOfSquares(outputNoSink.AsSpan());
        float normWithSink = TensorPrimitives.SumOfSquares(outputWithSink.AsSpan());

        Assert.True(normNoSink > 1.0f);
        Assert.True(normWithSink < normNoSink * 0.1f);
    }

    [Fact]
    public void PixelShuffle2x2_CorrectlyRearrangesSpatialTiles()
    {
        int gridY = 2;
        int gridX = 2;
        int inDim = 1;

        // Input: 4 patches in 2x2 grid, each with 1 channel:
        // [top-left (0), top-right (1)]
        // [bot-left (2), bot-right (3)]
        var input = new float[] { 10f, 20f, 30f, 40f };
        var output = new float[1 * 4 * inDim]; // 1 merged patch with 4 channels

        VisionOps.PixelShuffle2x2(input, gridY, gridX, inDim, output);

        // Expected merged order: [top-left, top-right, bot-left, bot-right]
        Assert.Equal(10f, output[0]);
        Assert.Equal(20f, output[1]);
        Assert.Equal(30f, output[2]);
        Assert.Equal(40f, output[3]);
    }

    [Fact]
    public void RmsNorm_NormalizesVectorAndAppliesGain()
    {
        int nTokens = 1;
        int embd = 4;
        var x = new float[] { 1f, 2f, 3f, 4f };
        var weight = new float[] { 2f, 2f, 2f, 2f };

        // RMS = sqrt((1 + 4 + 9 + 16)/4 + 1e-6) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
        // Normed x[0] = (1 / 2.7386) * 2 ≈ 0.7303
        fixed (float* pW = weight)
        {
            VisionOps.RmsNorm(x, nTokens, embd, pW, 1e-6f);
        }

        Assert.InRange(x[0], 0.72f, 0.74f);
        Assert.InRange(x[3], 2.91f, 2.93f);
    }

    [Fact]
    public void Activations_MatchAnalyticalDefinitions()
    {
        var vals = new float[] { -2.0f, 0.0f, 2.0f };

        // GELU(0) = 0, GELU(2) ≈ 1.909, GELU(-2) ≈ -0.045
        var gelu = (float[])vals.Clone();
        VisionOps.Gelu(gelu);
        Assert.InRange(gelu[0], -0.05f, -0.04f);
        Assert.Equal(0f, gelu[1]);
        Assert.InRange(gelu[2], 1.95f, 1.96f);

        // SiLU(0) = 0, SiLU(2) = 2 / (1 + e^-2) ≈ 1.7616
        var silu = (float[])vals.Clone();
        VisionOps.Silu(silu);
        Assert.Equal(0f, silu[1]);
        Assert.InRange(silu[2], 1.75f, 1.77f);
    }

    [Fact]
    public void ApplyMRoPE_PreservesVectorNormsWhileModifyingAngles()
    {
        int px = 2, py = 2;
        int heads = 2, headDim = 8;
        int total = px * py * heads * headDim;

        var q = new float[total];
        var k = new float[total];
        for (int i = 0; i < total; i++)
        {
            q[i] = (i % 7) + 1.0f;
            k[i] = (i % 5) + 1.0f;
        }

        float normQBefore = TensorPrimitives.SumOfSquares(q.AsSpan());
        float normKBefore = TensorPrimitives.SumOfSquares(k.AsSpan());

        VisionOps.ApplyMRoPE(q, k, px, py, heads, headDim, 10000.0f);

        float normQAfter = TensorPrimitives.SumOfSquares(q.AsSpan());
        float normKAfter = TensorPrimitives.SumOfSquares(k.AsSpan());

        // 2D Rotation is an isometric orthogonal transformation -> preserves L2 norm exactly
        Assert.InRange(normQAfter, normQBefore * 0.999f, normQBefore * 1.001f);
        Assert.InRange(normKAfter, normKBefore * 0.999f, normKBefore * 1.001f);
    }
}
