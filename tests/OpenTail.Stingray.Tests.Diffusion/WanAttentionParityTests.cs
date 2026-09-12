using System.Numerics.Tensors;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanAttentionParityTests
{
    [Fact]
    public void TiledMultiHeadAttention_MatchesDiffusionOps_SelfAttentionShape()
    {
        const int qSeq = 128;
        const int kvSeq = 128;
        const int numHeads = 12;
        const int headDim = 128;
        const int dim = numHeads * headDim;

        var random = new Random(42);
        var q = new float[qSeq * dim];
        var k = new float[kvSeq * dim];
        var v = new float[kvSeq * dim];

        for (int i = 0; i < q.Length; i++) q[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        var expected = DiffusionOps.MultiHeadAttention(q, k, v, qSeq, kvSeq, numHeads, headDim);
        var actual = new float[qSeq * dim];
        WanAttention.TiledMultiHeadAttention(q, k, v, actual, qSeq, kvSeq, numHeads, headDim);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-4f,
                $"Mismatch at index {i}: expected {expected[i]:F6}, actual {actual[i]:F6}, diff {MathF.Abs(expected[i] - actual[i]):E4}");
        }
    }

    [Fact]
    public void TiledMultiHeadAttention_MatchesDiffusionOps_CrossAttentionShape()
    {
        const int qSeq = 128;
        const int kvSeq = 64;
        const int numHeads = 12;
        const int headDim = 128;
        const int dim = numHeads * headDim;

        var random = new Random(1337);
        var q = new float[qSeq * dim];
        var k = new float[kvSeq * dim];
        var v = new float[kvSeq * dim];

        for (int i = 0; i < q.Length; i++) q[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        var expected = DiffusionOps.MultiHeadAttention(q, k, v, qSeq, kvSeq, numHeads, headDim);
        var actual = new float[qSeq * dim];
        WanAttention.TiledMultiHeadAttention(q, k, v, actual, qSeq, kvSeq, numHeads, headDim);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-4f,
                $"Mismatch at index {i}: expected {expected[i]:F6}, actual {actual[i]:F6}, diff {MathF.Abs(expected[i] - actual[i]):E4}");
        }
    }
}
