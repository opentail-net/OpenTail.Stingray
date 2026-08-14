using System;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPassTests;

public class PrefillAttentionSeamTests
{
    /// <summary>
    /// With exactly one causal position, softmax contains one element and attention must return
    /// that position's V verbatim. This is an oracle-free invariant that simultaneously checks
    /// GQA/MQA head broadcast at the real Gemma 12B shapes: SWA uses 8 Q / 4 KV heads at 256 and
    /// global attention uses 8 Q / 1 KV head at 512. Together with PagedKvCache's production
    /// geometry test it names a bad V-plane stride without relying on another backend.
    /// </summary>
    [Theory]
    [InlineData(8, 4, 256)]
    [InlineData(8, 1, 512)]
    public unsafe void BatchedCausalAttention_PositionZeroReturnsVForEveryGqaGroup(
        int numHeads, int numKvHeads, int headDim)
    {
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        var q = new float[qDim];
        var k = new float[kvDim];
        var v = new float[kvDim];
        var output = new float[qDim];
        for (int kvHead = 0; kvHead < numKvHeads; kvHead++)
            for (int dimension = 0; dimension < headDim; dimension++)
                v[kvHead * headDim + dimension] = kvHead * 10_000f + dimension;

        fixed (float* qPtr = q)
        fixed (float* kPtr = k)
        fixed (float* vPtr = v)
        fixed (float* outputPtr = output)
            OpenTail.Stingray.Engine.ForwardPass.ComputeBatchedCausalAttention(
                qPtr, kPtr, vPtr, outputPtr, N: 1, startPos: 0, numHeads, numKvHeads, headDim,
                scale: 1f);

        int headsPerKvGroup = numHeads / numKvHeads;
        for (int head = 0; head < numHeads; head++)
        {
            int kvHead = head / headsPerKvGroup;
            for (int dimension = 0; dimension < headDim; dimension++)
                Assert.Equal(v[kvHead * headDim + dimension], output[head * headDim + dimension]);
        }
    }

    [Fact]
    public unsafe void BatchedCausalAttention_HandComputedReference_MatchesFirstPrinciples()
    {
        // 2 tokens, 1 head, headDim=4, scale=0.5
        int N = 2;
        int numHeads = 1;
        int numKvHeads = 1;
        int headDim = 4;
        int startPos = 0;
        float scale = 0.5f;

        // Q vectors: [N, numHeads * headDim] = [2, 4]
        float[] qHost = new float[]
        {
            1.0f, 0.0f, 2.0f, -1.0f, // Token 0
            0.0f, 2.0f, 1.0f,  1.0f  // Token 1
        };

        // K vectors: [N, numKvHeads * headDim] = [2, 4]
        float[] kHost = new float[]
        {
            2.0f, 1.0f, 0.0f, 1.0f, // Position 0
            1.0f, 0.0f, 2.0f, 0.0f  // Position 1
        };

        // V vectors: [N, numKvHeads * headDim] = [2, 4]
        float[] vHost = new float[]
        {
            1.0f, 2.0f, 3.0f, 4.0f, // Position 0
            5.0f, 6.0f, 7.0f, 8.0f  // Position 1
        };

        float[] actualAttnOutHost = new float[N * numHeads * headDim];

        fixed (float* pQ = qHost)
        fixed (float* pK = kHost)
        fixed (float* pV = vHost)
        fixed (float* pOut = actualAttnOutHost)
        {
            OpenTail.Stingray.Engine.ForwardPass.ComputeBatchedCausalAttention(
                pQ, pK, pV, pOut,
                N, startPos, numHeads, numKvHeads, headDim, scale);
        }

        // Expected outputs from first-principles hand calculation:
        // Token 0: [1.0, 2.0, 3.0, 4.0]
        // Token 1: [2.5101627, 3.5101627, 4.5101627, 5.5101627]
        float[] expectedToken0 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] expectedToken1 = new float[] { 2.5101627f, 3.5101627f, 4.5101627f, 5.5101627f };

        for (int d = 0; d < headDim; d++)
        {
            Assert.Equal(expectedToken0[d], actualAttnOutHost[d], precision: 4);
            Assert.Equal(expectedToken1[d], actualAttnOutHost[headDim + d], precision: 4);
        }
    }
}
