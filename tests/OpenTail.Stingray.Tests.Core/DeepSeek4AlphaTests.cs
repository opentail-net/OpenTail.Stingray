using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// ALPHA/UNTESTED-AGAINST-REAL-WEIGHTS tests for DeepSeek4Graph (DeepSeek4Alpha.cs). These are
/// synthetic invariant/shape checks only -- they confirm the ported math is internally
/// self-consistent (e.g. Sinkhorn normalization actually produces a doubly-stochastic matrix),
/// NOT that it matches DeepSeek-V4's real behavior or the llama.cpp reference numerically. See
/// docs/058-deepseek-full-lineage-implementation-plan.md Phase 0.
/// </summary>
public class DeepSeek4AlphaTests
{
    [Fact]
    public void HyperConnectionSinkhorn_ProducesDoublyStochasticMatrix()
    {
        // Random-ish 4x4 input (hc==4, the only value the reference asserts -- deepseek4.cpp:362).
        float[] comb =
        [
            0.3f, -0.7f, 1.2f, 0.05f,
            -0.4f, 0.9f, -0.1f, 0.6f,
            0.2f, 0.2f, -0.3f, 1.1f,
            -0.6f, 0.4f, 0.8f, -0.2f,
        ];

        DeepSeek4Graph.HyperConnectionSinkhorn(comb, hc: 4, iterations: 20, eps: 1e-6f);

        for (int r = 0; r < 4; r++)
        {
            float rowSum = 0f;
            for (int c = 0; c < 4; c++) rowSum += comb[r * 4 + c];
            Assert.True(Math.Abs(rowSum - 1f) < 1e-3f, $"row {r} sum was {rowSum}, expected ~1");
        }

        for (int c = 0; c < 4; c++)
        {
            float colSum = 0f;
            for (int r = 0; r < 4; r++) colSum += comb[r * 4 + c];
            Assert.True(Math.Abs(colSum - 1f) < 1e-3f, $"col {c} sum was {colSum}, expected ~1");
        }

        foreach (float v in comb)
        {
            Assert.True(v >= 0f, "doubly-stochastic entries must be non-negative");
        }
    }

    [Fact]
    public void HyperConnectionSinkhorn_SingleIteration_StillRowNormalizedAfterSoftmax()
    {
        // With iterations=1 only norm_cols runs after the row-softmax+eps step (deepseek4.cpp:
        // 340-344) -- rows are NOT guaranteed to sum to 1 in this case, only columns are. This
        // test pins that asymmetry so a future "helpful" refactor that makes both axes always
        // sum to 1 gets caught as a behavior change, not silently accepted.
        float[] comb = [1f, 0f, 0f, 1f];
        DeepSeek4Graph.HyperConnectionSinkhorn(comb, hc: 2, iterations: 1, eps: 1e-6f);

        float col0 = comb[0] + comb[2];
        float col1 = comb[1] + comb[3];
        Assert.True(Math.Abs(col0 - 1f) < 1e-3f);
        Assert.True(Math.Abs(col1 - 1f) < 1e-3f);
    }

    [Fact]
    public void HyperConnectionMixDown_WeightedSumMatchesManualComputation()
    {
        // hc=2, embedDim=3.
        float[] x = [1f, 2f, 3f, /* stream 1 */ 10f, 20f, 30f];
        float[] weights = [0.5f, 0.25f];
        var result = new float[3];

        DeepSeek4Graph.HyperConnectionMixDown(x, weights, hc: 2, embedDim: 3, result);

        Assert.Equal(1f * 0.5f + 10f * 0.25f, result[0], 3);
        Assert.Equal(2f * 0.5f + 20f * 0.25f, result[1], 3);
        Assert.Equal(3f * 0.5f + 30f * 0.25f, result[2], 3);
    }

    [Fact]
    public void HyperConnectionMixUp_RoundTripsMixDownWhenCombIsIdentity()
    {
        // If comb is the identity matrix (each stream only mixes from itself) and post==1 for the
        // shared x contribution, mixUp's src-loop term for dst==src should reproduce residual
        // exactly and the x-broadcast term should add x*post on top -- a cheap way to confirm the
        // dst/src axis convention documented on HyperConnectionMixUp is applied consistently with
        // itself (not necessarily with the reference, which needs real weights to confirm).
        int hc = 2, embedDim = 2;
        float[] x = [100f, 200f];
        float[] residual = [1f, 2f, /* stream1 */ 3f, 4f];
        float[] post = [1f, 1f];
        float[] comb = [1f, 0f, 0f, 1f]; // identity: dst==src only
        var result = new float[hc * embedDim];

        DeepSeek4Graph.HyperConnectionMixUp(x, residual, post, comb, hc, embedDim, result);

        // stream 0: x*post[0] + residual(stream0)*comb[0,0] + residual(stream1)*comb[0,1]
        Assert.Equal(100f + 1f, result[0], 3);
        Assert.Equal(200f + 2f, result[1], 3);
        // stream 1: x*post[1] + residual(stream0)*comb[1,0] + residual(stream1)*comb[1,1]
        Assert.Equal(100f + 3f, result[2], 3);
        Assert.Equal(200f + 4f, result[3], 3);
    }

    [Fact]
    public void LightningIndexerScore_ReluZeroesNegativeDotProducts()
    {
        // 1 head, headDim 2, 2 keys. Key 0's dot product with q is negative -> ReLU'd to 0,
        // contributing nothing regardless of its weight; key 1's is positive and should show up.
        float[] q = [1f, 0f];
        float[] k = [-1f, 0f, /* key1 */ 2f, 0f];
        float[] weights = [10f, 10f];
        float[] mask = [0f, 0f];
        var scores = new float[2];

        DeepSeek4Graph.LightningIndexerScore(q, k, weights, mask, numHeads: 1, headDim: 2, numKeys: 2, scores);

        Assert.Equal(0f, scores[0], 3);
        Assert.Equal(20f, scores[1], 3); // relu(1*2) * 10 = 20
    }

    [Fact]
    public void LightningIndexerScore_AppliesCausalMaskAdditively()
    {
        float[] q = [1f];
        float[] k = [1f, /* key1 */ 1f];
        float[] weights = [1f, 1f];
        float[] mask = [0f, float.NegativeInfinity]; // key 1 disallowed
        var scores = new float[2];

        DeepSeek4Graph.LightningIndexerScore(q, k, weights, mask, numHeads: 1, headDim: 1, numKeys: 2, scores);

        Assert.Equal(1f, scores[0], 3);
        Assert.True(float.IsNegativeInfinity(scores[1]));
    }

    [Fact]
    public void SelectTopKIndices_ReturnsHighestScoringIndicesDescending()
    {
        float[] scores = [0.1f, 0.9f, 0.5f, 0.3f, 0.7f];
        int[] top3 = DeepSeek4Graph.SelectTopKIndices(scores, 3);

        Assert.Equal(3, top3.Length);
        Assert.Equal(1, top3[0]); // 0.9
        Assert.Equal(4, top3[1]); // 0.7
        Assert.Equal(2, top3[2]); // 0.5
    }

    [Fact]
    public void HcaCompressBlock_ProducesUnitNormAfterRmsNorm()
    {
        // ratio=2 rows, headDim=4, ropeDim=0 (skip RoPE entirely to isolate the softmax+RMSNorm
        // math from the RoPE delegate). After RMSNorm with unit gain, sum-of-squares/headDim
        // should be ~1 (up to eps).
        float[] kv = [1f, 2f, 3f, 4f, /* row1 */ 5f, 6f, 7f, 8f];
        float[] score = [0.1f, 0.2f, 0.3f, 0.4f, /* row1 */ 0.5f, 0.1f, 0.2f, 0.9f];
        float[] normWeight = [1f, 1f, 1f, 1f];
        var result = new float[4];

        DeepSeek4Graph.HcaCompressBlock(
            kv, score, ratio: 2, headDim: 4, ropeDim: 0,
            normWeight, rmsNormEps: 1e-6f,
            ropeApply: (_, _, _, _) => { /* no-op: ropeDim is 0, never invoked */ },
            compressRopeFreqBase: 10000f, blockPosition: 0f,
            result);

        float ss = 0f;
        foreach (float v in result) ss += v * v;
        float meanSquare = ss / 4;
        Assert.True(Math.Abs(meanSquare - 1f) < 1e-2f, $"post-RMSNorm mean square was {meanSquare}, expected ~1");
    }

    [Fact]
    public void FromGgufMetadata_ReadsDeepSeek4Keys()
    {
        var metadata = new Dictionary<string, object>
        {
            ["deepseek4.nextn_predict_layers"] = (ulong)1,
            ["deepseek4.attention.q_lora_rank"] = (ulong)1536,
            ["deepseek4.expert_feed_forward_length"] = (ulong)2048,
            ["deepseek4.expert_shared_count"] = (ulong)1,
            ["deepseek4.expert_weights_scale"] = 2.5f,
            ["deepseek4.expert_weights_norm"] = true,
            ["deepseek4.attention.indexer.head_count"] = (ulong)32,
            ["deepseek4.attention.indexer.key_length"] = (ulong)128,
            ["deepseek4.attention.indexer.top_k"] = (ulong)2048,
            ["deepseek4.attention.output_group_count"] = (ulong)2,
            ["deepseek4.attention.output_lora_rank"] = (ulong)512,
            ["deepseek4.hyper_connection.count"] = (ulong)4,
            ["deepseek4.hyper_connection.sinkhorn_iterations"] = (ulong)3,
            ["deepseek4.hash_layer_count"] = (ulong)1,
            ["deepseek4.attention.compress_ratios"] = new object[] { 0UL, 4UL, 128UL },
        };

        var hp = DeepSeek4Hyperparams.FromGgufMetadata(
            metadata, "deepseek4", numLayerAll: 4, embedDim: 7168, numHeads: 128, headDim: 192,
            ropeDim: 64, numExperts: 256, numExpertsUsed: 8);

        Assert.Equal(1, hp.NumLayerNextn);
        Assert.Equal(3, hp.NumLayer); // 4 total - 1 nextn
        Assert.Equal(1536, hp.QLoraRank);
        Assert.Equal(2.5f, hp.ExpertWeightsScale);
        Assert.True(hp.ExpertWeightsNorm);
        Assert.Equal(32, hp.IndexerNumHeads);
        Assert.Equal(2, hp.OutputGroupCount);
        Assert.Equal(3, hp.HyperConnectionSinkhornIterations);
        Assert.Equal([0, 4, 128], hp.CompressRatios);
    }

    [Fact]
    public void FromGgufMetadata_ThrowsWhenCompressRatiosShorterThanTrunkLayerCount()
    {
        var metadata = new Dictionary<string, object>
        {
            ["deepseek4.attention.compress_ratios"] = new object[] { 0UL }, // only 1, need 3
        };

        Assert.Throws<InvalidOperationException>(() =>
            DeepSeek4Hyperparams.FromGgufMetadata(
                metadata, "deepseek4", numLayerAll: 3, embedDim: 128, numHeads: 4, headDim: 32,
                ropeDim: 16, numExperts: 8, numExpertsUsed: 2));
    }

    [Fact]
    public void CompressedState_PersistThenGetBlock_RoundTrips()
    {
        var state = new DeepSeek4CompressedLayerState(headDim: 3);
        state.Persist([1f, 2f, 3f], [0.1f, 0.2f, 0.3f]);
        state.Persist([4f, 5f, 6f], [0.4f, 0.5f, 0.6f]);

        Assert.Equal(2, state.BlockCount);

        var (kv0, score0) = state.GetBlock(0);
        Assert.Equal([1f, 2f, 3f], kv0.ToArray());
        Assert.Equal([0.1f, 0.2f, 0.3f], score0.ToArray());

        var (kv1, _) = state.GetBlock(1);
        Assert.Equal([4f, 5f, 6f], kv1.ToArray());
    }

    [Fact]
    public void CompressedState_CopyRange_GathersContiguousBlocks()
    {
        var state = new DeepSeek4CompressedLayerState(headDim: 2);
        state.Persist([1f, 1f], [0f, 0f]);
        state.Persist([2f, 2f], [0f, 0f]);
        state.Persist([3f, 3f], [0f, 0f]);

        var kvOut = new float[2 * 2];
        var scoreOut = new float[2 * 2];
        state.CopyRange(startBlock: 1, count: 2, kvOut, scoreOut);

        Assert.Equal([2f, 2f, 3f, 3f], kvOut);
    }

    [Fact]
    public void CompressedState_GetBlock_OutOfRangeThrows()
    {
        var state = new DeepSeek4CompressedLayerState(headDim: 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => state.GetBlock(0));
    }

    [Fact]
    public void DeepSeek4CompressedState_AllocatesCsaHcaLidPerLayerRatio()
    {
        // ratios: layer0=0 (none), layer1=4 (CSA+LID), layer2=128 (HCA only).
        var state = new DeepSeek4CompressedState(numLayers: 3, headDim: 8, indexerHeadDim: 4, compressRatios: [0, 4, 128]);

        Assert.Null(state.Csa(0));
        Assert.Null(state.Hca(0));
        Assert.Null(state.Lid(0));

        Assert.NotNull(state.Csa(1));
        Assert.NotNull(state.Lid(1));
        Assert.Null(state.Hca(1));

        Assert.NotNull(state.Hca(2));
        Assert.Null(state.Csa(2));
        Assert.Null(state.Lid(2));
    }

    [Fact]
    public void DeepSeek4CompressedState_Clear_ResetsAllPersistedBlocks()
    {
        var state = new DeepSeek4CompressedState(numLayers: 1, headDim: 2, indexerHeadDim: 2, compressRatios: [4]);
        state.Csa(0)!.Persist([1f, 2f], [0f, 0f]);
        Assert.Equal(1, state.Csa(0)!.BlockCount);

        state.Clear();
        Assert.Equal(0, state.Csa(0)!.BlockCount);
    }

    [Theory]
    [InlineData(1, 4, 0, 0)]
    [InlineData(1, 4, 3, 3)]
    [InlineData(2, 4, 0, 4)]
    [InlineData(2, 4, 3, 7)]
    public void OverlapPrevRowIndex_MatchesImmediatelyPrecedingBlockHypothesis(int blockIndex, int ratio, int r, int expected)
    {
        // Block k>0's prev half is exactly block (k-1)'s own row range [((k-1)*ratio) .. (k*ratio)-1].
        Assert.Equal(expected, DeepSeek4Graph.OverlapPrevRowIndex(blockIndex, ratio, r));
    }

    [Theory]
    [InlineData(0, 4, 0)]
    [InlineData(0, 4, 3)]
    public void OverlapPrevRowIndex_IsNegativeForBlockZero(int blockIndex, int ratio, int r)
    {
        // Block 0 has no history -> a negative index (the caller substitutes the reference's
        // synthetic zero-KV/-inf-score row for any negative result -- see FinalizeOverlapBlock's
        // `prevRowIndex >= 0` check; the exact negative VALUE is not itself meaningful).
        Assert.True(DeepSeek4Graph.OverlapPrevRowIndex(blockIndex, ratio, r) < 0);
    }

    [Theory]
    [InlineData(0, 4, 0, 0)]
    [InlineData(0, 4, 3, 3)]
    [InlineData(1, 4, 0, 4)]
    [InlineData(2, 4, 0, 8)]
    public void OverlapCurRowIndex_MatchesThisBlocksOwnRowRange(int blockIndex, int ratio, int r, int expected)
    {
        Assert.Equal(expected, DeepSeek4Graph.OverlapCurRowIndex(blockIndex, ratio, r));
    }

    [Fact]
    public void OverlapRowIndices_ConsecutiveBlocksTileWithoutGapOrOverlap()
    {
        // Block k's cur range and block (k+1)'s prev range must be the SAME 4 raw-token
        // positions -- that's the entire point of the overlap scheme (block k+1's compression
        // reaches back into block k's own raw rows without recomputation). If this ever fails,
        // the two index functions have drifted out of sync with each other.
        const int ratio = 4;
        for (int k = 0; k < 5; k++)
        {
            for (int r = 0; r < ratio; r++)
            {
                int curOfK = DeepSeek4Graph.OverlapCurRowIndex(k, ratio, r);
                int prevOfKPlus1 = DeepSeek4Graph.OverlapPrevRowIndex(k + 1, ratio, r);
                Assert.Equal(curOfK, prevOfKPlus1);
            }
        }
    }
}
