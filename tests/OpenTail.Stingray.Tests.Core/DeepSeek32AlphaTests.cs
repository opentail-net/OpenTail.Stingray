using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// ALPHA/UNTESTED-AGAINST-REAL-WEIGHTS tests for DeepSeek32Hyperparams (DeepSeek32Alpha.cs).
/// Pure metadata-parsing checks only -- see docs/058-deepseek-full-lineage-implementation-plan.md
/// Phase 1.
/// </summary>
public class DeepSeek32AlphaTests
{
    [Fact]
    public void FromGgufMetadata_ReadsDeepSeek32Keys()
    {
        var metadata = new Dictionary<string, object>
        {
            ["deepseek32.nextn_predict_layers"] = (ulong)1,
            ["deepseek32.expert_feed_forward_length"] = (ulong)2048,
            ["deepseek32.leading_dense_block_count"] = (ulong)3,
            ["deepseek32.expert_weights_scale"] = 2.5f,
            ["deepseek32.expert_weights_norm"] = true,
            ["deepseek32.expert_shared_count"] = (ulong)1,
            ["deepseek32.attention.q_lora_rank"] = (ulong)1536,
            ["deepseek32.attention.kv_lora_rank"] = (ulong)512,
            ["deepseek32.attention.key_length_mla"] = (ulong)192,
            ["deepseek32.attention.value_length_mla"] = (ulong)128,
            ["deepseek32.attention.indexer.head_count"] = (ulong)64,
            ["deepseek32.attention.indexer.key_length"] = (ulong)128,
            ["deepseek32.attention.indexer.top_k"] = (ulong)2048,
            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]: raw GGUF value 0.0707 -> /0.1 -> 0.707.
            ["deepseek32.rope.scaling.yarn_log_multiplier"] = 0.0707f,
        };

        var hp = DeepSeek32Hyperparams.FromGgufMetadata(
            metadata, "deepseek32", numLayerAll: 61, embedDim: 7168, numHeads: 128, headDim: 192,
            ropeDim: 64, numExperts: 256, numExpertsUsed: 8);

        Assert.Equal(1, hp.NumLayerNextn);
        Assert.Equal(60, hp.NumLayer);
        Assert.Equal(3, hp.LeadingDenseBlockCount);
        Assert.Equal(2.5f, hp.ExpertWeightsScale);
        Assert.True(hp.ExpertWeightsNorm);
        Assert.Equal(1536, hp.QLoraRank);
        Assert.Equal(512, hp.KvLoraRank);
        Assert.Equal(192, hp.EmbedHeadKMlaOverride);
        Assert.Equal(192, hp.EffectiveHeadDimK);
        Assert.Equal(128, hp.EffectiveHeadDimV);
        Assert.Equal(64, hp.IndexerNumHeads);
        Assert.Equal(0.707f, hp.RopeYarnLogMul, 3);
    }

    [Fact]
    public void FromGgufMetadata_EffectiveHeadDim_FallsBackToPlainHeadDimWhenNoMlaOverride()
    {
        var metadata = new Dictionary<string, object>();
        var hp = DeepSeek32Hyperparams.FromGgufMetadata(
            metadata, "deepseek32", numLayerAll: 27, embedDim: 2048, numHeads: 16, headDim: 192,
            ropeDim: 64, numExperts: 64, numExpertsUsed: 6);

        Assert.Equal(0, hp.EmbedHeadKMlaOverride);
        Assert.Equal(192, hp.EffectiveHeadDimK); // falls back to plain HeadDim
        Assert.Equal(192, hp.EffectiveHeadDimV);
    }

    [Fact]
    public void FromGgufMetadata_YarnLogMul_ZeroWhenAbsent_NotDividedByZero()
    {
        var metadata = new Dictionary<string, object>();
        var hp = DeepSeek32Hyperparams.FromGgufMetadata(
            metadata, "deepseek32", numLayerAll: 27, embedDim: 2048, numHeads: 16, headDim: 192,
            ropeDim: 64, numExperts: 64, numExpertsUsed: 6);

        // Guards against a naive "always divide by 0.1" port producing 0/0.1=0 by coincidence
        // vs. a real bug that would divide a genuinely-absent value into something nonzero.
        Assert.Equal(0f, hp.RopeYarnLogMul);
    }
}
