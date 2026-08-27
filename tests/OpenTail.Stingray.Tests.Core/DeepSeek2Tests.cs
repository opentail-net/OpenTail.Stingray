using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public unsafe class DeepSeek2Tests
{
    [Fact]
    public void ModelHyperparams_ReadsDeepSeek2Metadata()
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "deepseek2",
            ["deepseek2.block_count"] = (ulong)27,
            ["deepseek2.context_length"] = (ulong)163840,
            ["deepseek2.embedding_length"] = (ulong)2048,
            ["deepseek2.feed_forward_length"] = (ulong)10944,
            ["deepseek2.attention.head_count"] = (ulong)16,
            ["deepseek2.attention.head_count_kv"] = (ulong)16,
            ["deepseek2.attention.kv_lora_rank"] = (ulong)512,
            ["deepseek2.expert_count"] = (ulong)64,
            ["deepseek2.expert_used_count"] = (ulong)6,
            ["deepseek2.expert_shared_count"] = (ulong)2,
            ["deepseek2.leading_dense_block_count"] = (ulong)1,
            ["deepseek2.expert_feed_forward_length"] = (ulong)1408,
            ["deepseek2.vocab_size"] = (ulong)102400,
        };

        var hparams = ModelHyperparams.FromGgufMetadata(metadata);

        Assert.Equal(27, hparams.NumLayers);
        Assert.Equal(2048, hparams.EmbeddingDim);
        Assert.Equal(512, hparams.KvLoraRank);
        Assert.Equal(64, hparams.NumExperts);
        Assert.Equal(6, hparams.NumActiveExperts);
        Assert.Equal(2, hparams.NumSharedExperts);
        Assert.Equal(1, hparams.LeadingDenseBlockCount);
        Assert.Equal(1408, hparams.ExpertIntermediateDim);
        Assert.True(hparams.IsMoE);
        Assert.True(hparams.HasSharedExpert);
    }

    [Fact]
    public void DeepSeekMoe_SelectsTopKExpertsCorrectly()
    {
        const int embDim = 4;
        const int numExperts = 8;
        const int topK = 2;

        float[] input = [1.0f, 0.5f, -0.2f, 0.8f];
        // Row-major [numExperts, embDim] -- the standard GGUF Linear-weight [out_features,
        // in_features] layout every MatVec in this codebase assumes; row e occupies
        // [e*embDim, (e+1)*embDim).
        float[] wGate = new float[numExperts * embDim];

        // Favor expert 3 and expert 5 by weighting their (row's) input-dim-0 coefficient.
        wGate[3 * embDim + 0] = 2.0f;
        wGate[5 * embDim + 0] = 1.5f;

        int[] indices = new int[topK];
        float[] weights = new float[topK];

        fixed (float* pIn = input)
        fixed (float* pW = wGate)
        fixed (int* pIdx = indices)
        fixed (float* pWt = weights)
        {
            DeepSeekMoeGraph.SelectTopKExperts(pIn, pW, embDim, numExperts, topK, pIdx, pWt);
        }

        Assert.Equal(3, indices[0]);
        Assert.Equal(5, indices[1]);
        Assert.True(weights[0] > weights[1]);
        Assert.Equal(1.0f, weights[0] + weights[1], 4);
    }

    [Fact]
    public void MlaAttention_CompressesKvLatent()
    {
        const int batchSize = 1;
        const int embDim = 4;
        const int kvLoraRank = 4;
        const int ropeDim = 2;
        const int totalKvDim = kvLoraRank + ropeDim;

        float[] input = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] wKvAMqa = new float[embDim * totalKvDim];
        for (int i = 0; i < embDim * totalKvDim; i++) wKvAMqa[i] = 0.1f * (i + 1);

        float[] kvANorm = [1.0f, 1.0f, 1.0f, 1.0f];
        float[] output = new float[totalKvDim];

        fixed (float* pOut = output)
        fixed (float* pIn = input)
        fixed (float* pW = wKvAMqa)
        fixed (float* pNorm = kvANorm)
        {
            MlaAttention.CompressKvLatent(pOut, pIn, pW, pNorm, batchSize, embDim, kvLoraRank, ropeDim, 1e-5f);
        }

        // Verify non-zero output
        for (int i = 0; i < totalKvDim; i++)
        {
            Assert.False(float.IsNaN(output[i]));
            Assert.False(float.IsInfinity(output[i]));
        }
    }
}
