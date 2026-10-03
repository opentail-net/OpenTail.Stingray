using System;
using System.Collections.Generic;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed unsafe class DeepSeek41AlphaTests
{
    [Fact]
    public void Hyperparams_Dimensions_MatchV41Specification()
    {
        var hp = new DeepSeek41Hyperparams();

        Assert.Equal(40, hp.NumLayer);
        Assert.Equal(43, hp.NumLayerAll);
        Assert.Equal(3, hp.NumLayerNextn);

        Assert.Equal(5120, hp.EmbedDim);
        Assert.Equal(64, hp.NumHeads);
        Assert.Equal(512, hp.HeadDim);
        Assert.Equal(64, hp.RopeDim);
        Assert.Equal(448, hp.NopeDim);

        Assert.Equal(1280, hp.QLoraRank);
        Assert.Equal(1024, hp.OutputLoraRank);
        Assert.Equal(8, hp.OutputGroupCount);

        Assert.Equal(384, hp.NumExperts);
        Assert.Equal(6, hp.NumExpertsUsed);
        Assert.Equal(1, hp.ExpertSharedCount);
        Assert.Equal(1.5f, hp.ExpertWeightsScale);
        Assert.True(hp.ExpertWeightsNorm);

        Assert.Equal(4, hp.HyperConnectionMultiplier);
        Assert.Equal(20, hp.HyperConnectionSinkhornIterations);

        Assert.Equal(64, hp.IndexerNumHeads);
        Assert.Equal(128, hp.IndexerHeadSize);
        Assert.Equal(2048, hp.IndexerCandidateTopK);
        Assert.Equal(8, hp.IndexerBlockSize);
        Assert.Equal(512, hp.IndexerFinalTopK);

        // Engram specification
        Assert.Equal([1, 14], hp.EngramLayers);
        Assert.Equal(4, hp.EngramMaxNgram);
        Assert.Equal(8, hp.EngramHeads);
        Assert.Equal(256, hp.EngramHeadDim);
        Assert.Equal(2048, hp.EngramEmbeddingDim);
        Assert.Equal(99092, hp.EngramCompressedVocab);

        // Compression ratios: [0, 0, 18 of 2, 20 of 1]
        Assert.Equal(40, hp.CompressRatios.Count);
        Assert.Equal(0, hp.CompressRatios[0]);
        Assert.Equal(0, hp.CompressRatios[1]);
        for (int i = 2; i < 20; i++)
        {
            Assert.Equal(2, hp.CompressRatios[i]);
        }
        for (int i = 20; i < 40; i++)
        {
            Assert.Equal(1, hp.CompressRatios[i]);
        }
    }

    [Fact]
    public void FromGgufMetadata_ParsesV41Architecture()
    {
        var meta = new Dictionary<string, object>
        {
            ["deepseek41.block_count"] = (ulong)43,
            ["deepseek41.nextn_predict_layers"] = (ulong)3,
            ["deepseek41.embedding_length"] = (ulong)5120,
            ["deepseek41.attention.head_count"] = (ulong)64,
            ["deepseek41.attention.key_length"] = (ulong)512,
            ["deepseek41.rope.dimension_count"] = (ulong)64,
            ["deepseek41.attention.q_lora_rank"] = (ulong)1280,
            ["deepseek41.attention.output_lora_rank"] = (ulong)1024,
            ["deepseek41.attention.output_group_count"] = (ulong)8,
            ["deepseek41.expert_count"] = (ulong)384,
            ["deepseek41.expert_used_count"] = (ulong)6,
            ["deepseek41.expert_weights_scale"] = 1.5f,
        };

        var hp = DeepSeek41Hyperparams.FromGgufMetadata(meta, "deepseek41");

        Assert.Equal(40, hp.NumLayer);
        Assert.Equal(43, hp.NumLayerAll);
        Assert.Equal(384, hp.NumExperts);
        Assert.Equal(6, hp.NumExpertsUsed);
        Assert.Equal(1.5f, hp.ExpertWeightsScale);
        Assert.Equal(8, hp.OutputGroupCount);
        Assert.Equal(1024, hp.OutputLoraRank);
        Assert.Equal(1280, hp.QLoraRank);
    }

    [Fact]
    public void Graph_SqrtSoftplusGate_WithScale15()
    {
        float[] logits = [0f, 25f, -35f, 1f];
        float scale = 1.5f;
        var scores = new float[logits.Length];

        DeepSeek41Graph.SqrtSoftplusGate(logits, scale, scores);

        // x = 0: 1.5 * sqrt(ln(2)) ~ 1.5 * 0.8325546 = 1.24883
        float expected0 = scale * MathF.Sqrt(MathF.Log(2f));
        Assert.Equal(expected0, scores[0], 1e-4f);

        // x = 25: 1.5 * sqrt(25) = 1.5 * 5 = 7.5
        Assert.Equal(7.5f, scores[1], 1e-3f);

        // x = -35: ~0
        Assert.True(scores[2] >= 0f && scores[2] < 1e-5f);
    }

    [Fact]
    public void Engram_RollingHash_ComputesBoundedHashes()
    {
        int compressedVocab = 99092;

        // 1-gram: [100]
        int[] t1 = [100];
        int h1 = DeepSeek41Engram.ComputeNgramHash(t1, compressedVocab);
        Assert.Equal(100, h1);

        // 2-gram: [50, 100] -> (50 * 1000003 + 100) % 99092
        int[] t2 = [50, 100];
        int h2 = DeepSeek41Engram.ComputeNgramHash(t2, compressedVocab);
        ulong expectedH2 = (50UL * 1000003UL + 100UL) % (ulong)compressedVocab;
        Assert.Equal((int)expectedH2, h2);

        Assert.True(h1 >= 0 && h1 < compressedVocab);
        Assert.True(h2 >= 0 && h2 < compressedVocab);
    }

    [Fact]
    public void Engram_HistoryBuffer_MaintainsMax4Tokens()
    {
        var hp = new DeepSeek41Hyperparams();
        var engram = new DeepSeek41Engram(hp);

        Span<int> hashes = stackalloc int[4];

        // Token 1
        engram.AppendToken(10);
        int count1 = engram.GetActiveNgramHashes(hashes);
        Assert.Equal(1, count1);

        // Token 2
        engram.AppendToken(20);
        int count2 = engram.GetActiveNgramHashes(hashes);
        Assert.Equal(2, count2);

        // Token 3
        engram.AppendToken(30);
        int count3 = engram.GetActiveNgramHashes(hashes);
        Assert.Equal(3, count3);

        // Token 4
        engram.AppendToken(40);
        int count4 = engram.GetActiveNgramHashes(hashes);
        Assert.Equal(4, count4);

        // Token 5: should still return max 4 n-grams (ending in [20, 30, 40, 50])
        engram.AppendToken(50);
        int count5 = engram.GetActiveNgramHashes(hashes);
        Assert.Equal(4, count5);
    }

    [Fact]
    public void Engram_LookupAndProject_CombinesAndProjects()
    {
        var hp = new DeepSeek41Hyperparams
        {
            EngramCompressedVocab = 10,
            EngramHeads = 2,
            EngramHeadDim = 2, // total engram dim = 4
            EmbedDim = 6       // projected dim = 6
        };
        var engram = new DeepSeek41Engram(hp);
        engram.AppendToken(3);

        int totalDim = 4;
        int embedDim = 6;
        float[] table = new float[10 * totalDim];
        // Populate row for token 3
        for (int d = 0; d < totalDim; d++) table[3 * totalDim + d] = 1.0f;

        float[] proj = new float[embedDim * totalDim];
        Array.Fill(proj, 0.5f);

        float[] output = new float[embedDim];

        fixed (float* tPtr = table, pPtr = proj)
        {
            engram.LookupAndProject(tPtr, pPtr, output);
        }

        // Output should be non-zero and finite
        for (int i = 0; i < embedDim; i++)
        {
            Assert.True(output[i] > 0f);
            Assert.True(float.IsFinite(output[i]));
        }
    }

    [Fact]
    public void Graph_SelectCandidateBlocks_SelectsTopK()
    {
        float[] scores = [0.1f, 0.8f, 0.4f, 0.95f, 0.2f, 0.6f];
        int candidateTopK = 4;
        int finalTopK = 2;

        int[] selected = DeepSeek41Graph.SelectCandidateBlocks(scores, candidateTopK, finalTopK);

        Assert.Equal(2, selected.Length);
        Assert.Equal(3, selected[0]); // 0.95
        Assert.Equal(1, selected[1]); // 0.8
    }
}
