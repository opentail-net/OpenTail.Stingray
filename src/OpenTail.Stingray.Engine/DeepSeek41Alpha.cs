using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- DeepSeek-V4.1 Flash ("deepseek41" GGUF architecture) implementation.
//
// Status as of 2026-10-03: V4.1 is a distinct architecture from V4, defined by:
//  - 40 layers, hidden dimension D = 5120, 64 attention heads, KV = 1, head_dim = 512
//  - q_lora_rank = 1280, o_lora_rank = 1024 with 8 output groups
//  - Compression ratios 0 / 1 / 2: [0, 0, 18 of ratio 2, 20 of ratio 1]
//  - MoE: 384 routed experts + 1 shared expert, top_k = 6, sqrtsoftplus routing, scale = 1.5
//  - Indexer: 8 index-source layers, candidate top_k = 2048, block size 8, final top_k = 512,
//    sliding window 128, YaRN with 64 RoPE dims
//  - Engram Subsystem: Embedded N-gram memory on layers 1 and 14:
//    max N-gram = 4, 8 heads x 256 dim = 2048, compressed vocab 99,092, projected to 5120
//  - NextN / MTP: 3 NextN draft-head layers
//
// "deepseek41" is NOT admitted in ModelCompatibility.cs per CLAUDE.md rule 14.
// ============================================================================================

/// <summary>
/// Hyperparameters for DeepSeek-V4.1 Flash (deepseek41).
/// </summary>
public sealed record DeepSeek41Hyperparams
{
    public int NumLayerNextn { get; init; } = 3;
    public int NumLayerAll { get; init; } = 43; // 40 trunk + 3 MTP
    public int NumLayer => NumLayerAll - NumLayerNextn; // 40

    public int EmbedDim { get; init; } = 5120;
    public int NumHeads { get; init; } = 64;
    public int HeadDim { get; init; } = 512;
    public int RopeDim { get; init; } = 64;
    public int NopeDim => HeadDim - RopeDim; // 448

    public int QLoraRank { get; init; } = 1280;
    public int OutputLoraRank { get; init; } = 1024;
    public int OutputGroupCount { get; init; } = 8;

    public int NumExperts { get; init; } = 384;
    public int NumExpertsUsed { get; init; } = 6;
    public int ExpertSharedCount { get; init; } = 1;
    public float ExpertWeightsScale { get; init; } = 1.5f;
    public bool ExpertWeightsNorm { get; init; } = true;
    public int ExpertFeedForwardLength { get; init; } = 2304;   // official moe_intermediate_size

    public int HyperConnectionMultiplier { get; init; } = 4;
    public int HyperConnectionSinkhornIterations { get; init; } = 20;
    public float HyperConnectionEpsilon { get; init; } = 1e-6f;

    public int IndexerNumHeads { get; init; } = 32;        // official index_n_heads
    public int IndexerHeadSize { get; init; } = 128;
    public int IndexerCandidateTopK { get; init; } = 2048;
    public int IndexerBlockSize { get; init; } = 8;
    public int IndexerFinalTopK { get; init; } = 512;

    /// <summary>YaRN rope scaling factor (official config: 16). Not applied by the forward pass yet.</summary>
    public float RopeScalingFactor { get; init; } = 1f;
    public int SlidingWindow { get; init; } = 128;

    // Engram parameters (layers 1 and 14)
    public IReadOnlyList<int> EngramLayers { get; init; } = [1, 14];
    public int EngramMaxNgram { get; init; } = 4;
    public int EngramHeads { get; init; } = 8;
    public int EngramHeadDim { get; init; } = 256;
    public int EngramEmbeddingDim => EngramHeads * EngramHeadDim; // 2048
    public int EngramCompressedVocab { get; init; } = 99092;

    public float RmsNormEps { get; init; } = 1e-6f;
    public float RopeFreqBase { get; init; } = 10000f;
    public float CompressRopeFreqBase { get; init; } = 10000f;

    /// <summary>
    /// Compression ratios: 0, 0, 18 of ratio 2, 20 of ratio 1 (40 layers total).
    /// </summary>
    public IReadOnlyList<int> CompressRatios { get; init; } = BuildDefaultCompressRatios();

    public static int[] BuildDefaultCompressRatios()
    {
        var ratios = new int[40];
        ratios[0] = 0;
        ratios[1] = 0;
        for (int i = 2; i < 20; i++) ratios[i] = 2;  // 18 layers of ratio 2
        for (int i = 20; i < 40; i++) ratios[i] = 1; // 20 layers of ratio 1
        return ratios;
    }

    public static DeepSeek41Hyperparams FromGgufMetadata(
        IReadOnlyDictionary<string, object> metadata,
        string arch = "deepseek41")
    {
        ulong GetUlong(string key, ulong def) =>
            metadata.TryGetValue(key, out var v) && v is ulong u ? u : def;
        int GetInt(string key, int def) => (int)GetUlong(key, (ulong)def);
        float GetFloat(string key, float def) =>
            metadata.TryGetValue(key, out var v) && v is float f ? f : def;

        int numLayerAll = GetInt($"{arch}.block_count", 43);
        int numLayerNextn = GetInt($"{arch}.nextn_predict_layers", 3);
        int embedDim = GetInt($"{arch}.embedding_length", 5120);
        int numHeads = GetInt($"{arch}.attention.head_count", 64);
        int headDim = GetInt($"{arch}.attention.key_length", 512);
        int ropeDim = GetInt($"{arch}.rope.dimension_count", 64);

        int qLoraRank = GetInt($"{arch}.attention.q_lora_rank", 1280);
        int outLoraRank = GetInt($"{arch}.attention.output_lora_rank", 1024);
        int outGroupCount = GetInt($"{arch}.attention.output_group_count", 8);

        int numExperts = GetInt($"{arch}.expert_count", 384);
        int numExpertsUsed = GetInt($"{arch}.expert_used_count", 6);
        float expertScale = GetFloat($"{arch}.expert_weights_scale", 1.5f);
        bool expertNorm = metadata.TryGetValue($"{arch}.expert_weights_norm", out var n) && n is bool b ? b : true;

        int[] compressRatios;
        if (metadata.TryGetValue($"{arch}.attention.compress_ratios", out var crObj) && crObj is Array arr)
        {
            compressRatios = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                object? item = arr.GetValue(i);
                compressRatios[i] = item is ulong ul ? (int)ul : Convert.ToInt32(item);
            }
        }
        else
        {
            compressRatios = BuildDefaultCompressRatios();
        }

        int engramMaxNgram = GetInt($"{arch}.engram.max_ngram", 4);
        int engramHeads = GetInt($"{arch}.engram.head_count", 8);
        int engramHeadDim = GetInt($"{arch}.engram.head_dim", 256);
        int engramCompressedVocab = GetInt($"{arch}.engram.compressed_vocab_size", 99092);

        return new DeepSeek41Hyperparams
        {
            NumLayerAll = numLayerAll,
            NumLayerNextn = numLayerNextn,
            EmbedDim = embedDim,
            NumHeads = numHeads,
            HeadDim = headDim,
            RopeDim = ropeDim,
            QLoraRank = qLoraRank,
            OutputLoraRank = outLoraRank,
            OutputGroupCount = outGroupCount,
            NumExperts = numExperts,
            ExpertFeedForwardLength = GetInt($"{arch}.expert_feed_forward_length", 2304),
            IndexerNumHeads = GetInt($"{arch}.attention.indexer.head_count", 32),
            IndexerHeadSize = GetInt($"{arch}.attention.indexer.key_length", 128),
            IndexerFinalTopK = GetInt($"{arch}.attention.indexer.top_k", 512),
            RopeScalingFactor = GetFloat($"{arch}.rope.scaling.factor", 1f),
            NumExpertsUsed = numExpertsUsed,
            ExpertWeightsScale = expertScale,
            ExpertWeightsNorm = expertNorm,
            CompressRatios = compressRatios,
            EngramMaxNgram = engramMaxNgram,
            EngramHeads = engramHeads,
            EngramHeadDim = engramHeadDim,
            EngramCompressedVocab = engramCompressedVocab,
        };
    }
}

/// <summary>
/// Engram Subsystem for DeepSeek-V4.1: embedded N-gram memory lookup and projection.
/// Maps n-gram token history (1..4) to compressed vocab hashes, gathers embeddings,
/// and projects to trunk hidden dimension.
/// </summary>
public sealed class DeepSeek41Engram
{
    private readonly int _maxNgram;
    private readonly int _compressedVocab;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _totalDim;
    private readonly int _embedDim;
    private readonly int[] _tokenHistory;
    private int _historyCount;

    public DeepSeek41Engram(DeepSeek41Hyperparams hp)
    {
        _maxNgram = hp.EngramMaxNgram;
        _compressedVocab = hp.EngramCompressedVocab;
        _heads = hp.EngramHeads;
        _headDim = hp.EngramHeadDim;
        _totalDim = hp.EngramEmbeddingDim; // 2048
        _embedDim = hp.EmbedDim;           // 5120
        _tokenHistory = new int[_maxNgram];
        _historyCount = 0;
    }

    /// <summary>
    /// Computes polynomial rolling hash for an N-gram of tokens into compressed vocabulary range [0, compressedVocab - 1].
    /// </summary>
    public static int ComputeNgramHash(ReadOnlySpan<int> tokens, int compressedVocab)
    {
        const ulong multiplier = 1000003UL;
        ulong h = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            h = (h * multiplier + (ulong)(tokens[i] & 0x7FFFFFFF)) % (ulong)compressedVocab;
        }
        return (int)h;
    }

    /// <summary>
    /// Appends a new token to the N-gram history ring buffer.
    /// </summary>
    public void AppendToken(int tokenId)
    {
        if (_historyCount < _maxNgram)
        {
            _tokenHistory[_historyCount++] = tokenId;
        }
        else
        {
            for (int i = 0; i < _maxNgram - 1; i++)
            {
                _tokenHistory[i] = _tokenHistory[i + 1];
            }
            _tokenHistory[_maxNgram - 1] = tokenId;
        }
    }

    /// <summary>
    /// Computes the active N-gram hash keys (1-gram through max 4-gram).
    /// Returns the number of valid N-grams available in current context.
    /// </summary>
    public int GetActiveNgramHashes(Span<int> hashesOut)
    {
        int count = Math.Min(_historyCount, _maxNgram);
        for (int k = 1; k <= count; k++)
        {
            // k-gram ending at current token
            int start = _historyCount - k;
            var slice = _tokenHistory.AsSpan(start, k);
            hashesOut[k - 1] = ComputeNgramHash(slice, _compressedVocab);
        }
        return count;
    }

    /// <summary>
    /// Looks up embeddings for active N-grams from the Engram table and projects to hidden dimension.
    /// </summary>
    public unsafe void LookupAndProject(
        float* engramTablePtr, // [compressedVocab, totalDim]
        float* projWeightPtr,  // [embedDim, totalDim]
        Span<float> residualOut)
    {
        Span<int> hashes = stackalloc int[_maxNgram];
        int ngramCount = GetActiveNgramHashes(hashes);
        if (ngramCount == 0) return;

        Span<float> combined = stackalloc float[_totalDim];
        combined.Clear();

        float scale = 1.0f / ngramCount;
        for (int i = 0; i < ngramCount; i++)
        {
            int h = hashes[i];
            float* row = engramTablePtr + (long)h * _totalDim;
            for (int d = 0; d < _totalDim; d++)
            {
                combined[d] += row[d] * scale;
            }
        }

        // Project combined [2048] -> residual [5120]
        fixed (float* cPtr = combined, outPtr = residualOut)
        {
            SimdKernels.MatVecF32(outPtr, projWeightPtr, null, cPtr, _embedDim, _totalDim);
        }
    }

    /// <summary>Resets the token history ring buffer.</summary>
    public void Reset()
    {
        _historyCount = 0;
        Array.Clear(_tokenHistory, 0, _tokenHistory.Length);
    }
}

/// <summary>
/// Computational graph primitives specific to DeepSeek-V4.1.
/// </summary>
public static class DeepSeek41Graph
{
    /// <summary>
    /// SqrtSoftplus gate with scale 1.5: score = 1.5 * sqrt(softplus(logit)).
    /// </summary>
    public static void SqrtSoftplusGate(ReadOnlySpan<float> logits, float scale, Span<float> scores)
    {
        for (int i = 0; i < logits.Length; i++)
        {
            float x = logits[i];
            float sp = x > 20f ? x : (x < -30f ? 0f : MathF.Log(1f + MathF.Exp(x)));
            scores[i] = scale * MathF.Sqrt(sp);
        }
    }

    /// <summary>
    /// Selects candidate blocks with block size 8 and candidate top-k (2048),
    /// before final top-k selection (512).
    /// </summary>
    public static int[] SelectCandidateBlocks(
        ReadOnlySpan<float> blockScores, int candidateTopK, int finalTopK, int blockSize = 8)
    {
        int totalBlocks = blockScores.Length;
        if (totalBlocks == 0) return [];

        var scoresArr = blockScores.ToArray();
        var indices = new int[totalBlocks];
        for (int i = 0; i < totalBlocks; i++) indices[i] = i;

        // Partial sort or full sort descending by score
        Array.Sort(indices, (a, b) => scoresArr[b].CompareTo(scoresArr[a]));

        int kCandidates = Math.Min(candidateTopK, totalBlocks);
        int kFinal = Math.Min(finalTopK, kCandidates);
        var result = new int[kFinal];
        Array.Copy(indices, result, kFinal);
        return result;
    }
}
