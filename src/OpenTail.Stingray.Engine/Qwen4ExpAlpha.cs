using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- Qwen 3.8 Flash Next ("qwen4exp" GGUF architecture) implementation.
//
// Status as of 2026-10-03: ported directly from llama.cpp reference
// (examples/llama.cpp/llama.cpp/src/models/qwen4exp.cpp, 1478 lines, upstream commit bed0a8566)
// and TensorSharp Models/Qwen4Exp/.
//
// "qwen4exp" is NOT admitted in ModelCompatibility.cs per CLAUDE.md rule 14. Real checkpoints
// (~72.5 GB minimum for UD-IQ1_S, ~180B total params) do not fit in this machine's 64 GB RAM.
// Verification is gated on synthetic component and full-stack parity tests.
//
// Architecture:
// - 48 layers in strict 3:1 pattern (full_attention_interval = 4):
//   - (il + 1) % 4 != 0 (36 layers): GatedDeltaNet (GDN) linear attention + MoE
//   - (il + 1) % 4 == 0 (12 layers: 3, 7, 11, ..., 47): Qwen Sparse Attention (QSA) + MoE
// - 4-stream GatedResidual hyper-connections (low-rank bottleneck 320, 2*sigmoid injection combine).
// - PLE n-gram embedding block at layer 2 with signed-sqrt gate and dilated depthwise causal 1D conv.
// - QSA indexer with key pooling over compress_ratio blocks, top-k block selection, and gated Q.
// - 512 routed experts (top-10) + 1 shared expert (intermediate width 640).
// - GDN state recurrence in FP32 (mamba_ssm_dtype = float32).
// ============================================================================================

/// <summary>
/// Qwen4Exp-specific hyperparameters read from GGUF metadata, mirroring
/// <c>llama_model_qwen4exp::load_arch_hparams</c> (qwen4exp.cpp:26-170).
/// </summary>
public sealed record Qwen4ExpHyperparams
{
    public int EmbedDim { get; init; } = 2560;
    public int NumHeads { get; init; } = 32;
    public int NumHeadsKv { get; init; } = 8;
    public int HeadDim { get; init; } = 128;
    public int NumLayer { get; init; } = 48;
    public int VocabSize { get; init; } = 152064;
    public int ContextLength { get; init; } = 4096;

    public int NumLayerNextn { get; init; }
    public int ExpertFeedForwardLength { get; init; }
    public int ExpertSharedFeedForwardLength { get; init; }

    /// <summary>Routed expert count (<c>expert_count</c>); 0 means "no routed experts" (synthetic fixtures only).</summary>
    public int ExpertCount { get; init; }
    public int ExpertUsedCount { get; init; } = 10;

    /// <summary>Routed-weight scale (<c>expert_weights_scale</c>); 0 = unscaled, as in llama.cpp's build_moe_ffn.</summary>
    public float ExpertWeightsScale { get; init; }
    public float RmsNormEps { get; init; } = 1e-6f;
    public IReadOnlyList<int>? RopeDimensionSections { get; init; }
    public int SsmConvKernel { get; init; }
    public int SsmInnerSize { get; init; }
    public int SsmStateSize { get; init; }
    public int SsmDtRank { get; init; }
    public int SsmGroupCount { get; init; }
    public int HyperConnectionCount { get; init; } = 4;
    public int HyperConnectionLowRank { get; init; } = 320;
    public int IndexerHeadCount { get; init; }
    public int IndexerKeyLength { get; init; }
    public int IndexerTopK { get; init; }
    public int IndexerKPool { get; init; }
    public IReadOnlyList<int>? CompressRatios { get; init; }
    public IReadOnlyList<int>? PleLayers { get; init; }
    public int PleNgramSize { get; init; }
    public int PleHeadsPerNgram { get; init; }
    public int PleConvKernel { get; init; }
    public int PleEosTokenId { get; init; }
    public int PleImageTokenId { get; init; }
    public int PleEmbeddingLengthPerLayer { get; init; }
    public IReadOnlyList<ulong>? PleLayerMultipliers { get; init; }
    public IReadOnlyList<uint>? PleHeadOffsets { get; init; }
    public IReadOnlyList<uint>? PleHeadVocabSizes { get; init; }
    public int FullAttentionInterval { get; init; } = 4;
    public IReadOnlyList<bool>? RecurrentLayers { get; init; }

    /// <summary>Whether layer <paramref name="layerIndex"/> is a recurrent (GDN) linear attention layer.</summary>
    public bool IsRecurrentLayer(int layerIndex)
    {
        if (RecurrentLayers != null && layerIndex < RecurrentLayers.Count)
        {
            return RecurrentLayers[layerIndex];
        }
        return (layerIndex + 1) % FullAttentionInterval != 0;
    }

    /// <summary>Whether layer <paramref name="layerIndex"/> has the PLE block injected.</summary>
    public bool IsPleLayer(int layerIndex)
    {
        if (PleLayers == null) return false;
        for (int i = 0; i < PleLayers.Count; i++)
        {
            if (PleLayers[i] == layerIndex) return true;
        }
        return false;
    }

    /// <summary>
    /// Reads <c>qwen4exp</c> metadata keys per <c>load_arch_hparams</c> (qwen4exp.cpp:26-170).
    /// </summary>
    public static Qwen4ExpHyperparams FromMetadata(IReadOnlyDictionary<string, object> metadata, int totalLayers)
    {
        const string arch = "qwen4exp";

        var compressRatios = GetIntArray(metadata, $"{arch}.attention.compress_ratios")
            ?? GetIntArray(metadata, $"{arch}.attention.compress_ratio");

        int kpool = 0;
        if (compressRatios != null)
        {
            for (int i = 0; i < compressRatios.Count; i++)
            {
                int r = compressRatios[i];
                if (r > 0)
                {
                    kpool = r;
                    break;
                }
            }
        }

        var pleLayers = GetIntArray(metadata, $"{arch}.ple.layers");
        int pleNgramSize = GetInt(metadata, $"{arch}.ple.ngram_size");
        int pleHeadsPerNgram = GetInt(metadata, $"{arch}.ple.heads_per_ngram");
        int pleConvKernel = GetInt(metadata, $"{arch}.ple.conv_kernel");
        int pleEosTokenId = GetInt(metadata, $"{arch}.ple.eos_token_id");
        int pleImageTokenId = GetInt(metadata, $"{arch}.ple.image_token_id", pleEosTokenId);
        int pleEmbPerLayer = GetInt(metadata, $"{arch}.embedding_length_per_layer");

        var pleMultipliers = GetUlongArray(metadata, $"{arch}.ple.layer_multipliers");
        var pleOffsets = GetUintArray(metadata, $"{arch}.ple.head_offsets");
        var pleVocabSizes = GetUintArray(metadata, $"{arch}.ple.head_vocab_sizes");

        var recrLayers = GetBoolArray(metadata, $"{arch}.attention.recurrent_layers");
        int fullAttnInterval = GetInt(metadata, $"{arch}.full_attention_interval", 4);
        if (fullAttnInterval <= 0) fullAttnInterval = 4;

        return new Qwen4ExpHyperparams
        {
            EmbedDim = GetInt(metadata, $"{arch}.embedding_length", GetInt(metadata, "general.embedding_length", 2560)),
            NumHeads = GetInt(metadata, $"{arch}.attention.head_count", 32),
            NumHeadsKv = GetInt(metadata, $"{arch}.attention.head_count_kv", 8),
            HeadDim = GetInt(metadata, $"{arch}.attention.key_length", 128),
            NumLayer = totalLayers > 0 ? totalLayers : GetInt(metadata, $"{arch}.block_count", 48),
            VocabSize = GetInt(metadata, $"{arch}.vocab_size", 152064),
            ContextLength = GetInt(metadata, $"{arch}.context_length", 4096),

            NumLayerNextn = GetInt(metadata, $"{arch}.nextn.layer_count"),
            ExpertFeedForwardLength = GetInt(metadata, $"{arch}.expert_feed_forward_length", 640),
            ExpertSharedFeedForwardLength = GetInt(metadata, $"{arch}.expert_shared_feed_forward_length"),
            ExpertCount = GetInt(metadata, $"{arch}.expert_count"),
            ExpertUsedCount = GetInt(metadata, $"{arch}.expert_used_count", 10),
            ExpertWeightsScale = GetFloat(metadata, $"{arch}.expert_weights_scale"),
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-6f),
            RopeDimensionSections = GetIntArray(metadata, $"{arch}.rope.dimension_sections"),
            SsmConvKernel = GetInt(metadata, $"{arch}.ssm.conv_kernel", 4),
            SsmInnerSize = GetInt(metadata, $"{arch}.ssm.inner_size"),
            SsmStateSize = GetInt(metadata, $"{arch}.ssm.state_size", 128),
            SsmDtRank = GetInt(metadata, $"{arch}.ssm.time_step_rank"),
            SsmGroupCount = GetInt(metadata, $"{arch}.ssm.group_count"),
            HyperConnectionCount = GetInt(metadata, $"{arch}.hyper_connection.count", 4),
            HyperConnectionLowRank = GetInt(metadata, $"{arch}.hyper_connection.low_rank", 320),
            IndexerHeadCount = GetInt(metadata, $"{arch}.attention.indexer_head_count"),
            IndexerKeyLength = GetInt(metadata, $"{arch}.attention.indexer_key_length"),
            IndexerTopK = GetInt(metadata, $"{arch}.attention.indexer_top_k"),
            IndexerKPool = kpool,
            CompressRatios = compressRatios,
            PleLayers = pleLayers,
            PleNgramSize = pleNgramSize,
            PleHeadsPerNgram = pleHeadsPerNgram,
            PleConvKernel = pleConvKernel,
            PleEosTokenId = pleEosTokenId,
            PleImageTokenId = pleImageTokenId,
            PleEmbeddingLengthPerLayer = pleEmbPerLayer,
            PleLayerMultipliers = pleMultipliers,
            PleHeadOffsets = pleOffsets,
            PleHeadVocabSizes = pleVocabSizes,
            FullAttentionInterval = fullAttnInterval,
            RecurrentLayers = recrLayers,
        };
    }

    private static int GetInt(IReadOnlyDictionary<string, object> m, string key, int fallback = 0)
    {
        if (!m.TryGetValue(key, out var v)) return fallback;
        if (v is System.Collections.IList list) return list.Count > 0 ? Convert.ToInt32(list[0], System.Globalization.CultureInfo.InvariantCulture) : fallback;
        return Convert.ToInt32(v, System.Globalization.CultureInfo.InvariantCulture);
    }

    private static float GetFloat(IReadOnlyDictionary<string, object> m, string key, float fallback = 0f) =>
        m.TryGetValue(key, out var v) ? Convert.ToSingle(v, System.Globalization.CultureInfo.InvariantCulture) : fallback;

    private static IReadOnlyList<int>? GetIntArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<int> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new int[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = Convert.ToInt32(list[i], System.Globalization.CultureInfo.InvariantCulture);
                return result;
            }
            default: return null;
        }
    }

    private static IReadOnlyList<bool>? GetBoolArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<bool> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new bool[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = Convert.ToBoolean(list[i], System.Globalization.CultureInfo.InvariantCulture);
                return result;
            }
            default: return null;
        }
    }

    private static IReadOnlyList<uint>? GetUintArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<uint> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new uint[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = Convert.ToUInt32(list[i], System.Globalization.CultureInfo.InvariantCulture);
                return result;
            }
            default: return null;
        }
    }

    private static IReadOnlyList<ulong>? GetUlongArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<ulong> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new ulong[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = Convert.ToUInt64(list[i], System.Globalization.CultureInfo.InvariantCulture);
                return result;
            }
            default: return null;
        }
    }
}

/// <summary>
/// Implements Qwen4Exp's GatedResidual Hyper-Connections:
/// <c>build_hc_mix</c> and <c>build_hc_combine</c> (qwen4exp.cpp:314-398).
/// </summary>
public static class Qwen4ExpGatedResidual
{
    /// <summary>
    /// Performs grouped RMSNorm, low-rank bottleneck gating, stream averaging, and injection weight projection
    /// for a single token across <paramref name="hc"/> parallel residual streams.
    /// Ports <c>llama_model_qwen4exp::graph::build_hc_mix</c> (qwen4exp.cpp:314-368).
    /// </summary>
    /// <param name="x">Input 4-stream residual buffer of length <c>hc * embedDim</c>.</param>
    /// <param name="wNorm">Grouped RMSNorm gamma weights of length <c>hc * embedDim</c> (shape [embedDim, hc]).</param>
    /// <param name="wDown">Down projection weights of shape [hcLowRank, hc * embedDim] (row-major).</param>
    /// <param name="wUp">Up projection weights of shape [hc * embedDim, hcLowRank] (row-major).</param>
    /// <param name="wInject">Optional injection weights of shape [hc, hc * embedDim] (row-major). Can be empty for head norm.</param>
    /// <param name="mixed">Output span of length <c>embedDim</c> receiving the collapsed block input.</param>
    /// <param name="inject">Output span of length <c>hc</c> receiving per-stream injection weights (if wInject is provided).</param>
    /// <param name="hc">Number of parallel streams (typically 4).</param>
    /// <param name="embedDim">Hidden dimension per stream (e.g. 2560).</param>
    /// <param name="hcLowRank">Low-rank bottleneck rank (typically 320).</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    public static void Mix(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> wNorm,
        ReadOnlySpan<float> wDown,
        ReadOnlySpan<float> wUp,
        ReadOnlySpan<float> wInject,
        Span<float> mixed,
        Span<float> inject,
        int hc,
        int embedDim,
        int hcLowRank,
        float eps)
    {
        int hcDim = hc * embedDim;
        Span<float> xn = stackalloc float[hcDim <= 1024 ? hcDim : 0];
        float[]? rentedXn = null;
        if (xn.IsEmpty)
        {
            rentedXn = System.Buffers.ArrayPool<float>.Shared.Rent(hcDim);
            xn = rentedXn.AsSpan(0, hcDim);
        }

        try
        {
            // 1. Grouped RMSNorm: each stream c in [0, hc-1] is independently normalized over embedDim,
            // then elementwise multiplied by wNorm[c * embedDim .. (c+1)*embedDim - 1].
            for (int c = 0; c < hc; c++)
            {
                var xc = x.Slice(c * embedDim, embedDim);
                var wc = wNorm.Slice(c * embedDim, embedDim);
                var xnc = xn.Slice(c * embedDim, embedDim);

                float sumSq = TensorPrimitives.SumOfSquares(xc);
                float invRms = 1.0f / MathF.Sqrt(sumSq / embedDim + eps);

                TensorPrimitives.Multiply(xc, invRms, xnc);
                TensorPrimitives.Multiply(xnc, wc, xnc);
            }

            // 2. Low-rank bottleneck down-projection: lo = SiLU(wDown * xn * (1.0f / hc))
            Span<float> lo = stackalloc float[hcLowRank <= 512 ? hcLowRank : 0];
            float[]? rentedLo = null;
            if (lo.IsEmpty)
            {
                rentedLo = System.Buffers.ArrayPool<float>.Shared.Rent(hcLowRank);
                lo = rentedLo.AsSpan(0, hcLowRank);
            }

            try
            {
                float invHc = 1.0f / hc;
                for (int r = 0; r < hcLowRank; r++)
                {
                    var row = wDown.Slice(r * hcDim, hcDim);
                    float dot = TensorPrimitives.Dot(row, xn) * invHc;
                    // SiLU(dot) = dot * sigmoid(dot)
                    float sig = 1.0f / (1.0f + MathF.Exp(-dot));
                    lo[r] = dot * sig;
                }

                // 3. Up-projection: gate = wUp * lo (shape [hcDim])
                // Gating: gated = xn * sigmoid(gate)
                // Collapse: mixed[i] = (1 / hc) * sum_c gated[c, i]
                mixed.Clear();

                Span<float> gateChunk = stackalloc float[Math.Min(embedDim, 1024)];
                for (int c = 0; c < hc; c++)
                {
                    int streamOffset = c * embedDim;
                    for (int chunkStart = 0; chunkStart < embedDim; chunkStart += gateChunk.Length)
                    {
                        int curChunk = Math.Min(gateChunk.Length, embedDim - chunkStart);
                        for (int i = 0; i < curChunk; i++)
                        {
                            int k = streamOffset + chunkStart + i;
                            var upRow = wUp.Slice(k * hcLowRank, hcLowRank);
                            float gVal = TensorPrimitives.Dot(upRow, lo);
                            float sig = 1.0f / (1.0f + MathF.Exp(-gVal));
                            float gated = xn[k] * sig;
                            mixed[chunkStart + i] += gated * invHc;
                        }
                    }
                }

                // 4. Injection weights projection: inject = wInject * xn (shape [hc])
                if (!wInject.IsEmpty && !inject.IsEmpty)
                {
                    for (int c = 0; c < hc; c++)
                    {
                        var injRow = wInject.Slice(c * hcDim, hcDim);
                        inject[c] = TensorPrimitives.Dot(injRow, xn);
                    }
                }
            }
            finally
            {
                if (rentedLo != null) System.Buffers.ArrayPool<float>.Shared.Return(rentedLo);
            }
        }
        finally
        {
            if (rentedXn != null) System.Buffers.ArrayPool<float>.Shared.Return(rentedXn);
        }
    }

    /// <summary>
    /// Combines the sub-block output into the multi-stream residual with 2*sigmoid learned per-stream injection.
    /// Ports <c>llama_model_qwen4exp::graph::build_hc_combine</c> (qwen4exp.cpp:370-398).
    /// </summary>
    /// <param name="residual">4-stream residual buffer of length <c>hc * embedDim</c>, updated in place.</param>
    /// <param name="blockOut">Output from token mixer (attention/GDN) or MoE of length <c>embedDim</c>.</param>
    /// <param name="inject">Per-stream injection weights of length <c>hc</c>.</param>
    /// <param name="hc">Number of streams (4).</param>
    /// <param name="embedDim">Hidden dimension per stream (2560).</param>
    public static void Combine(
        Span<float> residual,
        ReadOnlySpan<float> blockOut,
        ReadOnlySpan<float> inject,
        int hc,
        int embedDim)
    {
        float invHc = 1.0f / hc;
        for (int c = 0; c < hc; c++)
        {
            // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
            float w = 2.0f / (1.0f + MathF.Exp(-inject[c] * invHc));
            var streamResidual = residual.Slice(c * embedDim, embedDim);
            for (int i = 0; i < embedDim; i++)
            {
                streamResidual[i] += blockOut[i] * w;
            }
        }
    }
}

/// <summary>
/// Implements Qwen4Exp's Per-Layer Embedding (PLE) n-gram block:
/// signed-sqrt query-key gating, broadcast value norm, and dilated depthwise causal 1D conv.
/// Ports <c>llama_model_qwen4exp::graph::build_ple</c> (qwen4exp.cpp:1390-1478).
/// </summary>
public static class Qwen4ExpPle
{
    /// <summary>
    /// Computes the per-stream signed square root dot-product gate:
    /// <c>s = sum(key * query) / sqrt(embedDim); gate = sigmoid(sgn(s) * sqrt(|s|))</c>.
    /// (qwen4exp.cpp:1410-1416).
    /// </summary>
    public static void ComputePleGate(
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> query,
        Span<float> gate,
        int hc,
        int embedDim)
    {
        float invSqrtD = 1.0f / MathF.Sqrt(embedDim);
        for (int c = 0; c < hc; c++)
        {
            var kc = key.Slice(c * embedDim, embedDim);
            var qc = query.Slice(c * embedDim, embedDim);
            float dot = TensorPrimitives.Dot(kc, qc) * invSqrtD;
            float mag = MathF.Sqrt(Math.Max(MathF.Abs(dot), 1e-6f));
            float sgn = dot >= 0 ? 1.0f : -1.0f;
            float gVal = sgn * mag;
            gate[c] = 1.0f / (1.0f + MathF.Exp(-gVal));
        }
    }

    /// <summary>
    /// Applies grouped RMSNorm to a [hc * embedDim] vector with weights [embedDim, hc].
    /// </summary>
    public static void GroupedRmsNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        Span<float> output,
        int hc,
        int embedDim,
        float eps)
    {
        for (int c = 0; c < hc; c++)
        {
            var inSlice = input.Slice(c * embedDim, embedDim);
            var wSlice = weight.Slice(c * embedDim, embedDim);
            var outSlice = output.Slice(c * embedDim, embedDim);

            float sumSq = TensorPrimitives.SumOfSquares(inSlice);
            float invRms = 1.0f / MathF.Sqrt(sumSq / embedDim + eps);

            TensorPrimitives.Multiply(inSlice, invRms, outSlice);
            TensorPrimitives.Multiply(outSlice, wSlice, outSlice);
        }
    }

    /// <summary>
    /// Executes a single dilated causal step over the depthwise 1D conv buffer:
    /// <c>conv_out = SiLU(sum_k history(t - (kern-1-k)*dilation) * w_k)</c>.
    /// (qwen4exp.cpp:1430-1474).
    /// </summary>
    public static void StepDilatedConv(
        ReadOnlySpan<float> historyBuffer,
        ReadOnlySpan<float> convWeights,
        Span<float> convOut,
        int hcDim,
        int kernelSize,
        int dilation,
        int totalHistorySlots)
    {
        // totalHistorySlots >= (kernelSize - 1) * dilation + 1 (the current token is at index totalHistorySlots - 1)
        int hist = (kernelSize - 1) * dilation;
        convOut.Clear();

        for (int k = 0; k < kernelSize; k++)
        {
            int tapOffset = hist - (kernelSize - 1 - k) * dilation;
            var pastSlot = historyBuffer.Slice(tapOffset * hcDim, hcDim);
            var wk = convWeights.Slice(k * hcDim, hcDim);

            for (int i = 0; i < hcDim; i++)
            {
                convOut[i] += pastSlot[i] * wk[i];
            }
        }

        // SiLU activation: x * sigmoid(x)
        for (int i = 0; i < hcDim; i++)
        {
            float v = convOut[i];
            convOut[i] = v / (1.0f + MathF.Exp(-v));
        }
    }
}

/// <summary>
/// Implements QSA (Qwen Sparse Attention) indexer key pooling, scoring, top-k block selection,
/// and gated Q projection (qwen4exp.cpp:744-844, 960-1030).
/// </summary>
public static class Qwen4ExpQsa
{
    /// <summary>
    /// Pools <paramref name="kpool"/> consecutive raw indexer keys by their arithmetic mean,
    /// applies RMSNorm, and prepares the pooled key vector for indexer scoring.
    /// (qwen4exp.cpp:777-785).
    /// </summary>
    public static void PoolIndexerKeys(
        ReadOnlySpan<float> blockKeys,
        ReadOnlySpan<float> normWeight,
        Span<float> pooledKey,
        int kpool,
        int idxDim,
        float eps)
    {
        pooledKey.Clear();
        float invKpool = 1.0f / kpool;

        for (int i = 0; i < kpool; i++)
        {
            var key_i = blockKeys.Slice(i * idxDim, idxDim);
            for (int d = 0; d < idxDim; d++)
            {
                pooledKey[d] += key_i[d] * invKpool;
            }
        }

        // RMSNorm
        float sumSq = TensorPrimitives.SumOfSquares(pooledKey);
        float invRms = 1.0f / MathF.Sqrt(sumSq / idxDim + eps);
        TensorPrimitives.Multiply(pooledKey, invRms, pooledKey);
        TensorPrimitives.Multiply(pooledKey, normWeight, pooledKey);
    }

    /// <summary>
    /// Computes the unweighted multi-head indexer score for a single query token against one pooled block:
    /// <c>score = (1 / sqrt(idxDim)) * sum_h ReLU(Q_h . K_pool)</c>.
    /// (qwen4exp.cpp:815-827).
    /// </summary>
    public static float ComputeBlockScore(
        ReadOnlySpan<float> queryMultiHead,
        ReadOnlySpan<float> pooledKey,
        int numIndexerHeads,
        int idxDim)
    {
        float scoreSum = 0f;
        float invSqrtD = 1.0f / MathF.Sqrt(idxDim);

        for (int h = 0; h < numIndexerHeads; h++)
        {
            var qh = queryMultiHead.Slice(h * idxDim, idxDim);
            float dot = TensorPrimitives.Dot(qh, pooledKey);
            if (dot > 0f) scoreSum += dot;
        }

        return scoreSum * invSqrtD;
    }

    /// <summary>
    /// Splits the interleaved Q projection [q | gate] per head, applying RMSNorm to Q and exposing the gate.
    /// (qwen4exp.cpp:967-994).
    /// </summary>
    public static void SplitAndNormQGated(
        ReadOnlySpan<float> qFull,
        ReadOnlySpan<float> qNormWeight,
        Span<float> qOut,
        Span<float> gateOut,
        int numHeads,
        int headDim,
        float eps)
    {
        int doubleHeadDim = headDim * 2;
        for (int h = 0; h < numHeads; h++)
        {
            var src = qFull.Slice(h * doubleHeadDim, doubleHeadDim);
            var qHead = src.Slice(0, headDim);
            var gHead = src.Slice(headDim, headDim);

            var qDst = qOut.Slice(h * headDim, headDim);
            var gDst = gateOut.Slice(h * headDim, headDim);

            // RMSNorm on Q
            float sumSq = TensorPrimitives.SumOfSquares(qHead);
            float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + eps);
            TensorPrimitives.Multiply(qHead, invRms, qDst);
            TensorPrimitives.Multiply(qDst, qNormWeight, qDst);

            gHead.CopyTo(gDst);
        }
    }

    /// <summary>
    /// Applies the sigmoid gate to the multi-head attention output:
    /// <c>attnOut = attnOut * sigmoid(gate)</c>.
    /// (qwen4exp.cpp:1025-1030).
    /// </summary>
    public static void ApplyAttentionGate(Span<float> attnOut, ReadOnlySpan<float> gate)
    {
        for (int i = 0; i < attnOut.Length; i++)
        {
            float sig = 1.0f / (1.0f + MathF.Exp(-gate[i]));
            attnOut[i] *= sig;
        }
    }

    /// <summary>
    /// Selects top-k pool blocks according to multi-head ReLU scores, always retaining the active incomplete tail.
    /// Returns the total number of selected pool indices written to <paramref name="selectedPoolIndices"/>.
    /// </summary>
    public static int SelectTopKPools(
        ReadOnlySpan<float> poolScores,
        int topKPoolCount,
        Span<int> selectedPoolIndices)
    {
        int totalPools = poolScores.Length;
        if (totalPools == 0) return 0;
        if (totalPools <= topKPoolCount)
        {
            for (int i = 0; i < totalPools; i++) selectedPoolIndices[i] = i;
            return totalPools;
        }

        // Partial selection: pick topKPoolCount highest scores
        Span<int> indices = stackalloc int[totalPools];
        for (int i = 0; i < totalPools; i++) indices[i] = i;

        // Simple selection sort for topK
        for (int i = 0; i < topKPoolCount; i++)
        {
            int maxIdx = i;
            float maxScore = poolScores[indices[i]];
            for (int j = i + 1; j < totalPools; j++)
            {
                float s = poolScores[indices[j]];
                if (s > maxScore)
                {
                    maxScore = s;
                    maxIdx = j;
                }
            }
            if (maxIdx != i)
            {
                (indices[i], indices[maxIdx]) = (indices[maxIdx], indices[i]);
            }
            selectedPoolIndices[i] = indices[i];
        }

        return topKPoolCount;
    }
}

/// <summary>
/// 4-section IMRoPE (Interleaved M-RoPE) rotary embeddings for Qwen4Exp QSA and indexer.
/// </summary>
public static class Qwen4ExpRope
{
    /// <summary>
    /// Computes which M-RoPE section (0, 1, 2, or 3) corresponds to dimension pair <paramref name="pair"/>.
    /// Section 3 is unrotated (position 0 / identity).
    /// </summary>
    public static int GetMropeComponent(int pair, IReadOnlyList<int>? sections)
    {
        if (sections == null || sections.Count == 0) return 0;
        int s0 = sections[0];
        int s1 = sections.Count > 1 ? sections[1] : 0;
        int s2 = sections.Count > 2 ? sections[2] : 0;
        int s3 = sections.Count > 3 ? sections[3] : 0;
        int total = s0 + s1 + s2 + s3;
        if (total <= 0) return 0;

        int sector = pair % total;
        if (sector % 3 == 1 && sector < 3 * s1) return 1;
        if (sector % 3 == 2 && sector < 3 * s2) return 2;
        if (sector % 3 == 0 && sector < 3 * s0) return 0;
        return 3;
    }

    /// <summary>
    /// Applies 4-section IMRoPE rotation to <paramref name="vec"/> in place.
    /// </summary>
    public static unsafe void ApplyImRope(
        Span<float> vec,
        int pos,
        int numHeads,
        int headDim,
        IReadOnlyList<int>? sections,
        float ropeTheta = 1000000.0f)
    {
        if (headDim <= 0 || (headDim & 1) != 0) return;
        int halfDim = headDim / 2;

        Span<float> cosTab = stackalloc float[halfDim];
        Span<float> sinTab = stackalloc float[halfDim];

        for (int i = 0; i < halfDim; i++)
        {
            int comp = GetMropeComponent(i, sections);
            float p = comp switch
            {
                0 => pos,
                1 => pos,
                2 => pos,
                _ => 0f // unrotated 4th component
            };
            float angle = p * MathF.Pow(ropeTheta, -2f * i / headDim);
            cosTab[i] = MathF.Cos(angle);
            sinTab[i] = MathF.Sin(angle);
        }

        fixed (float* pVec = vec, pCos = cosTab, pSin = sinTab)
        {
            SimdKernels.ApplyRoPECachedNeox(pVec, pCos, pSin, numHeads, headDim);
        }
    }
}

/// <summary>
/// Maintains token sequence history and computes multi-head PLE n-gram table row indices.
/// </summary>
public sealed class Qwen4ExpPleHasher
{
    private readonly int _ngramSize;
    private readonly int _eosTokenId;
    private readonly ulong[] _multipliers;
    private readonly uint[] _headOffsets;
    private readonly uint[] _headVocabSizes;
    private readonly List<int> _tokenHistory = new();

    public Qwen4ExpPleHasher(Qwen4ExpHyperparams hp)
    {
        _ngramSize = hp.PleNgramSize > 0 ? hp.PleNgramSize : 2;
        _eosTokenId = hp.PleEosTokenId;
        _multipliers = hp.PleLayerMultipliers != null ? hp.PleLayerMultipliers.ToArray() : [1UL, 10007UL];
        _headOffsets = hp.PleHeadOffsets != null ? hp.PleHeadOffsets.ToArray() : [0U];
        _headVocabSizes = hp.PleHeadVocabSizes != null ? hp.PleHeadVocabSizes.ToArray() : [65536U];
    }

    public int NumHeads => _headOffsets.Length;

    public void PushToken(int token)
    {
        if (token == _eosTokenId)
        {
            _tokenHistory.Clear();
            return;
        }
        _tokenHistory.Add(token);
    }

    public void PushTokens(ReadOnlySpan<int> tokens)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            PushToken(tokens[i]);
        }
    }

    public void Reset() => _tokenHistory.Clear();

    /// <summary>
    /// Computes row indices in the PLE table for each head at the current token position.
    /// Uses XOR accumulation across n-gram multiplier products per llama.cpp (qwen4exp.cpp) and TensorSharp.
    /// </summary>
    public void ComputeRowIndices(Span<long> outRowIndices)
    {
        int heads = _headOffsets.Length;
        int histLen = _tokenHistory.Count;

        for (int h = 0; h < heads; h++)
        {
            ulong hash = 0;
            for (int k = 0; k < _ngramSize; k++)
            {
                int tIdx = histLen - 1 - k;
                ulong tokenVal = tIdx >= 0 ? (ulong)_tokenHistory[tIdx] : 0UL;
                ulong mult = k < _multipliers.Length ? _multipliers[k] : 1UL;
                ulong prod = unchecked(tokenVal * mult);
                hash = k == 0 ? prod : (hash ^ prod);
            }

            uint vocabSize = h < _headVocabSizes.Length && _headVocabSizes[h] > 0 ? _headVocabSizes[h] : 65536U;
            uint offset = h < _headOffsets.Length ? _headOffsets[h] : 0U;
            long row = offset + (long)(hash % vocabSize);
            outRowIndices[h] = row;
        }
    }
}

