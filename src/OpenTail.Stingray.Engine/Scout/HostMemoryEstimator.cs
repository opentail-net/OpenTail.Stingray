namespace OpenTail.Stingray.Engine.Scout;

/// <summary>One term of the host-RAM estimate. <see cref="Bytes"/> is null when the term could not be established (never 0 as a stand-in).</summary>
public sealed record WorkingSetComponent(string Name, long? Bytes, Certainty Certainty, string Basis);

public sealed record HostWorkingSet(Certainty Certainty, long? Bytes, string Source, IReadOnlyList<WorkingSetComponent> Components);

/// <summary>
/// Upper-bound estimate of the peak working set of a CPU run of a text-generation GGUF (plan §5). It is a safety gate for choosing what to run
/// on a 64 GB host, not a prediction: every term is chosen to over- rather than under-count, and any term that cannot be established makes the
/// whole estimate Unknown so the gate stays closed. It deliberately says nothing about GPU placement (CLAUDE.md rule 13).
///
/// Terms and where they come from (measured 2026-10-09, CPU, STINGRAY_GC_STATS peakWorkingSet; docs/reference/061-coverage-tooling.md):
///   weights   every tensor byte. The CPU path pre-faults the whole mapped file (ForwardPass.PrefaultWeights); a MoE's routed experts are touched
///             over time, so the full size is the right upper bound even where the pre-fault pass skips them.
///   repack    Q4_K dense weights are repacked into an ADDITIONAL anonymous copy, 1216/1152 of those bytes, capped at a quarter of available memory
///             (ForwardPass.ResolveQ4Kx8CacheBudget). Available memory is taken to be the budget. This is what makes a Q4_K_M file cost ~1.8x its size.
///   base      runtime, native libraries, tokenizer, rope tables: 130-140 MiB measured on 135M and 360M models; 192 MiB is used.
///   kv        fp32 K and V for every non-aliased layer at the requested context. Real runs touch only the used part of the allocation, so this is
///             an upper bound. Only computed for the plain-attention families; MLA (DeepSeek2) is modelled below; other recurrent families have different state and are Unknown.
///   scratch   batched-prefill buffers, scaled to the tokens in flight: 1.5 x ctx x (3 x ffn + 4 x hidden) x 4 bytes (measured: Mistral-7B, 608 tokens, 179 MiB
///             against 144 MiB for the formula without the 1.5).
///
/// The hybrid recurrent family (HybridGdn: qwen35, qwen35moe, qwen4exp; Gated-DeltaNet layers interleaved with full attention) is modelled separately, from
/// calibration runs on Qwen3.5 0.8B / 4B / 9B (dense) and Qwen3.6-35B-A3B (MoE), 2026-10-09:
///   repack    NONE. Its forward pass does not use the Q4_K repack cache: peak is the file plus a roughly constant 350-420 MiB (0.8B: file 508 -> 857 MiB; 9B: 5417 -> 5810),
///             where the dense formula would have over-counted by nearly 2x.
///   base      448 MiB (measured fixed overhead beyond file and state: 330-356 MiB).
///   state     exact and eager: per GDN layer (ConvKernel-1) x ConvChannels + NumVHeads x HeadDim x HeadDim floats; does not grow with context (GdnStateCache).
///   kv        attention layers ONLY (PagedKvCache allocates pages per layer on first write, and GDN layers never write): 2 x kvHeads x headDim x ctx x 4 bytes each, plus one per MTP head layer.
///   scratch   1.10 x ctx x (3 x hidden + 5 x valueDim + 3 x ffn + topk x (2 x expertFfn + hidden)) x 4 bytes; the last term is the batched-prefill expert buffers (MoE only).
///             Measured at ~1700 prompt tokens, estimate vs measured: 0.8B 1.10, 4B 1.06, 9B 1.04, 35B-A3B 1.005 (see 061-coverage-tooling.md).
/// </summary>
public static class HostMemoryEstimator
{
    public const long BaseOverheadBytes = 192L << 20;
    /// <summary>Fixed overhead of the hybrid recurrent path, measured 330-356 MiB beyond file and state.</summary>
    public const long RwkvBaseOverheadBytes = 128L << 20;
    private const long RwkvScratchBytes = 32L << 20;
    public const long HybridBaseOverheadBytes = 448L << 20;
    private const double HybridScratchMargin = 1.10;
    private const double RepackFactor = 1216.0 / 1152.0;

    public static HostWorkingSet Estimate(IReadOnlyList<GgufTensorInfo> tensors, ModelHyperparams? hp, string? family, int contextTokens, long? budgetBytes)
    {
        var parts = new List<WorkingSetComponent>();

        long weights = 0, q4k = 0;
        foreach (var t in tensors)
        {
            long size;
            try { size = t.ByteSize; }
            catch (OverflowException) { return Unknown("a tensor size overflows; the index is malformed", parts); }
            weights += size;
            // Routed-expert stacks (blk.N.ffn_*_exps) are not repacked: Phi-3.5-MoE (7.1 GiB of Q4_K, almost all expert stacks) measured 19,622 MiB against 19,104 MiB of weights,
            // i.e. no repack copy. Dense projections and shared experts are, so they stay in the term. Other MoE families are not measured.
            if (t.DType == DType.Q4_K && !t.Name.Contains("_exps.", StringComparison.Ordinal)) q4k += size;
        }
        parts.Add(new("weights", weights, Certainty.Known, "sum of all tensor bytes in the index; the CPU path touches every page"));

        long repack = (long)(q4k * RepackFactor);
        string repackBasis = $"{q4k} Q4_K bytes (excluding routed-expert stacks) x 1216/1152 (additional anonymous copy)";
        if (budgetBytes is long b && repack > b / 4)
        {
            repack = b / 4;
            repackBasis += $", capped at a quarter of the budget ({repack} bytes)";
        }
        parts.Add(new("q4k_repack", repack, Certainty.Estimated, repackBasis));
        parts.Add(new("base", BaseOverheadBytes, Certainty.Estimated, "runtime and native libraries; measured 130-140 MiB, 192 MiB used"));

        if (hp is null)
        {
            parts.Add(new("kv_cache", null, Certainty.Unknown, "hyperparameters could not be resolved from the metadata"));
            parts.Add(new("prefill_scratch", null, Certainty.Unknown, "hyperparameters could not be resolved from the metadata"));
            return Unknown("hyperparameters could not be resolved", parts);
        }
        // RWKV declares head_count 0 in its metadata, so it must be handled before the attention-shaped completeness check below.
        if (family is "Rwkv")
            return EstimateRwkv(parts, hp, weights);
        // Zero-valued essentials mean the metadata was incomplete and the resolver filled defaults: an estimate built on them would silently under-count.
        var missing = new List<string>();
        if (hp.NumLayers <= 0) missing.Add("block_count");
        if (hp.EmbeddingDim <= 0) missing.Add("embedding_length");
        if (hp.NumHeads <= 0) missing.Add("attention.head_count");
        if (hp.NumKvHeads <= 0) missing.Add("attention.head_count_kv");
        if (hp.HeadDim <= 0) missing.Add("head dimension");
        if (hp.IntermediateDim <= 0 && !(hp.IsMoE && hp.ExpertIntermediateDim > 0)) missing.Add("feed_forward_length");
        if (missing.Count > 0)
        {
            string why = "hyperparameters incomplete in the metadata (" + string.Join(", ", missing) + ")";
            parts.Add(new("kv_cache", null, Certainty.Unknown, why));
            parts.Add(new("prefill_scratch", null, Certainty.Unknown, why));
            return Unknown(why, parts);
        }
        if (family is "HybridGdn")
            return EstimateHybridGdn(parts, hp, contextTokens, weights, budgetBytes);
        bool mla = family is "DeepSeek2Mla" && hp.KvLoraRank > 0 && hp.MlaVHeadDim > 0;
        if (family is not "Dense" && !mla)
        {
            string why = family is null
                ? "the architecture is not registered, so its state layout is unknown"
                : $"KV/state layout of the '{family}' family is not modelled yet";
            parts.Add(new("kv_cache", null, Certainty.Unknown, why));
            parts.Add(new("prefill_scratch", null, Certainty.Unknown, why));
            return Unknown(why, parts);
        }

        int ctx = hp.ContextLength > 0 ? Math.Min(contextTokens, hp.ContextLength) : contextTokens;
        // MLA (DeepSeek2): the engine caches the EXPANDED per-head K (key_length) and V (value_length) in fp32, not the compressed latent (measured: slope 0.88-0.96 MiB per token on
        // DeepSeek-V2-Lite, of which 0.53 is this term).
        long kv = mla ? (long)hp.NumLayers * hp.NumHeads * (hp.HeadDim + hp.MlaVHeadDim) * ctx * sizeof(float) : KvBytes(hp, ctx);
        if (mla) parts.Add(new("mla_base", 96L << 20, Certainty.Estimated, "extra fixed overhead of the MLA path: 267 MiB measured at a 15-token prompt on V2-Lite against the 192 MiB base"));
        // Absorbed-MLA GGUFs (split attn_k_b / attn_v_b, e.g. Kimi-VL / Moonlight) cost a further fixed ~275 MiB: measured 543 MiB over the file at a 24-token prompt, against 267 on V2-Lite.
        // The expansion of the kv_b projection to fp32 explains it (27 x 512 x 16 x 256 x 4 = 226 MiB); (key + value) per head instead of 256 adds the safety margin.
        if (mla && tensors.Any(t => t.Name.EndsWith(".attn_k_b.weight", StringComparison.Ordinal)))
            parts.Add(new("mla_absorbed_expansion", (long)hp.NumLayers * hp.KvLoraRank * hp.NumHeads * (hp.HeadDim + hp.MlaVHeadDim) * sizeof(float), Certainty.Estimated, "fp32 expansion of the kv_b projection for split attn_k_b / attn_v_b layouts; measured +276 MiB on Kimi-VL"));
        parts.Add(new("kv_cache", kv, Certainty.Estimated, mla
            ? $"fp32 expanded K ({hp.HeadDim}) + V ({hp.MlaVHeadDim}) per head for {ctx} tokens over {hp.NumLayers} layers; an upper bound"
            : $"fp32 K+V for {ctx} tokens over the non-aliased layers; an upper bound (only the used part is touched)"));

        long ffn = hp.IsMoE
            ? Math.Max(hp.IntermediateDim, (long)Math.Max(1, hp.NumActiveExperts) * hp.ExpertIntermediateDim + hp.SharedExpertIntermediateDim)
            : hp.IntermediateDim;
        // MoE batched prefill also holds per-token expert buffers: topk x (gate + up of the expert FFN, plus a hidden-sized down output). Measured directly on the hybrid MoE
        // (they explain its extra ~160 MiB at ~1700 tokens); the dense-family MoE (Phi-3.5-MoE) fit its measurement without the term (1.016x) but the buffers exist there too.
        long moeFloats = hp.IsMoE ? (long)Math.Max(1, hp.NumActiveExperts) * (2L * hp.ExpertIntermediateDim + hp.EmbeddingDim) : 0;
        // MLA runs carry a further 25% (the MLA projections hold extra per-token buffers); with a 288 MiB base this keeps the estimate 0.5-1.5% above the measured peak on V2-Lite.
        double scratchFactor = mla ? 1.25 : 1.0;
        long scratch = (long)(scratchFactor * 1.5 * ctx * (3 * ffn + 4L * hp.EmbeddingDim) * sizeof(float)) + ctx * moeFloats * sizeof(float);
        parts.Add(new("prefill_scratch", scratch, Certainty.Estimated,
            $"1.5 x {ctx} tokens x (3 x ffn {ffn} + 4 x hidden {hp.EmbeddingDim}) x 4 bytes" + (hp.IsMoE ? $" + {ctx} x expert buffers {moeFloats} x 4 bytes" : "")));

        long total = parts.Sum(p => p.Bytes ?? 0);
        return new HostWorkingSet(Certainty.Estimated, total,
            "upper bound for a CPU run at the stated context; calibrated on measured dense runs, not a prediction (see 061-coverage-tooling.md)", parts);
    }

    /// <summary>
    /// RWKV (recurrent, no KV cache). Measured 2026-10-10 on RWKV7 G1h 1.5B and 2.9B Q4_K_M (CPU, STINGRAY_GC_STATS peakWorkingSet; prompts of 17 / 734 / 1454 tokens):
    /// 1.5B (file 1039 MiB) 1092 / 1152 / 1143 MiB, 2.9B (file 1959 MiB) 2005 / 2060 / 2041 MiB. Peak is the file plus 46-113 MiB, flat in prompt length (tokens are processed
    /// sequentially), and there is no Q4_K repack copy. Terms: weights, base 128 MiB, the wkv state, and a fixed 32 MiB for prefill buffers.
    /// </summary>
    private static HostWorkingSet EstimateRwkv(List<WorkingSetComponent> parts, ModelHyperparams hp, long weights)
    {
        parts.RemoveAll(p => p.Name is "q4k_repack" or "base");
        parts.Add(new("q4k_repack", 0, Certainty.Estimated, "none: the RWKV path does not build the Q4_K repack cache (measured: peak is file size plus 46-113 MiB)"));
        parts.Add(new("base", RwkvBaseOverheadBytes, Certainty.Estimated, "runtime, native libraries and fixed buffers; 128 MiB covers the measured 46-113 MiB total overhead"));
        if (hp.NumLayers <= 0 || hp.EmbeddingDim <= 0)
        {
            string why = "block_count or embedding_length missing from the metadata";
            parts.Add(new("kv_cache", null, Certainty.Unknown, why));
            parts.Add(new("prefill_scratch", null, Certainty.Unknown, why));
            return Unknown(why, parts);
        }
        int headSize = hp.HeadDim > 0 ? hp.HeadDim : 64;
        long state = (long)hp.NumLayers * ((long)hp.EmbeddingDim * headSize + 2L * hp.EmbeddingDim) * sizeof(float);
        parts.Add(new("rwkv_state", state, Certainty.Estimated, $"{hp.NumLayers} layers x ({hp.EmbeddingDim} x head size {headSize} wkv matrix + 2 x {hp.EmbeddingDim} token-shift) floats; does not grow with context"));
        parts.Add(new("kv_cache", 0, Certainty.Known, "none: RWKV keeps a fixed-size recurrent state, not a KV cache"));
        parts.Add(new("prefill_scratch", RwkvScratchBytes, Certainty.Estimated, "fixed prefill buffers; measured +55 MiB between a 17-token and 734/1454-token prompt, flat in length; 32 MiB is the residual after the base term"));
        long total = parts.Sum(p => p.Bytes ?? 0);
        return new HostWorkingSet(Certainty.Estimated, total,
            "upper bound for a CPU run; RWKV7 calibrated on two measured models, not a prediction (see 061-coverage-tooling.md)", parts);
    }

    /// <summary>The hybrid recurrent family. <paramref name="parts"/> already holds the weights term; the dense repack and base terms are replaced (see the class remarks).</summary>
    private static HostWorkingSet EstimateHybridGdn(List<WorkingSetComponent> parts, ModelHyperparams hp, int contextTokens, long weights, long? budgetBytes)
    {
        // The dense branch added its repack and base terms before dispatching here; this family has neither, so drop them and say so.
        parts.RemoveAll(p => p.Name is "q4k_repack" or "base");
        parts.Add(new("q4k_repack", 0, Certainty.Estimated, "none: the hybrid path does not build the Q4_K repack cache (measured: peak is file size plus a roughly constant overhead)"));
        parts.Add(new("base", HybridBaseOverheadBytes, Certainty.Estimated, "runtime, native libraries and fixed buffers; measured 330-356 MiB beyond file and state on four models, 448 MiB used"));

        var gdn = hp.Gdn;
        var types = hp.LayerTypes;
        if (gdn is null || types is null || types.Count == 0)
            return UnknownHybrid(parts, "the Gated-DeltaNet configuration or per-layer types could not be resolved");
        int gdnLayers = types.Count(t => t == LayerType.GatedDeltaNet);
        int attnLayers = types.Count(t => t == LayerType.Attention);
        if (gdnLayers + attnLayers != types.Count)
            return UnknownHybrid(parts, "the model has a layer type this estimator does not model");
        if (gdn.NumVHeads <= 0 || gdn.HeadDim <= 0 || gdn.ConvChannels <= 0)
            return UnknownHybrid(parts, "the Gated-DeltaNet dimensions are incomplete in the metadata");

        int ctx = hp.ContextLength > 0 ? Math.Min(contextTokens, hp.ContextLength) : contextTokens;

        long stateFloats = (long)Math.Max(0, gdn.ConvKernel - 1) * gdn.ConvChannels + (long)gdn.NumVHeads * gdn.HeadDim * gdn.HeadDim;
        long state = gdnLayers * stateFloats * sizeof(float);
        parts.Add(new("gdn_state", state, Certainty.Estimated,
            $"{gdnLayers} Gated-DeltaNet layers x ({gdn.ConvKernel - 1} x {gdn.ConvChannels} conv + {gdn.NumVHeads} x {gdn.HeadDim} x {gdn.HeadDim} state) floats; eager, does not grow with context"));

        int kvLayers = attnLayers + Math.Max(0, hp.NumMtpLayers);
        long kv = (long)kvLayers * 2 * hp.NumKvHeads * hp.HeadDim * ctx * sizeof(float);
        parts.Add(new("kv_cache", kv, Certainty.Estimated,
            $"fp32 K+V for {ctx} tokens in the {kvLayers} attention layers only ({gdnLayers} Gated-DeltaNet layers store no KV); an upper bound"));

        long ffn = hp.IsMoE
            ? Math.Max(hp.IntermediateDim, (long)Math.Max(1, hp.NumActiveExperts) * hp.ExpertIntermediateDim + hp.SharedExpertIntermediateDim)
            : hp.IntermediateDim;
        long moeFloats = hp.IsMoE ? (long)Math.Max(1, hp.NumActiveExperts) * (2L * hp.ExpertIntermediateDim + hp.EmbeddingDim) : 0;
        long perToken = 3L * hp.EmbeddingDim + 5L * gdn.ValueDim + 3L * ffn + moeFloats;
        long scratch = (long)(HybridScratchMargin * ctx * perToken * sizeof(float));
        parts.Add(new("prefill_scratch", scratch, Certainty.Estimated,
            $"1.10 x {ctx} tokens x (3 x hidden {hp.EmbeddingDim} + 5 x value {gdn.ValueDim} + 3 x ffn {ffn}" + (hp.IsMoE ? $" + expert buffers {moeFloats}" : "") + ") x 4 bytes"));

        long total = parts.Sum(p => p.Bytes ?? 0);
        return new HostWorkingSet(Certainty.Estimated, total,
            "upper bound for a CPU run at the stated context; hybrid recurrent family calibrated on four measured models, not a prediction (see 061-coverage-tooling.md)", parts);
    }

    private static HostWorkingSet UnknownHybrid(List<WorkingSetComponent> parts, string why)
    {
        parts.Add(new("kv_cache", null, Certainty.Unknown, why));
        parts.Add(new("prefill_scratch", null, Certainty.Unknown, why));
        return Unknown(why, parts);
    }
    /// <summary>fp32 K+V bytes for <paramref name="ctx"/> tokens: per-layer head dims / KV heads, sliding-window layers capped at their window, aliased layers free.</summary>
    internal static long KvBytes(ModelHyperparams hp, int ctx)
    {
        long total = 0;
        for (int i = 0; i < hp.NumLayers; i++)
        {
            if (hp.KvSourceLayer is { } src && i < src.Count && src[i] >= 0) continue;
            int heads = hp.LayerKvHeads is { } lkv && i < lkv.Count ? lkv[i] : hp.NumKvHeads;
            int dim = hp.LayerHeadDim is { } lhd && i < lhd.Count ? lhd[i] : hp.HeadDim;
            long layerCtx = hp.IsSwaLayer is { } swa && i < swa.Count && swa[i] && hp.SlidingWindowSize > 0
                ? Math.Min(ctx, hp.SlidingWindowSize) : ctx;
            total += 2L * heads * dim * layerCtx * sizeof(float);
        }
        return total;
    }

    private static HostWorkingSet Unknown(string why, List<WorkingSetComponent> parts) =>
        new(Certainty.Unknown, null, why, parts);
}
