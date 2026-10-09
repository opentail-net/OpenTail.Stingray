namespace OpenTail.Stingray.Cli.Scout;

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
///             an upper bound. Only computed for the plain-attention families; MLA, hybrid and recurrent families have different state and are Unknown.
///   scratch   batched-prefill buffers, scaled to the tokens in flight: 1.5 x ctx x (3 x ffn + 4 x hidden) x 4 bytes (measured: Mistral-7B, 608 tokens, 179 MiB
///             against 144 MiB for the formula without the 1.5).
/// </summary>
public static class HostMemoryEstimator
{
    public const long BaseOverheadBytes = 192L << 20;
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
        if (family is not "Dense")
        {
            string why = family is null
                ? "the architecture is not registered, so its state layout is unknown"
                : $"KV/state layout of the '{family}' family is not modelled yet";
            parts.Add(new("kv_cache", null, Certainty.Unknown, why));
            parts.Add(new("prefill_scratch", null, Certainty.Unknown, why));
            return Unknown(why, parts);
        }

        int ctx = hp.ContextLength > 0 ? Math.Min(contextTokens, hp.ContextLength) : contextTokens;
        long kv = KvBytes(hp, ctx);
        parts.Add(new("kv_cache", kv, Certainty.Estimated, $"fp32 K+V for {ctx} tokens over the non-aliased layers; an upper bound (only the used part is touched)"));

        long ffn = hp.IsMoE
            ? Math.Max(hp.IntermediateDim, (long)Math.Max(1, hp.NumActiveExperts) * hp.ExpertIntermediateDim + hp.SharedExpertIntermediateDim)
            : hp.IntermediateDim;
        long scratch = (long)(1.5 * ctx * (3 * ffn + 4L * hp.EmbeddingDim) * sizeof(float));
        parts.Add(new("prefill_scratch", scratch, Certainty.Estimated, $"1.5 x {ctx} tokens x (3 x ffn {ffn} + 4 x hidden {hp.EmbeddingDim}) x 4 bytes"));

        long total = parts.Sum(p => p.Bytes ?? 0);
        return new HostWorkingSet(Certainty.Estimated, total,
            "upper bound for a CPU run at the stated context; calibrated on measured dense runs, not a prediction (see 061-coverage-tooling.md)", parts);
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
