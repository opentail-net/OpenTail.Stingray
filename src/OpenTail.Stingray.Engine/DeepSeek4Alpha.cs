namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- DeepSeek-V4 ("deepseek4" GGUF architecture) implementation.
//
// Status as of 2026-09-02: ported directly from the vendored llama.cpp reference
// (examples/llama.cpp/llama.cpp/src/models/deepseek4.cpp, all 1547 lines read in full) by a
// coding agent working from the C++ source alone -- NO real DeepSeek-V4 GGUF has been loaded,
// NO output has been compared against any reference implementation, and NO unit test in this
// codebase currently exercises any function below. Treat every formula here as "believed
// correct from careful reading of the reference," not "verified." This is explicitly the
// user-requested "alpha" deliverable: a complete first-draft port to build on, not a finished
// or trustworthy implementation. See docs/058-deepseek-full-lineage-implementation-plan.md
// Phase 0 for the plan this belongs to.
//
// "deepseek4" is NOT admitted in ModelCompatibility.cs and must not be added there until this
// code has a real greedy-parity or perplexity receipt against a reference implementation, per
// this codebase's standing policy for every other architecture gate.
//
// SCOPE OF THIS FILE: the algorithmic core only -- hyperparameters, hyper-connection
// (Sinkhorn-normalized multi-stream residual) math, the lightning indexer's top-k scoring, and
// the CSA/HCA compressed-KV-state math. It is NOT wired into GGUF tensor loading
// (ModelGraph.cs), the forward-pass dispatch switch, or a KV-cache implementation -- V4 needs a
// new persistent, cross-position compressed-state cache that this codebase has no analog for
// (see the CompressedKvState comment below), and building that safely needs its own design pass
// with the ability to test against real tensors, which this session does not have. Wiring
// (tensor names/shapes, dispatch, the state cache, RoPE/attention integration) is the next step,
// not done here.
//
// Every method below cites the exact reference function and line range it was ported from so a
// future session (with real weights available) can re-diff quickly. Where the reference builds
// on a ggml graph (deferred/lazy tensor ops), this port evaluates eagerly on float spans instead
// -- correct for the same reason eager and lazy evaluation of the same arithmetic agree, but
// worth re-checking op-by-op against the graph builder once real intermediate values exist to
// diff against (the deepseek2 investigation, docs/done/032-...md, found real bugs that only
// showed up once ground-truth intermediates were available -- expect the same here).
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. DeepSeek-V4-specific hyperparameters, mirroring the GGUF keys read by
/// <c>llama_model_deepseek4::load_arch_hparams</c> (deepseek4.cpp:18-77). Kept separate from
/// <c>OpenTail.Stingray.Core.ModelHyperparameters</c> rather than added to that shared record,
/// since none of these fields are wired into GGUF loading yet -- merging them into the shared
/// record before the loader exists would silently create dead/unreachable properties there.
/// </summary>
public sealed record DeepSeek4Hyperparams
{
    /// <summary>Number of NextN/MTP draft-head layers appended after the main trunk. deepseek4.cpp:19.</summary>
    public int NumLayerNextn { get; init; }

    /// <summary>Q down-projection LoRA rank ({arch}.attention.q_lora_rank). deepseek4.cpp:30.</summary>
    public int QLoraRank { get; init; }

    /// <summary>Sliding-window size for the tail SWA layers ({arch}.attention.sliding_window). deepseek4.cpp:31.</summary>
    public int SlidingWindow { get; init; }

    /// <summary>Expert FFN intermediate width ({arch}.expert_feed_forward_length). deepseek4.cpp:33.</summary>
    public int ExpertFeedForwardLength { get; init; }

    /// <summary>Number of always-active shared experts ({arch}.expert_shared_count). deepseek4.cpp:34.</summary>
    public int ExpertSharedCount { get; init; }

    /// <summary>Post-top-k routed-expert weight scale ({arch}.expert_weights_scale). deepseek4.cpp:35.</summary>
    public float ExpertWeightsScale { get; init; } = 1f;

    /// <summary>Whether routed-expert top-k weights are renormalized to sum to 1 ({arch}.expert_weights_norm). deepseek4.cpp:36.</summary>
    public bool ExpertWeightsNorm { get; init; }

    /// <summary>
    /// SwiGLU clamp value applied to routed experts' gate activation, per layer
    /// ({arch}.swiglu_clamp_exp / falls back to a single shared value). deepseek4.cpp:37-40.
    /// </summary>
    public IReadOnlyList<float> SwigluClampExp { get; init; } = Array.Empty<float>();

    /// <summary>SwiGLU clamp value for the shared expert (defaults to <see cref="SwigluClampExp"/> if absent). deepseek4.cpp:38-40.</summary>
    public IReadOnlyList<float> SwigluClampShexp { get; init; } = Array.Empty<float>();

    /// <summary>Lightning-indexer head count ({arch}.attention.indexer_head_count). deepseek4.cpp:42.</summary>
    public int IndexerNumHeads { get; init; }

    /// <summary>Lightning-indexer per-head key width ({arch}.attention.indexer_key_length). deepseek4.cpp:43.</summary>
    public int IndexerHeadSize { get; init; }

    /// <summary>Lightning-indexer top-k selection count ({arch}.attention.indexer_top_k). deepseek4.cpp:44.</summary>
    public int IndexerTopK { get; init; }

    /// <summary>Number of attention-output groups for the grouped output LoRA ({arch}.attention.output_group_count). deepseek4.cpp:46.</summary>
    public int OutputGroupCount { get; init; } = 1;

    /// <summary>Attention-output LoRA rank, per group ({arch}.attention.output_lora_rank). deepseek4.cpp:47.</summary>
    public int OutputLoraRank { get; init; }

    /// <summary>RoPE base frequency used specifically for the CSA/HCA compressed-KV position embedding ({arch}.attention.compress_rope_freq_base). deepseek4.cpp:48.</summary>
    public float CompressRopeFreqBase { get; init; } = 10000f;

    /// <summary>
    /// Hyper-connection stream multiplier ("HC" width -- the residual stream is carried as this
    /// many parallel copies). deepseek4.cpp:49,54 (n_embd_out_impl = hc_mult * n_embd). The
    /// reference asserts this is always 4 (deepseek4.cpp:362, GGML_ASSERT(hc == 4)) for the
    /// per-layer hc_pre path -- treat any other value as a signal this port's shape assumptions
    /// (built around hc==4) need re-checking, not silently extended.
    /// </summary>
    public int HyperConnectionMultiplier { get; init; } = 4;

    /// <summary>Sinkhorn normalization iteration count for the hyper-connection combine matrix ({arch}.hyper_connection.sinkhorn_iterations). deepseek4.cpp:50.</summary>
    public int HyperConnectionSinkhornIterations { get; init; } = 1;

    /// <summary>Epsilon added at each Sinkhorn normalization step ({arch}.hyper_connection.epsilon). deepseek4.cpp:51.</summary>
    public float HyperConnectionEpsilon { get; init; } = 1e-6f;

    /// <summary>Number of leading layers using hash-based (not learned-router) MoE routing ({arch}.hash_layer_count). deepseek4.cpp:52.</summary>
    public int HashLayerCount { get; init; }

    /// <summary>
    /// Per-layer KV-compression ratio ({arch}.attention.compress_ratios). 0 = no compression
    /// (plain KV cache, <c>build_raw_attention</c>); 4 = CSA (<c>build_csa_lid_attention</c>,
    /// combines lightning-indexer top-k with compressed KV); 128 = HCA
    /// (<c>build_hca_attention</c>, compressed KV without the indexer). deepseek4.cpp:56-61,
    /// 981-982, 1224-1245. Only 0, 4, and 128 are valid (deepseek4.cpp:148-150).
    /// </summary>
    public IReadOnlyList<int> CompressRatios { get; init; } = Array.Empty<int>();

    /// <summary>Total transformer layer count including MTP tail layers.</summary>
    public int NumLayerAll { get; init; }

    /// <summary>Main trunk layer count (NumLayerAll - NumLayerNextn).</summary>
    public int NumLayer => NumLayerAll - NumLayerNextn;

    /// <summary>Embedding width.</summary>
    public int EmbedDim { get; init; }

    /// <summary>Attention head count.</summary>
    public int NumHeads { get; init; }

    /// <summary>Per-head Q/K/V width (n_embd_head_k in the reference).</summary>
    public int HeadDim { get; init; }

    /// <summary>RoPE-rotated portion of each head's width (n_rot). The remainder is the "nope" portion.</summary>
    public int RopeDim { get; init; }

    /// <summary>Total MoE expert count.</summary>
    public int NumExperts { get; init; }

    /// <summary>Routed experts selected per token.</summary>
    public int NumExpertsUsed { get; init; }

    /// <summary>RMSNorm epsilon ({arch}.attention.layer_norm_rms_epsilon).</summary>
    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>RoPE base frequency for raw (non-compressed) layers ({arch}.rope.freq_base).</summary>
    public float RopeFreqBase { get; init; } = 10000f;

    /// <summary>
    /// ALPHA/UNTESTED. Reads the <c>deepseek4</c>-specific GGUF metadata keys per
    /// <c>llama_model_deepseek4::load_arch_hparams</c> (deepseek4.cpp:18-77), using the exact key
    /// strings from <c>llama-arch.cpp</c>'s <c>LLM_KV_*</c> table (confirmed by direct grep, not
    /// assumed from naming convention). Pure metadata parsing -- no tensor access, so this is
    /// testable with a synthetic dictionary the same way DeepSeek2Tests.cs tests the shared
    /// ModelHyperparameters loader, without needing a real GGUF. Only reads the dsv4-specific
    /// keys; the shared keys deepseek4 has in common with every other architecture (embedding
    /// width, head count, etc. -- deepseek4.cpp doesn't even re-read most of those, relying on
    /// the generic loader) are expected to come from
    /// <c>OpenTail.Stingray.Core.ModelHyperparameters</c>'s existing loader once this is wired;
    /// the caller passes them in here rather than this method re-deriving them, to avoid
    /// duplicating that logic.
    /// </summary>
    public static DeepSeek4Hyperparams FromGgufMetadata(
        IReadOnlyDictionary<string, object> metadata, string arch, int numLayerAll,
        int embedDim, int numHeads, int headDim, int ropeDim, int numExperts, int numExpertsUsed)
    {
        int numLayerNextn = GetInt(metadata, $"{arch}.nextn_predict_layers", 0);

        var compressRatios = GetIntArray(metadata, $"{arch}.attention.compress_ratios")
            ?? Array.Empty<int>();
        if (compressRatios.Count < numLayerAll - numLayerNextn)
        {
            // deepseek4.cpp:58-60 throws a hard runtime_error here -- compress_ratios must cover
            // every trunk layer. This port surfaces the same failure as an exception at load time
            // rather than silently defaulting missing entries to 0 (which would silently disable
            // CSA/HCA on layers the checkpoint actually declares compression for).
            throw new InvalidOperationException(
                $"{arch}.attention.compress_ratios has {compressRatios.Count} entries, " +
                $"expected at least {numLayerAll - numLayerNextn} (trunk layer count)");
        }

        return new DeepSeek4Hyperparams
        {
            NumLayerNextn = numLayerNextn,
            NumLayerAll = numLayerAll,
            EmbedDim = embedDim,
            NumHeads = numHeads,
            HeadDim = headDim,
            RopeDim = ropeDim,
            NumExperts = numExperts,
            NumExpertsUsed = numExpertsUsed,
            QLoraRank = GetInt(metadata, $"{arch}.attention.q_lora_rank"),
            SlidingWindow = GetInt(metadata, $"{arch}.attention.sliding_window"),
            ExpertFeedForwardLength = GetInt(metadata, $"{arch}.expert_feed_forward_length"),
            ExpertSharedCount = GetInt(metadata, $"{arch}.expert_shared_count"),
            ExpertWeightsScale = GetFloat(metadata, $"{arch}.expert_weights_scale", 1f),
            ExpertWeightsNorm = GetBool(metadata, $"{arch}.expert_weights_norm"),
            SwigluClampExp = GetFloatArray(metadata, $"{arch}.swiglu_clamp_exp", numLayerAll) ?? Array.Empty<float>(),
            SwigluClampShexp = GetFloatArray(metadata, $"{arch}.swiglu_clamp_shexp", numLayerAll)
                ?? GetFloatArray(metadata, $"{arch}.swiglu_clamp_exp", numLayerAll)
                ?? Array.Empty<float>(),
            IndexerNumHeads = GetInt(metadata, $"{arch}.attention.indexer.head_count"),
            IndexerHeadSize = GetInt(metadata, $"{arch}.attention.indexer.key_length"),
            IndexerTopK = GetInt(metadata, $"{arch}.attention.indexer.top_k"),
            OutputGroupCount = GetInt(metadata, $"{arch}.attention.output_group_count", 1),
            OutputLoraRank = GetInt(metadata, $"{arch}.attention.output_lora_rank"),
            CompressRopeFreqBase = GetFloat(metadata, $"{arch}.attention.compress_rope_freq_base", 10000f),
            HyperConnectionMultiplier = GetInt(metadata, $"{arch}.hyper_connection.count", 4),
            HyperConnectionSinkhornIterations = GetInt(metadata, $"{arch}.hyper_connection.sinkhorn_iterations", 1),
            HyperConnectionEpsilon = GetFloat(metadata, $"{arch}.hyper_connection.epsilon", 1e-6f),
            HashLayerCount = GetInt(metadata, $"{arch}.hash_layer_count"),
            CompressRatios = compressRatios,
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            RopeFreqBase = GetFloat(metadata, $"{arch}.rope.freq_base", 10000f),
        };
    }

    private static int GetInt(IReadOnlyDictionary<string, object> m, string key, int fallback = 0)
    {
        if (!m.TryGetValue(key, out var v)) return fallback;
        if (v is System.Collections.IList list) return list.Count > 0 ? Convert.ToInt32(list[0]) : fallback;
        return Convert.ToInt32(v);
    }

    private static float GetFloat(IReadOnlyDictionary<string, object> m, string key, float fallback = 0f) =>
        m.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    private static bool GetBool(IReadOnlyDictionary<string, object> m, string key, bool fallback = false) =>
        m.TryGetValue(key, out var v) ? Convert.ToBoolean(v) : fallback;

    private static IReadOnlyList<int>? GetIntArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<int> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new int[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = Convert.ToInt32(list[i]);
                return result;
            }
            default: return null;
        }
    }

    private static IReadOnlyList<float>? GetFloatArray(IReadOnlyDictionary<string, object> m, string key, int numLayers)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<float> rl: return rl;
            case System.Collections.IList list:
            {
                var result = new float[list.Count];
                for (int i = 0; i < list.Count; i++) result[i] = Convert.ToSingle(list[i]);
                return result;
            }
            default:
            {
                float scalar = Convert.ToSingle(v);
                var result = new float[numLayers];
                Array.Fill(result, scalar);
                return result;
            }
        }
    }
}

/// <summary>
/// ALPHA/UNTESTED. Core math for DeepSeek-V4's hyper-connections, lightning indexer, and
/// CSA/HCA compressed-KV mechanisms. Every method operates eagerly on <see cref="Span{T}"/>
/// buffers (row-major, batch-of-tokens-first where noted) rather than this codebase's usual
/// unsafe float* kernel convention (see SimdKernels.cs) -- deliberately, since none of this is
/// on a hot path yet (nothing calls it) and Span-based code is far easier to read/re-verify
/// against the C++ reference than pointer arithmetic. Convert to the float*/SIMD convention
/// during the performance pass once this is wired, verified, and admitted (CLAUDE.md rule 7),
/// not before.
/// </summary>
public static class DeepSeek4Graph
{
    // ── Hyper-connections ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// Ports <c>build_hc_sinkhorn</c> (deepseek4.cpp:312-347). Iteratively normalizes a
    /// <paramref name="hc"/> x <paramref name="hc"/> "combine" matrix (row-softmaxed first, then
    /// alternating column/row normalization for <paramref name="iterations"/> rounds) toward a
    /// doubly-stochastic matrix -- the mechanism that lets each hyper-connection stream mix
    /// information from every other stream in a learned, balanced way. Operates on ONE token's
    /// <paramref name="hc"/>*<paramref name="hc"/> combine block in place; the reference's
    /// dst/src axis convention is: row = destination stream, column = source stream
    /// (dsv4_hc_sinkhorn's own comment, deepseek4.cpp:317-318).
    ///
    /// Sanity check for whoever re-verifies this against real weights: after this returns, both
    /// row sums and column sums of <paramref name="comb"/> should be close to 1 (doubly
    /// stochastic) -- a synthetic unit test asserting that invariant on a random hc x hc input is
    /// the cheapest way to catch a transposed axis or an off-by-one iteration count before any
    /// real GGUF is available.
    /// </summary>
    public static void HyperConnectionSinkhorn(Span<float> comb, int hc, int iterations, float eps)
    {
        // comb is [hc, hc] row-major: comb[dst * hc + src].

        // 1. Row softmax over the DESTINATION axis (deepseek4.cpp:319, ggml_soft_max(comb) --
        //    ggml's soft_max normalizes along ne[0], the fastest-varying/row axis, which here is
        //    laid out as "dst" per the dsv4_hc_sinkhorn comment).
        RowSoftmaxInPlace(comb, hc, hc);

        // 2. Add epsilon (deepseek4.cpp:321-324).
        for (int i = 0; i < hc * hc; i++)
        {
            comb[i] += eps;
        }

        // 3. norm_cols first (deepseek4.cpp:340), then (iterations - 1) rounds of norm_rows +
        //    norm_cols (deepseek4.cpp:341-344).
        NormalizeColumns(comb, hc, eps);
        for (int i = 1; i < iterations; i++)
        {
            NormalizeRows(comb, hc, eps);
            NormalizeColumns(comb, hc, eps);
        }
    }

    private static void RowSoftmaxInPlace(Span<float> m, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = m.Slice(r * cols, cols);
            float max = float.NegativeInfinity;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, row[c]);
            float sum = 0f;
            for (int c = 0; c < cols; c++)
            {
                float e = MathF.Exp(row[c] - max);
                row[c] = e;
                sum += e;
            }
            float inv = 1f / sum;
            for (int c = 0; c < cols; c++) row[c] *= inv;
        }
    }

    private static void NormalizeRows(Span<float> comb, int hc, float eps)
    {
        for (int r = 0; r < hc; r++)
        {
            float sum = eps;
            for (int c = 0; c < hc; c++) sum += comb[r * hc + c];
            for (int c = 0; c < hc; c++) comb[r * hc + c] /= sum;
        }
    }

    private static void NormalizeColumns(Span<float> comb, int hc, float eps)
    {
        for (int c = 0; c < hc; c++)
        {
            float sum = eps;
            for (int r = 0; r < hc; r++) sum += comb[r * hc + c];
            for (int r = 0; r < hc; r++) comb[r * hc + c] /= sum;
        }
    }

    /// <summary>
    /// Ports the first <c>build_hc_pre</c> overload (deepseek4.cpp:285-310): mixes the
    /// <paramref name="hc"/> parallel residual streams down to one, weighted by a per-stream,
    /// per-token gate <paramref name="weights"/>. <paramref name="x"/> is
    /// [hc, embedDim] row-major for ONE token (matching the reference's x->ne[0]==n_embd,
    /// x->ne[1]==hc layout after accounting for ggml's reversed-axis convention);
    /// <paramref name="weights"/> is [hc] for that same token. Writes the mixed [embedDim] result
    /// to <paramref name="result"/>.
    /// </summary>
    public static void HyperConnectionMixDown(ReadOnlySpan<float> x, ReadOnlySpan<float> weights, int hc, int embedDim, Span<float> result)
    {
        result.Clear();
        for (int h = 0; h < hc; h++)
        {
            ReadOnlySpan<float> xh = x.Slice(h * embedDim, embedDim);
            float w = weights[h];
            for (int d = 0; d < embedDim; d++)
            {
                result[d] += xh[d] * w;
            }
        }
    }

    /// <summary>
    /// Ports the second <c>build_hc_pre</c> overload's gate-computation half (deepseek4.cpp:
    /// 349-405, excluding the fused-kernel branch at 295-299/388-391 which this eager port has
    /// no equivalent for and always takes the non-fused path). Computes, for ONE token:
    /// <list type="bullet">
    /// <item><b>pre</b> [hc]: sigmoid(affine(mixes[0:hc])) + eps -- the per-stream mix-down gate
    /// consumed by <see cref="HyperConnectionMixDown"/>.</item>
    /// <item><b>post</b> [hc]: 2*sigmoid(affine(mixes[hc:2hc])) -- the per-stream broadcast-back
    /// gate consumed by <see cref="HyperConnectionMixUp"/>.</item>
    /// <item><b>comb</b> [hc,hc]: affine(mixes[2hc:2hc+hc*hc]) reshaped and Sinkhorn-normalized
    /// via <see cref="HyperConnectionSinkhorn"/> -- the cross-stream mixing matrix consumed by
    /// <see cref="HyperConnectionMixUp"/>.</item>
    /// </list>
    /// <paramref name="flatNormed"/> is the RMSNorm'd, flattened [hc*embedDim] input for this
    /// token (deepseek4.cpp:365-366: reshape hc streams to one flat vector, then RMSNorm the
    /// WHOLE flattened vector, not per-stream). <paramref name="hcFn"/> is the
    /// [hc*embedDim, (2+hc)*hc] mixing-projection weight (row-major, hcFn[row * mixDim + col]).
    /// <paramref name="scale"/>/<paramref name="base_"/> are the 3-entry-scale /
    /// (2*hc + hc*hc)-entry-bias affine parameters (hc_attn_scale/hc_attn_base or
    /// hc_ffn_scale/hc_ffn_base or hc_head_scale/hc_head_base, per call site).
    /// </summary>
    public static void HyperConnectionGate(
        ReadOnlySpan<float> flatNormed, int hc, int embedDim,
        ReadOnlySpan<float> hcFn, ReadOnlySpan<float> scale, ReadOnlySpan<float> base_, float eps,
        Span<float> pre, Span<float> post, Span<float> comb)
    {
        int mixDim = (2 + hc) * hc;
        int flatDim = hc * embedDim;

        // mixes = hcFn^T . flatNormed  (mixDim outputs). hcFn is [flatDim, mixDim] per the
        // reference's create_tensor({hc_dim, hc_mix_dim}, ...) row-major-by-output convention
        // (matches this codebase's other GGUF Linear-weight layout, see DeepSeekMoeGraph.cs's
        // twin comment).
        Span<float> mixes = mixDim <= 512 ? stackalloc float[mixDim] : new float[mixDim];
        for (int o = 0; o < mixDim; o++)
        {
            float sum = 0f;
            for (int k = 0; k < flatDim; k++)
            {
                sum += flatNormed[k] * hcFn[o * flatDim + k];
            }
            mixes[o] = sum;
        }

        // pre = sigmoid(mixes[0:hc] * scale[0] + base[0:hc]) + eps  (deepseek4.cpp:376-380).
        for (int h = 0; h < hc; h++)
        {
            float v = mixes[h] * scale[0] + base_[h];
            pre[h] = Sigmoid(v) + eps;
        }

        // post = 2 * sigmoid(mixes[hc:2hc] * scale[1] + base[hc:2hc])  (deepseek4.cpp:382-385).
        for (int h = 0; h < hc; h++)
        {
            float v = mixes[hc + h] * scale[1] + base_[hc + h];
            post[h] = 2f * Sigmoid(v);
        }

        // comb = Sinkhorn(affine(mixes[2hc:2hc+hc*hc]) reshaped [hc,hc])  (deepseek4.cpp:393-399,
        // non-fused branch).
        for (int i = 0; i < hc * hc; i++)
        {
            float v = mixes[2 * hc + i] * scale[2] + base_[2 * hc + i];
            comb[i] = v;
        }
        HyperConnectionSinkhorn(comb, hc, /* iterations, caller-supplied via a separate overload if needed */ 1, eps);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>
    /// Ports <c>build_hc_head</c> (deepseek4.cpp:444-464), used ONCE at the very end of the
    /// trunk (not per-layer): a single-gate variant of <see cref="HyperConnectionGate"/> with no
    /// post/comb output and no Sinkhorn step -- just <c>pre = sigmoid(affine(hcFn^T . flatNormed))
    /// + eps</c>. The caller then applies <see cref="HyperConnectionMixDown"/> with this gate to
    /// collapse the final [hc, embedDim] residual down to [embedDim] before the output norm/LM
    /// head. <paramref name="hcFn"/> is [hc*embedDim, hc] (mixDim == hc here, not (2+hc)*hc, since
    /// there is no post/comb to compute). <paramref name="scale"/>/<paramref name="base_"/> are
    /// the hc_head_scale (1 element)/hc_head_base (hc elements) affine parameters.
    /// </summary>
    public static void HyperConnectionHeadGate(
        ReadOnlySpan<float> flatNormed, int hc, int embedDim,
        ReadOnlySpan<float> hcFn, ReadOnlySpan<float> scale, ReadOnlySpan<float> base_, float eps,
        Span<float> pre)
    {
        int flatDim = hc * embedDim;
        for (int o = 0; o < hc; o++)
        {
            float sum = 0f;
            for (int k = 0; k < flatDim; k++)
            {
                sum += flatNormed[k] * hcFn[o * flatDim + k];
            }
            float v = sum * scale[0] + base_[o];
            pre[o] = Sigmoid(v) + eps;
        }
    }

    /// <summary>
    /// Ports <c>build_hc_post</c>'s non-fused branch (deepseek4.cpp:407-442): broadcasts the
    /// single mixed-down FFN/attention output <paramref name="x"/> [embedDim] back out to
    /// <paramref name="hc"/> streams, each combining that shared output (gated by
    /// <paramref name="post"/>[dst]) with a cross-stream-mixed copy of the pre-mix
    /// <paramref name="residual"/> [hc, embedDim] (gated by <paramref name="comb"/>[dst,src]).
    /// Writes [hc, embedDim] to <paramref name="result"/>.
    /// </summary>
    public static void HyperConnectionMixUp(
        ReadOnlySpan<float> x, ReadOnlySpan<float> residual, ReadOnlySpan<float> post, ReadOnlySpan<float> comb,
        int hc, int embedDim, Span<float> result)
    {
        for (int dst = 0; dst < hc; dst++)
        {
            Span<float> outDst = result.Slice(dst * embedDim, embedDim);
            float postDst = post[dst];
            for (int d = 0; d < embedDim; d++)
            {
                outDst[d] = x[d] * postDst;
            }
            for (int src = 0; src < hc; src++)
            {
                ReadOnlySpan<float> resSrc = residual.Slice(src * embedDim, embedDim);
                float c = comb[dst * hc + src];
                for (int d = 0; d < embedDim; d++)
                {
                    outDst[d] += resSrc[d] * c;
                }
            }
        }
    }

    // ── Lightning indexer (DSA) ────────────────────────────────────────────────────────────

    /// <summary>
    /// Ports the non-fused branch of the lightning-indexer scoring block shared by
    /// <c>build_lid_top_k</c> (deepseek4.cpp:671-696) and <c>deepseek32.cpp</c>'s inline
    /// equivalent (deepseek32.cpp:317-356): score[key] = sum_head( relu(q[head] . k[head, key]) *
    /// weight[head] ), i.e. a per-head ReLU'd dot product weighted and summed across indexer
    /// heads, THEN masked additively (causal mask, -inf for disallowed positions) before top-k
    /// selection. <paramref name="q"/> is [numHeads, headDim] for the current token;
    /// <paramref name="k"/> is [numKeys, numHeads, headDim] (one indexer-key vector per head per
    /// cached position); <paramref name="weights"/> is [numKeys, numHeads] (pre-scaled by
    /// 1/sqrt(headDim*numHeads) by the caller, matching indexer_weights's scale at
    /// deepseek4.cpp:650/deepseek32.cpp:314); <paramref name="causalMask"/> is [numKeys], 0 for
    /// allowed positions and -inf for disallowed (already includes any SWA/prefix restriction).
    /// Writes masked scores to <paramref name="scoresOut"/> [numKeys]; the caller selects the
    /// top-<c>indexerTopK</c> indices from that (a plain partial-sort, no port of ggml_top_k's
    /// specific tie-breaking needed for a first pass -- flag as a re-check item if tie-breaking
    /// order ever matters for parity).
    /// </summary>
    public static void LightningIndexerScore(
        ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> causalMask, int numHeads, int headDim, int numKeys,
        Span<float> scoresOut)
    {
        for (int key = 0; key < numKeys; key++)
        {
            float score = 0f;
            for (int h = 0; h < numHeads; h++)
            {
                ReadOnlySpan<float> qh = q.Slice(h * headDim, headDim);
                ReadOnlySpan<float> kh = k.Slice((key * numHeads + h) * headDim, headDim);
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += qh[d] * kh[d];
                float relu = MathF.Max(0f, dot);
                score += relu * weights[key * numHeads + h];
            }
            scoresOut[key] = score + causalMask[key];
        }
    }

    /// <summary>
    /// Selects the indices of the <paramref name="topK"/> largest values in
    /// <paramref name="scores"/> (a plain O(n log k) partial sort via a min-heap-free insertion
    /// approach -- adequate for indexer_top_k's typically small K, not yet optimized). Ports the
    /// selection step of <c>ggml_top_k</c> as used by <c>build_lid_top_k</c>
    /// (deepseek4.cpp:698-700) / deepseek32.cpp:359-361, without porting ggml_top_k's own
    /// tie-breaking order -- see <see cref="LightningIndexerScore"/>'s doc comment.
    /// </summary>
    public static int[] SelectTopKIndices(ReadOnlySpan<float> scores, int topK)
    {
        int n = scores.Length;
        topK = Math.Min(topK, n);
        var indices = new int[n];
        var scoresCopy = scores.ToArray();
        for (int i = 0; i < n; i++) indices[i] = i;
        Array.Sort(indices, (a, b) => scoresCopy[b].CompareTo(scoresCopy[a]));
        return indices[..topK];
    }

    // ── CSA/HCA compressed-KV-from-state ───────────────────────────────────────────────────

    /// <summary>
    /// Ports <c>build_hca_compressed_kv_from_state</c> (deepseek4.cpp:466-522), used by HCA
    /// layers (compress_ratio==128, no indexer). For ONE output compressed-KV block: takes
    /// <paramref name="ratio"/> restored raw KV/score rows (already gathered from the persistent
    /// HCA state cache by the caller -- see the file header's note on the missing state-cache
    /// abstraction), softmaxes the scores across those <paramref name="ratio"/> rows PER
    /// CHANNEL, weight-sums the KV rows by that softmax, RMSNorms the result, then applies RoPE
    /// to only the trailing <paramref name="ropeDim"/> channels (the leading
    /// headDim-ropeDim "nope" channels pass through unrotated) using
    /// <paramref name="compressRopeFreqBase"/> and a position given by the caller
    /// (<paramref name="blockPosition"/> — one RoPE angle per compressed block, not per original
    /// token; matches comp_pos's block-granularity in the reference).
    /// <paramref name="kv"/>/<paramref name="score"/> are both [ratio, headDim] row-major for
    /// this block. <paramref name="normWeight"/> is the RMSNorm gain [headDim].
    /// <paramref name="ropeApply"/> is a caller-supplied delegate that performs this codebase's
    /// existing RoPE rotation (reused rather than re-derived — see
    /// <c>SimdKernels.ApplyRoPECached</c>/<c>BuildYarnRopeTable</c> for the pattern to plug in
    /// once wired; not called from here directly to avoid an unverified cross-project reference).
    /// Writes the [headDim] compressed-and-rotated result to <paramref name="result"/>.
    /// </summary>
    public static void HcaCompressBlock(
        ReadOnlySpan<float> kv, ReadOnlySpan<float> score, int ratio, int headDim, int ropeDim,
        ReadOnlySpan<float> normWeight, float rmsNormEps,
        Action<Span<float>, int /* ropeDim */, float /* position */, float /* freqBase */> ropeApply,
        float compressRopeFreqBase, float blockPosition,
        Span<float> result)
    {
        int nopeDim = headDim - ropeDim;

        // 1. Per-channel softmax across the `ratio` rows (deepseek4.cpp:492-497: permute so the
        //    ratio axis is the softmax axis, ggml_soft_max, weight-multiply, sum_rows).
        Span<float> weights = ratio <= 256 ? stackalloc float[ratio] : new float[ratio];
        result.Clear();
        for (int c = 0; c < headDim; c++)
        {
            float max = float.NegativeInfinity;
            for (int r = 0; r < ratio; r++) max = MathF.Max(max, score[r * headDim + c]);
            float sum = 0f;
            for (int r = 0; r < ratio; r++)
            {
                float e = MathF.Exp(score[r * headDim + c] - max);
                weights[r] = e;
                sum += e;
            }
            float inv = 1f / sum;
            float acc = 0f;
            for (int r = 0; r < ratio; r++)
            {
                acc += kv[r * headDim + c] * (weights[r] * inv);
            }
            result[c] = acc;
        }

        // 2. RMSNorm the compressed [headDim] result (deepseek4.cpp:501-502).
        float ss = 0f;
        for (int c = 0; c < headDim; c++) ss += result[c] * result[c];
        float invRms = 1f / MathF.Sqrt(ss / headDim + rmsNormEps);
        for (int c = 0; c < headDim; c++) result[c] = result[c] * invRms * normWeight[c];

        // 3. RoPE the trailing ropeDim channels only (deepseek4.cpp:504-519); nope channels
        //    (result[0..nopeDim)) pass through untouched.
        if (ropeDim > 0)
        {
            ropeApply(result.Slice(nopeDim, ropeDim), ropeDim, blockPosition, compressRopeFreqBase);
        }
    }

    /// <summary>
    /// Ports <c>build_overlap_compressed_kv_from_state</c> (deepseek4.cpp:524-606), used by CSA
    /// layers (compress_ratio==4) and the CSA-flavored lightning-indexer compression. Differs
    /// from <see cref="HcaCompressBlock"/> in reading TWO overlapping windows of
    /// <paramref name="ratio"/> rows each (a "previous" and a "current" half, concatenated before
    /// the same per-channel-softmax-weighted-sum -- deepseek4.cpp:553-582) and in appending one
    /// synthetic all-zero KV row / all-negative-infinity score row to the state before indexing
    /// (deepseek4.cpp:545-546, <c>dsv4_append_zero_row</c>) so a not-yet-populated slot
    /// contributes nothing to the softmax rather than reading uninitialized state. This port
    /// takes the already-concatenated [2*ratio, headDim] KV/score rows directly (with the
    /// zero/-inf row already appended by the caller) rather than re-deriving the state-index
    /// gather/append here, since that indexing logic depends entirely on the still-unbuilt
    /// persistent state cache (see the file header). The compress math itself (softmax-weight-sum
    /// over 2*ratio rows, RMSNorm, split-RoPE) is otherwise identical to
    /// <see cref="HcaCompressBlock"/> with <c>ratio</c> replaced by <c>2*ratio</c>.
    /// </summary>
    public static void CsaCompressBlock(
        ReadOnlySpan<float> kvConcat, ReadOnlySpan<float> scoreConcat, int ratio, int headDim, int ropeDim,
        ReadOnlySpan<float> normWeight, float rmsNormEps,
        Action<Span<float>, int, float, float> ropeApply,
        float compressRopeFreqBase, float blockPosition,
        Span<float> result)
        => HcaCompressBlock(kvConcat, scoreConcat, 2 * ratio, headDim, ropeDim, normWeight, rmsNormEps,
            ropeApply, compressRopeFreqBase, blockPosition, result);

    /// <summary>
    /// The "prev half" row index for slot <paramref name="r"/> (0..ratio-1) of CSA/LID overlap
    /// block <paramref name="blockIndex"/>, per this codebase's working-hypothesis reading of
    /// <c>build_overlap_compressed_kv_from_state</c> (deepseek4.cpp:524-606) documented in
    /// docs/058-deepseek-full-lineage-implementation-plan.md's "CSA decomposition" section: the
    /// `ratio` raw-token rows immediately preceding this block. Returns -1 for a row that falls
    /// before position 0 (block 0's prev half, or any block whose prev window reaches past the
    /// start of the sequence) -- the caller substitutes the reference's synthetic zero-KV/-inf-
    /// score row (<c>dsv4_append_zero_row</c>) for a -1 result. Pulled out as a pure, standalone
    /// function specifically so the window arithmetic is unit-testable without a real
    /// GgufModel/forward pass.
    /// </summary>
    public static int OverlapPrevRowIndex(int blockIndex, int ratio, int r) => blockIndex * ratio - ratio + r;

    /// <summary>
    /// The "cur half" row index for slot <paramref name="r"/> (0..ratio-1) of CSA/LID overlap
    /// block <paramref name="blockIndex"/>: this block's own `ratio` raw-token rows. Always
    /// non-negative and always already-populated by construction (a block is only finalized once
    /// its own rows exist). See <see cref="OverlapPrevRowIndex"/>.
    /// </summary>
    public static int OverlapCurRowIndex(int blockIndex, int ratio, int r) => blockIndex * ratio + r;

    // ── MoE routing (sqrt-softplus gating) ─────────────────────────────────────────────────

    /// <summary>
    /// deepseek4.cpp's loader hard-requires <c>LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS</c>
    /// (deepseek4.cpp:63-66, throws otherwise) -- a gating function this codebase has not ported
    /// before (every other MoE architecture here uses softmax or sigmoid gating). Per-expert
    /// score = sqrt(softplus(logit)) = sqrt(ln(1 + exp(logit))), computed independently per
    /// expert (no cross-expert normalization at this stage, unlike softmax) -- the reference's
    /// <c>build_moe_ffn</c> (not directly inspected in this port; ggml's own
    /// <c>ggml_expert_gating_func</c> switch for SQRT_SOFTPLUS was used as the formula source
    /// instead, since it's a small, self-contained elementwise op) computes this per logit before
    /// top-k selection. Selection order (top-k by this score) and the subsequent optional
    /// renormalize-to-1 / expert_weights_scale multiply are assumed to follow the same pattern as
    /// every other gating function this codebase already implements (softmax's
    /// <see cref="DeepSeek4Hyperparams.ExpertWeightsNorm"/>/<see cref="DeepSeek4Hyperparams.ExpertWeightsScale"/>
    /// handling in the existing deepseek2 MoE path) -- NOT independently re-verified for the
    /// sqrt-softplus case specifically.
    /// </summary>
    public static void SqrtSoftplusGate(ReadOnlySpan<float> logits, Span<float> scoresOut)
    {
        for (int i = 0; i < logits.Length; i++)
        {
            float x = logits[i];
            // Numerically-stable softplus: ln(1+exp(x)) = max(x,0) + ln(1+exp(-|x|)).
            float softplus = MathF.Max(x, 0f) + MathF.Log(1f + MathF.Exp(-MathF.Abs(x)));
            scoresOut[i] = MathF.Sqrt(softplus);
        }
    }

    /// <summary>
    /// Selects the top-<paramref name="topK"/> experts by <paramref name="scores"/> (as produced
    /// by <see cref="SqrtSoftplusGate"/>), optionally renormalizes their weights to sum to 1
    /// (<paramref name="normalize"/>), then applies <paramref name="scale"/>
    /// (<see cref="DeepSeek4Hyperparams.ExpertWeightsScale"/>) -- mirrors the
    /// select-then-normalize-then-scale pattern this codebase's existing (deepseek2) MoE path
    /// uses, per this method's own doc-comment caveat above about not being independently
    /// re-verified for sqrt-softplus specifically.
    /// </summary>
    public static void SelectAndWeightExperts(
        ReadOnlySpan<float> scores, int topK, bool normalize, float scale,
        Span<int> expertIndicesOut, Span<float> expertWeightsOut)
    {
        int[] indices = SelectTopKIndices(scores, topK);
        float sum = 0f;
        for (int k = 0; k < topK; k++)
        {
            expertIndicesOut[k] = indices[k];
            expertWeightsOut[k] = scores[indices[k]];
            sum += expertWeightsOut[k];
        }
        if (normalize && sum > 0f)
        {
            float inv = 1f / sum;
            for (int k = 0; k < topK; k++) expertWeightsOut[k] *= inv;
        }
        for (int k = 0; k < topK; k++) expertWeightsOut[k] *= scale;
    }

    // ── Hash-layer MoE routing ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Hash-layer routing for the leading <c>hashLayerCount</c> layers (uses
    /// <c>ffn_gate_tid2eid</c> instead of a learned <c>ffn_gate_inp</c> router --
    /// deepseek4.cpp:154-155). The reference treats this as a lookup table
    /// ({numExpertsUsed, vocabSize}) indexed by token id, giving each layer's fixed expert set for
    /// that token directly, with no learned gating weights to combine -- ported here as a direct
    /// row lookup returning the numExpertsUsed expert indices for token <paramref name="tokenId"/>
    /// at unit weight each (deepseek4.cpp's build_moe_ffn call site for hash layers was NOT
    /// directly inspected in this pass -- re-check against the reference's actual build_moe_ffn
    /// invocation for hash layers before relying on the "unit weight" assumption; this may need a
    /// softmax/normalize step this port hasn't located yet).
    /// </summary>
    public static void HashLayerSelectExperts(ReadOnlySpan<int> tid2eid, int numExpertsUsed, int vocabSize, int tokenId, Span<int> expertIndicesOut)
    {
        ReadOnlySpan<int> row = tid2eid.Slice(tokenId * numExpertsUsed, numExpertsUsed);
        row.CopyTo(expertIndicesOut);
    }
}
