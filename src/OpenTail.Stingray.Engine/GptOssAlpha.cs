namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- GPT-OSS ("gpt-oss" GGUF architecture, llama.cpp class `openai_moe`)
// implementation. Ported from examples/llama.cpp/llama.cpp/src/models/openai-moe.cpp (177 lines,
// read in full) plus the specific ggml primitives it calls into (attention sinks -- ggml's
// ggml_soft_max_add_sinks, examples/ggml/src/ggml-cpu/ops.cpp:5541-5551; the OAI SwiGLU variant
// -- ggml_swiglu_oai, examples/ggml/src/ggml-cuda/unary.cuh:107-114; select-then-softmax MoE
// gating -- LAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT, llama-graph.cpp:1970-1973/2048-2053).
// See docs/060-gpt-oss-implementation-plan.md for the full plan and the architecture-mapping
// table (what's genuinely new vs. already-implemented elsewhere in this codebase).
//
// NO real gpt-oss GGUF was loaded while writing this -- a download was started in the background
// in parallel with this file being written, per explicit user direction not to wait for it.
// Every formula below is "believed correct from reading the reference," not verified.
// "gpt-oss" is NOT admitted in ModelCompatibility.cs.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. gpt-oss-specific hyperparameters, mirroring the GGUF keys
/// <c>llama_model_openai_moe::load_arch_hparams</c> reads (openai-moe.cpp:3-22).
/// </summary>
public sealed record GptOssHyperparams
{
    public int NumLayer { get; init; }
    public int EmbedDim { get; init; }
    public int NumHeads { get; init; }
    public int NumHeadsKv { get; init; }
    public int HeadDim { get; init; } // n_rot in the reference -- gpt-oss has no separate nope/rope split, the whole head is rotated.
    public int NumExperts { get; init; }
    public int NumExpertsUsed { get; init; }
    public int VocabSize { get; init; }

    /// <summary>RMSNorm epsilon ({arch}.attention.layer_norm_rms_epsilon). openai-moe.cpp:4.</summary>
    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>Expert FFN intermediate width ({arch}.expert_feed_forward_length). openai-moe.cpp:5.</summary>
    public int ExpertFeedForwardLength { get; init; }

    /// <summary>Sliding-window size in tokens ({arch}.attention.sliding_window). openai-moe.cpp:6.</summary>
    public int SlidingWindow { get; init; }

    /// <summary>
    /// SWA alternation period ({arch}.attention.sliding_window_pattern). openai-moe.cpp:9-11
    /// defaults this to 2 if the GGUF doesn't declare it (unusual among this codebase's other SWA
    /// architectures, which all read an explicit period -- gpt-oss's own reference hardcodes the
    /// literal `2` as the fallback value passed to get_key_or_arr, not just a C# default).
    /// </summary>
    public int SwaPeriod { get; init; } = 2;

    /// <summary>Standard (global-layer) RoPE base frequency ({arch}.rope.freq_base).</summary>
    public float RopeFreqBase { get; init; } = 10000f;

    /// <summary>
    /// SWA-layer RoPE base frequency ({arch}.rope.freq_base_swa) -- defaults to
    /// <see cref="RopeFreqBase"/> if the GGUF doesn't declare a separate value (openai-moe.cpp:
    /// 13-15: cparams start equal, then the SWA-specific key overrides only if present).
    /// </summary>
    public float RopeFreqBaseSwa { get; init; } = 10000f;

    /// <summary>
    /// True for layer <paramref name="il"/> being sliding-window, false for global/full attention.
    /// Reproduces llama_hparams::set_swa_pattern's dense_first=false formula (llama-hparams.cpp:
    /// 13-16): <c>(il % swaPeriod) &lt; (swaPeriod - 1)</c>. For the default swaPeriod=2, this
    /// means EVEN layers (0, 2, 4, ...) are SWA and ODD layers are global -- SWA-first,
    /// alternating strictly 1:1. NOT independently re-verified against a real GGUF's actual
    /// per-layer behavior.
    /// </summary>
    public bool IsSwaLayer(int il) => SwaPeriod == 0 || (il % SwaPeriod) < (SwaPeriod - 1);

    public static GptOssHyperparams FromGgufMetadata(
        IReadOnlyDictionary<string, object> metadata, string arch, int numLayer,
        int embedDim, int numHeads, int numHeadsKv, int headDim, int numExperts,
        int numExpertsUsed, int vocabSize)
    {
        float ropeFreqBase = GetFloat(metadata, $"{arch}.rope.freq_base", 10000f);
        return new GptOssHyperparams
        {
            NumLayer = numLayer,
            EmbedDim = embedDim,
            NumHeads = numHeads,
            NumHeadsKv = numHeadsKv,
            HeadDim = headDim,
            NumExperts = numExperts,
            NumExpertsUsed = numExpertsUsed,
            VocabSize = vocabSize,
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            ExpertFeedForwardLength = GetInt(metadata, $"{arch}.expert_feed_forward_length"),
            SlidingWindow = GetInt(metadata, $"{arch}.attention.sliding_window"),
            SwaPeriod = GetInt(metadata, $"{arch}.attention.sliding_window_pattern", 2),
            RopeFreqBase = ropeFreqBase,
            RopeFreqBaseSwa = GetFloat(metadata, $"{arch}.rope.freq_base_swa", ropeFreqBase),
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
}

/// <summary>
/// ALPHA/UNTESTED. Core math for GPT-OSS's three new primitives: attention sinks, the OAI SwiGLU
/// variant, and select-then-softmax MoE gating. Span-based, not the unsafe float* SIMD
/// convention -- same reasoning as DeepSeek4Graph (nothing calls this yet; convert to SIMD only
/// after this is wired, verified, and admitted, per CLAUDE.md rule 7).
/// </summary>
public static class GptOssGraph
{
    /// <summary>
    /// Softmax with an attention-sink correction, per <c>ggml_soft_max_add_sinks</c>
    /// (examples/ggml/src/ggml-cpu/ops.cpp:5541-5551): the sink is one extra "virtual key" whose
    /// score participates in the denominator but contributes NOTHING to the weighted-V sum (it
    /// has no corresponding V row -- <paramref name="scores"/> is normalized in place at its
    /// original length, the sink is never appended to it). Mathematically:
    /// <c>max' = max(max(scores), sink)</c>; each score's softmax numerator is computed against
    /// <c>max'</c> as usual; the denominator additionally includes <c>exp(sink - max')</c>. This
    /// uniformly shrinks every real key's weight — output values sum to LESS than 1 whenever the
    /// sink absorbs a non-trivial share, by construction (a cheap invariant to unit-test:
    /// sum-of-output &lt; 1 when sink is present and not vanishingly small relative to the real
    /// scores, sum-of-output == 1 when <paramref name="sink"/> is null).
    /// </summary>
    public static void SoftmaxWithSink(Span<float> scores, float? sink)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) max = MathF.Max(max, scores[i]);
        if (sink is { } s) max = MathF.Max(max, s);

        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        if (sink is { } s2)
        {
            sum += MathF.Exp(s2 - max);
        }

        float inv = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= inv;
    }

    /// <summary>
    /// GPT-OSS's specific clamped SwiGLU variant, <c>ggml_swiglu_oai</c> (formula extracted from
    /// examples/ggml/src/ggml-cuda/unary.cuh:107-114 — a plain scalar op, easiest to read there;
    /// the CPU reference at examples/ggml/src/ggml-cpu/ops.cpp:3325 implements the identical
    /// formula and is the one this should ultimately be diffed against). <paramref name="alpha"/>
    /// and <paramref name="limit"/> default to the reference's own compile-time constants
    /// (1.702/7.0 — the reference itself comments "TODO: move to hparams?", i.e. even upstream
    /// treats these as provisional, not GGUF-declared). CRITICAL DETAIL, easy to get wrong by
    /// pattern-matching against this codebase's existing multiplicative SwiGLU kernels: the
    /// combine is <c>swish * (1 + up_clamped)</c> — ADDITIVE, not <c>gate * up</c>.
    /// </summary>
    public static void SwigluOai(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> output, float alpha = 1.702f, float limit = 7.0f)
    {
        for (int i = 0; i < gate.Length; i++)
        {
            float x = MathF.Min(gate[i], limit);
            float g = MathF.Max(MathF.Min(up[i], limit), -limit);
            float swish = x / (1f + MathF.Exp(-alpha * x));
            output[i] = swish * (1f + g);
        }
    }

    /// <summary>
    /// GPT-OSS's MoE gating: select the top-<paramref name="topK"/> experts by RAW router logit
    /// (NOT a softmax/sigmoid probability), THEN softmax ONLY the selected subset
    /// (<c>LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT</c>, llama-graph.cpp:1970-1973,
    /// 2048-2053). This is the reverse order from every other MoE architecture in this codebase
    /// (which all softmax/sigmoid the FULL expert set first, then select) — a real, load-bearing
    /// difference: the deepseek2 investigation (docs/done/032-...md) found this codebase's MoE
    /// routers can be extremely sensitive to exactly this kind of ordering when logits are close,
    /// so getting this backwards is not a cosmetic bug.
    /// </summary>
    public static void SelectThenSoftmaxGate(ReadOnlySpan<float> logits, int topK, Span<int> expertIndicesOut, Span<float> expertWeightsOut)
    {
        var indices = new int[logits.Length];
        for (int i = 0; i < logits.Length; i++) indices[i] = i;
        var logitsCopy = logits.ToArray();
        Array.Sort(indices, (a, b) => logitsCopy[b].CompareTo(logitsCopy[a]));

        float max = float.NegativeInfinity;
        for (int k = 0; k < topK; k++) max = MathF.Max(max, logitsCopy[indices[k]]);
        float sum = 0f;
        for (int k = 0; k < topK; k++)
        {
            float e = MathF.Exp(logitsCopy[indices[k]] - max);
            expertWeightsOut[k] = e;
            sum += e;
        }
        float inv = 1f / sum;
        for (int k = 0; k < topK; k++)
        {
            expertIndicesOut[k] = indices[k];
            expertWeightsOut[k] *= inv;
        }
    }
}
