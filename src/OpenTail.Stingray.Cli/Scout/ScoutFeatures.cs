using System.Globalization;

namespace OpenTail.Stingray.Cli.Scout;

/// <summary>
/// Conservative, explainable feature rules over GGUF metadata keys and tensor names/shapes (plan §4). Each rule fires only on evidence it
/// can cite. <see cref="Certainty.Known"/> means the structure itself was observed (these tensors / this key exist); it never means the
/// semantics are understood or supported. <see cref="Certainty.Hypothesis"/> means the evidence is partial, name-only or one-sided.
/// Tensor and key names below were checked against the pinned reference (examples/llama.cpp src/llama-arch.cpp, 2026-10-09):
/// attn_qkv, attn_q_norm/attn_k_norm, ffn_gate_inp, ffn_{gate,up,down}_exps / _shexp, attn_q_a / attn_kv_a_mqa, ssm_a / ssm_conv1d,
/// time_mix_*, nextn.*, and the keys %s.nextn_predict_layers, %s.expert_shared_count, %s.attention.kv_lora_rank, %s.rope.dimension_sections,
/// %s.rope.dimension_count, %s.rope.scaling.type, %s.attention.head_count(_kv), %s.attention.indexer.*.
/// </summary>
public static class ScoutFeatures
{
    private const int MaxEvidence = 6;

    public static IReadOnlyList<ScoutFinding> Evaluate(IReadOnlyDictionary<string, object> md, IReadOnlyList<GgufTensorInfo> tensors)
    {
        string arch = md.TryGetValue("general.architecture", out var a) ? Convert.ToString(a, CultureInfo.InvariantCulture) ?? "" : "";
        var ctx = new Ctx(arch, md, tensors);
        var findings = new List<ScoutFinding>();
        Moe(ctx, findings);
        Attention(ctx, findings);
        Mla(ctx, findings);
        Indexer(ctx, findings);
        Recurrent(ctx, findings);
        Mtp(ctx, findings);
        Rope(ctx, findings);
        Projector(ctx, findings);
        ExoticQuant(ctx, findings);
        return findings.OrderBy(f => f.Id, StringComparer.Ordinal).ToArray();
    }

    private sealed class Ctx(string arch, IReadOnlyDictionary<string, object> md, IReadOnlyList<GgufTensorInfo> tensors)
    {
        public string Arch { get; } = arch;
        public IReadOnlyDictionary<string, object> Md { get; } = md;
        public IReadOnlyList<GgufTensorInfo> Tensors { get; } = tensors;

        /// <summary>Tensors whose normalized pattern ends with the given suffix (e.g. "attn_qkv.weight"), name-ordered.</summary>
        public IReadOnlyList<GgufTensorInfo> WithSuffix(string suffix) =>
            Tensors.Where(t => t.Name.EndsWith("." + suffix, StringComparison.Ordinal) || t.Name == suffix).OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();

        public IReadOnlyList<GgufTensorInfo> Containing(string part) =>
            Tensors.Where(t => t.Name.Contains(part, StringComparison.Ordinal)).OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();

        public string Key(string suffix) => $"{Arch}.{suffix}";
        public bool Has(string suffix) => Md.ContainsKey(Key(suffix));

        public double? Number(string suffix) =>
            Md.TryGetValue(Key(suffix), out var v) && v is IConvertible c && v is not string and not bool ? Convert.ToDouble(c, CultureInfo.InvariantCulture) : null;

        public string? Text(string suffix) => Md.TryGetValue(Key(suffix), out var v) ? Convert.ToString(v, CultureInfo.InvariantCulture) : null;

        public EvidenceItem KeyEvidence(string suffix) =>
            new("metadata", Key(suffix), Md.TryGetValue(Key(suffix), out var v) ? Display(v) : null);

        public IReadOnlyList<EvidenceItem> TensorEvidence(IEnumerable<GgufTensorInfo> ts, int max = MaxEvidence) =>
            ts.Take(max).Select(t => new EvidenceItem("tensor", t.Name, $"{t.DType} [{string.Join(",", t.Dimensions.Take(t.NDimensions))}]")).ToArray();
    }

    private static string Display(object v) => v switch
    {
        Array arr => $"array[{arr.Length}]",
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        _ => Convert.ToString(v, CultureInfo.InvariantCulture) ?? "",
    };

    private static void Add(List<ScoutFinding> list, string id, string summary, Certainty c, IEnumerable<EvidenceItem> ev, string caveat) =>
        list.Add(new ScoutFinding(id, summary, c, ev.ToArray(), caveat));

    // ── routed experts ────────────────────────────────────────────────────────

    private static void Moe(Ctx c, List<ScoutFinding> f)
    {
        var router = c.WithSuffix("ffn_gate_inp.weight");
        var exps = c.Tensors.Where(t => t.Name.Contains("ffn_", StringComparison.Ordinal) && t.Name.Contains("_exps.", StringComparison.Ordinal))
            .OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        double? count = c.Number("expert_count");
        bool metaSaysExperts = count is > 1;

        if (exps.Length > 0 || metaSaysExperts)
        {
            bool both = exps.Length > 0 && metaSaysExperts;
            var ev = new List<EvidenceItem>();
            if (c.Has("expert_count")) ev.Add(c.KeyEvidence("expert_count"));
            if (c.Has("expert_used_count")) ev.Add(c.KeyEvidence("expert_used_count"));
            ev.AddRange(c.TensorEvidence(router, 2));
            ev.AddRange(c.TensorEvidence(exps, 3));
            Add(f, "ffn.expert_routed",
                both ? $"Expert-routed FFN: {count:0} experts" + (c.Number("expert_used_count") is double k ? $", top-{k:0}" : "") + ", stacked expert tensors present."
                     : metaSaysExperts ? "Metadata declares routed experts but no stacked expert tensors were found." : "Stacked expert tensors present but no expert_count metadata.",
                both ? Certainty.Known : Certainty.Hypothesis, ev,
                "Structure only. Router normalisation, gating function, expert bias and dense-first-layers rules need the reference forward pass.");
        }

        if (c.Has("expert_shared_count") || c.Containing("_shexp.").Count > 0)
        {
            var sh = c.Containing("_shexp.");
            bool counted = c.Number("expert_shared_count") is > 0;
            var ev = new List<EvidenceItem>();
            if (c.Has("expert_shared_count")) ev.Add(c.KeyEvidence("expert_shared_count"));
            ev.AddRange(c.TensorEvidence(sh, 3));
            Add(f, "ffn.shared_expert", "Shared (always-on) expert pattern.",
                counted && sh.Count > 0 ? Certainty.Known : Certainty.Hypothesis, ev,
                "A shared-expert tensor group changes the FFN sum; confirm scaling/gating against the reference.");
        }
    }

    // ── attention layout ──────────────────────────────────────────────────────

    private static void Attention(Ctx c, List<ScoutFinding> f)
    {
        var fused = c.WithSuffix("attn_qkv.weight");
        var q = c.WithSuffix("attn_q.weight");
        var k = c.WithSuffix("attn_k.weight");
        var v = c.WithSuffix("attn_v.weight");

        if (fused.Count > 0 && q.Count > 0)
            Add(f, "attn.qkv_layout_mixed", "Some layers use a fused attn_qkv, others separate attn_q/k/v.", Certainty.Known,
                c.TensorEvidence(fused, 3).Concat(c.TensorEvidence(q, 3)),
                "Mixed layouts usually mean a hybrid stack (attention layers differ); check which layers are which.");
        else if (fused.Count > 0)
            Add(f, "attn.fused_qkv", "Fused QKV projection.", Certainty.Known, c.TensorEvidence(fused, 3),
                "The Q/K/V split order and head interleaving are model-specific; verify against the reference.");
        else if (q.Count > 0 && k.Count > 0 && v.Count > 0)
            Add(f, "attn.separate_qkv", "Separate Q, K and V projections.", Certainty.Known,
                c.TensorEvidence(q, 1).Concat(c.TensorEvidence(k, 1)).Concat(c.TensorEvidence(v, 1)), "The common layout; no special handling implied.");

        var qn = c.WithSuffix("attn_q_norm.weight");
        var kn = c.WithSuffix("attn_k_norm.weight");
        if (qn.Count > 0 || kn.Count > 0)
            Add(f, "attn.qk_norm", qn.Count > 0 && kn.Count > 0 ? "Per-head RMS norm on both Q and K." : "A norm on only one of Q/K.",
                qn.Count > 0 && kn.Count > 0 ? Certainty.Known : Certainty.Hypothesis,
                c.TensorEvidence(qn, 2).Concat(c.TensorEvidence(kn, 2)),
                "Norm placement relative to RoPE and the norm's epsilon are part of the semantics.");

        double? heads = c.Number("attention.head_count");
        if (heads is null)
        {
            if (c.Md.TryGetValue(c.Key("attention.head_count"), out var arr) && arr is Array)
                Add(f, "attn.head_count_per_layer", "Head counts vary per layer (array metadata).", Certainty.Known, [c.KeyEvidence("attention.head_count")],
                    "Per-layer head counts imply a heterogeneous stack; a single-hyperparameter loader will misread it.");
            return;
        }
        double? kv = c.Number("attention.head_count_kv");
        if (kv is null)
        {
            if (c.Md.TryGetValue(c.Key("attention.head_count_kv"), out var kvArr) && kvArr is Array)
                Add(f, "attn.kv_heads_per_layer", "KV head counts vary per layer (array metadata).", Certainty.Known,
                    [c.KeyEvidence("attention.head_count"), c.KeyEvidence("attention.head_count_kv")],
                    "Heterogeneous KV heads: sliding/global or attention/recurrent layers share the file.");
            return;
        }
        string kind = kv == heads ? "multi-head (MHA)" : kv == 1 ? "multi-query (MQA)" : kv < heads ? $"grouped-query (GQA), {heads / kv:0.##} Q heads per KV head" : "";
        if (kind.Length > 0)
            Add(f, kv == heads ? "attn.mha" : kv == 1 ? "attn.mqa" : "attn.gqa", $"Attention is {kind}.", Certainty.Known,
                [c.KeyEvidence("attention.head_count"), c.KeyEvidence("attention.head_count_kv")],
                "Head counts are from metadata; the actual projection shapes are the ground truth.");
    }

    // ── MLA / sparse indexer ──────────────────────────────────────────────────

    private static void Mla(Ctx c, List<ScoutFinding> f)
    {
        var qa = c.WithSuffix("attn_q_a.weight");
        var kva = c.WithSuffix("attn_kv_a_mqa.weight");
        bool tensors = kva.Count > 0;
        bool meta = c.Has("attention.kv_lora_rank");
        if (!tensors && !meta && qa.Count == 0) return;
        var ev = new List<EvidenceItem>();
        if (meta) ev.Add(c.KeyEvidence("attention.kv_lora_rank"));
        if (c.Has("attention.q_lora_rank")) ev.Add(c.KeyEvidence("attention.q_lora_rank"));
        ev.AddRange(c.TensorEvidence(kva, 2));
        ev.AddRange(c.TensorEvidence(qa, 2));
        Add(f, "attn.mla_low_rank", "Multi-head latent attention (low-rank compressed Q/KV).",
            tensors && meta ? Certainty.Known : Certainty.Hypothesis, ev,
            "Needs the DeepSeek2Mla forward path; the compressed-cache layout and RoPE split are not implied by tensor presence.");
    }

    private static void Indexer(Ctx c, List<ScoutFinding> f)
    {
        var t = c.Containing("indexer");
        bool meta = c.Has("attention.indexer.head_count");
        if (t.Count == 0 && !meta) return;
        var ev = new List<EvidenceItem>();
        if (meta) ev.Add(c.KeyEvidence("attention.indexer.head_count"));
        ev.AddRange(c.TensorEvidence(t, 4));
        Add(f, "attn.sparse_indexer", "A sparse-attention indexer / compressed-attention tensor family.",
            t.Count > 0 && meta ? Certainty.Known : Certainty.Hypothesis, ev,
            "Indexer top-k selection semantics are architecture-specific; treat as an unimplemented forward-pass component until proven otherwise.");
    }

    // ── recurrent / hybrid ────────────────────────────────────────────────────

    private static void Recurrent(Ctx c, List<ScoutFinding> f)
    {
        var ssm = c.Tensors.Where(t => System.Text.RegularExpressions.Regex.IsMatch(t.Name, @"(^|\.)ssm_[a-z0-9_]+(\.|$)")).OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        if (ssm.Length > 0)
        {
            var ssmLayers = Layers(ssm);
            var attnLayers = Layers(c.WithSuffix("attn_q.weight").Concat(c.WithSuffix("attn_qkv.weight")));
            bool hybrid = attnLayers.Count > 0 && ssmLayers.Count > 0 && !ssmLayers.SetEquals(attnLayers);
            var ev = c.TensorEvidence(ssm, 4).ToList();
            if (hybrid) ev.Add(new EvidenceItem("layers", "ssm", ScoutAnalyzer.Ranges(ssmLayers)));
            if (hybrid) ev.Add(new EvidenceItem("layers", "attention", ScoutAnalyzer.Ranges(attnLayers)));
            Add(f, hybrid ? "recurrent.hybrid_ssm" : "recurrent.ssm",
                hybrid ? "State-space/recurrent tensors in some layers and attention in others." : "State-space style (ssm_*) tensors.",
                hybrid ? Certainty.Known : Certainty.Hypothesis, ev,
                "ssm_* names are shared by Mamba-1/2, Gated-DeltaNet and similar; the tensor names alone do not say which recurrence, so name the variant from the reference before porting.");
        }

        var rwkv = c.Containing("time_mix_");
        if (rwkv.Count > 0)
            Add(f, "recurrent.rwkv_time_mix", "RWKV-style time-mix tensors.", Certainty.Hypothesis, c.TensorEvidence(rwkv, 4),
                "RWKV generations (v6/v7) differ in tensor sets and recurrence; confirm the version from the reference arch.");
    }

    private static SortedSet<int> Layers(IEnumerable<GgufTensorInfo> ts)
    {
        var set = new SortedSet<int>();
        foreach (var t in ts)
        {
            var n = ScoutAnalyzer.Normalize(t.Name);
            if (n.Stack == "blk") set.Add(n.Layer);
        }
        return set;
    }

    // ── multi-token / next-n prediction heads ─────────────────────────────────

    private static void Mtp(Ctx c, List<ScoutFinding> f)
    {
        var t = c.Tensors.Where(x => x.Name.Contains("nextn", StringComparison.Ordinal) || x.Name.Contains("mtp", StringComparison.Ordinal))
            .OrderBy(x => x.Name, StringComparer.Ordinal).ToArray();
        bool meta = c.Has("nextn_predict_layers");
        if (t.Length == 0 && !meta) return;
        var ev = new List<EvidenceItem>();
        if (meta) ev.Add(c.KeyEvidence("nextn_predict_layers"));
        ev.AddRange(c.TensorEvidence(t, 4));
        // One isolated tensor never proves a working head (plan §9): require several tensors AND the metadata count for Known.
        bool strong = meta && t.Length >= 2 && c.Number("nextn_predict_layers") is > 0;
        Add(f, "mtp.nextn_head", "Auxiliary next-n / multi-token-prediction head.", strong ? Certainty.Known : Certainty.Hypothesis, ev,
            strong ? "Structure observed. The runtime speculative path and the head's exact wiring are not implied."
                   : "Weak evidence (name-only or a single tensor). Not proof of a usable MTP head.");
    }

    // ── RoPE ──────────────────────────────────────────────────────────────────

    private static void Rope(Ctx c, List<ScoutFinding> f)
    {
        if (c.Has("rope.dimension_sections"))
            Add(f, "rope.multi_axis", "Multi-axis RoPE sections declared (M-RoPE style).", Certainty.Known, [c.KeyEvidence("rope.dimension_sections")],
                "Section order and which axes the text path uses are not implied by the key.");

        string? scaling = c.Text("rope.scaling.type");
        if (scaling is { Length: > 0 } and not "none")
            Add(f, "rope.scaling", $"RoPE scaling type '{scaling}'.", Certainty.Known,
                new[] { c.KeyEvidence("rope.scaling.type") }.Concat(new[] { "rope.scaling.factor", "rope.scaling.original_context_length", "rope.scaling.yarn_beta_fast" }
                    .Where(c.Has).Select(c.KeyEvidence)),
                "Confirm the scaling implementation (linear/yarn/longrope) and any attention-factor term against the reference.");

        double? dims = c.Number("rope.dimension_count");
        double? heads = c.Number("attention.head_count");
        double? emb = c.Number("embedding_length");
        double? keyLen = c.Number("attention.key_length");
        double? headDim = keyLen ?? (heads is > 0 && emb is > 0 ? emb / heads : null);
        if (dims is > 0 && headDim is > 0 && dims < headDim)
            Add(f, "rope.partial", $"RoPE covers {dims:0} of {headDim:0} head dimensions.", Certainty.Known,
                [c.KeyEvidence("rope.dimension_count"), keyLen is not null ? c.KeyEvidence("attention.key_length") : c.KeyEvidence("embedding_length")],
                "Partial RoPE: which dimensions rotate (first vs last) and pairing (neox vs interleaved) must come from the reference.");
    }

    // ── multimodal companion evidence ─────────────────────────────────────────

    private static void Projector(Ctx c, List<ScoutFinding> f)
    {
        if (string.Equals(c.Arch, "clip", StringComparison.Ordinal))
            Add(f, "multimodal.projector_file", "This file declares architecture 'clip': a multimodal projector (mmproj), not a text model.", Certainty.Known,
                [new EvidenceItem("metadata", "general.architecture", "clip")],
                "It only works beside a matching text GGUF; a projector alone is not a complete user-facing multimodal path.");

        var vis = c.Tensors.Where(t => t.Name.StartsWith("v.", StringComparison.Ordinal) || t.Name.StartsWith("mm.", StringComparison.Ordinal))
            .OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        var aud = c.Tensors.Where(t => t.Name.StartsWith("a.", StringComparison.Ordinal)).OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        if (vis.Length > 0 && !string.Equals(c.Arch, "clip", StringComparison.Ordinal))
            Add(f, "multimodal.vision_tensors_in_text_file", "Vision/projector tensor prefixes (v.*, mm.*) inside a non-projector file.", Certainty.Hypothesis,
                c.TensorEvidence(vis, 4), "Name-prefix evidence only; an embedded tower still needs its own parity-verified preprocessing.");
        if (aud.Length > 0)
            Add(f, "multimodal.audio_tensors", "Audio tower tensor prefix (a.*).", Certainty.Hypothesis, c.TensorEvidence(aud, 4),
                "Name-prefix evidence only; audio front-end (mel/feature extraction) is not implied.");
    }

    // ── exotic storage ────────────────────────────────────────────────────────

    private static void ExoticQuant(Ctx c, List<ScoutFinding> f)
    {
        var exotic = c.Tensors.Where(t => t.DType is DType.IQ1_S or DType.IQ1_M or DType.TQ1_0 or DType.TQ2_0 or DType.MXFP4 or DType.NVFP4 or DType.Q1_0 or DType.Q2_0)
            .OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        if (exotic.Length > 0)
            Add(f, "quant.low_bit_or_block_fp", "Low-bit or block-floating-point storage types in use: " +
                string.Join(", ", exotic.Select(t => t.DType.ToString()).Distinct().Order(StringComparer.Ordinal)), Certainty.Known,
                c.TensorEvidence(exotic, 4), "These pass the dtype gate but are the least-tested kernels; expect a parity run, not an assumption.");
    }
}
