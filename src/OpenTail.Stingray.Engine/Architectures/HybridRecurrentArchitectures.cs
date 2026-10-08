namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    // nemotron_h — NVIDIA Nemotron-H / Nemotron Nano v2 (llama.cpp nemotron-h.cpp): every layer is exactly one of
    // Mamba-2 (8 B/C groups), NoPE attention, or a non-gated ReLU² MLP, each with its own RMSNorm + residual.
    // ADMITTED 2026-09-27 on the shared Mamba-2 mixer (ForwardPass.Mamba2.cs, CPU). Evidence: nemotron-nano-12b-v2-vl
    // Q2_K text decoder, wikitext second-half [1024,+) PPL at -c 2048 6.6332 vs llama-perplexity --chunks 1 6.6338;
    // NemotronHParityTests teacher-forces llama-server's continuation and matches its token at all 22 positions
    // where llama.cpp's top-1 margin exceeds 1.5 nats (Q2_K forks free-running greedy at close pairs, e.g. '."' vs
    // '."' + newline at 0.22 nats in llama.cpp, 0.10 the other way in ours).
    public static readonly ArchitectureDescriptor Nemotronh = new()
    {
        Id = "nemotron_h",
        Traits = new() { Hybrid = OpenTail.Stingray.Core.HybridKind.Mamba2, SingleSublayerBlocks = true },
        // nemotron-h.cpp: non-gated ReLU^2 FFN (LLM_FFN_RELU_SQR); attention never applies RoPE.
        ApplyModelSemantics = ctx => ctx.Baseline with { UsesReluSquared = true, NoRopeLayerStep = 1 },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "NVIDIA Nemotron-H / Nemotron Nano v2 (`nemotron_h`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // lfm2 — Liquid LFM2 (llama.cpp lfm2.cpp): gated short-conv layers (in_proj -> b|c|x, causal depthwise conv of
    // b*x over shortconv.l_cache steps, gate by c, out_proj) + GQA attention with QK-norm and NeoX RoPE, SwiGLU FFN
    // on every layer; final norm stored as token_embd_norm. ADMITTED 2026-09-27 (ForwardPass.ShortConv.cs, CPU).
    // Evidence: LFM2-1.2B Q8_0 wikitext second-half [1024,+) PPL at -c 2048 10.9195 vs llama-perplexity --chunks 1
    // 10.9543 (0.3%); with BOS the next-token top-5 after "The capital of France is" matches llama-server to 0.14
    // nats; Lfm2ParityTests teacher-forces llama-server continuations (14 confident positions match). The model
    // degenerates without BOS, so admit-arch now prepends it like llama-server does. Licence: LFM Open License
    // v1.0 (free commercial use under $10M revenue); documented, not a support gate.
    public static readonly ArchitectureDescriptor Lfm2 = new()
    {
        Id = "lfm2",
        Traits = new() { Hybrid = OpenTail.Stingray.Core.HybridKind.ShortConv },
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Liquid LFM2 (`lfm2`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // lfm2moe -- Liquid LFM2-MoE (llama.cpp lfm2.cpp, MoE FFN: sigmoid gating + exp_probs_b selection bias, top-k
    // renormalised). ADMITTED 2026-10-01. LFM2-8B-A1B-Q4_K_M: batched prefill matches token-by-token bit-for-bit with
    // STINGRAY_PREFILL_ATTN_FLASH64=0 (the earlier "drift" was flash-64's online softmax, not a bug); wikitext
    // [512,1024) PPL at -c 1024 is 7.3425 per-token / 7.2882 batched-flash vs llama-perplexity 7.9130 +/- 1.09
    // (<0.6 SE); greedy "The capital of France is" continuation matches llama-completion.
    // docs/1-correctness/12-lfm2moe-batched-per-token-parity-plan.md.
    public static readonly ArchitectureDescriptor Lfm2moe = new()
    {
        Id = "lfm2moe",
        Traits = new() { Hybrid = OpenTail.Stingray.Core.HybridKind.ShortConv },
        UsesNeoxRope = true,
        ApplyModelSemantics = ctx => ctx.Baseline with { NormalizeMoeTopKWeights = true },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Liquid LFM2-MoE (`lfm2moe`)",
        EvidenceDoc = "docs/done/12-lfm2moe-batched-per-token-parity-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
