namespace OpenTail.Stingray.Engine;

internal static class Lfm3Architecture
{
    // lfm3 (LFM3-Omni-2B, text path). ADMITTED 2026-11-25, TEXT ONLY, CPU ONLY (CLAUDE.md rule 14). Evidence:
    // docs/2-coverage/2026-11-24-lfm3-omni-2b-port-plan.md (golden: 11/11 exact on a chat prompt, 10/10 exact on a
    // 394-token prompt past the 256-token sliding window; llama.cpp master 33aeb17).
    // Ported from llama.cpp src/models/lfm3.cpp and lfm-base.cpp (master, 2026-11-24; revision not pinned): the LFM2
    // short-conv hybrid (layers with head_count_kv 0 are short-conv, the rest attention), QK-norm, sandwich post-norms
    // (post_attention_norm / post_ffw_norm, both from the tensors), a per-head sigmoid attention output gate
    // (attn_gate: [embDim -> numHeads], applied to the attention output before wo), SwiGLU FFN (SiLU, parallel),
    // NEOX RoPE on every layer (llama-model.cpp rope-type group that includes LFM3), no embedding scale, and an
    // rope_freqs.weight mask on non-SWA layers. Omni inputs (image, audio) are not part of this port.
    // The pre-tokenizer is lfm2 (llama-vocab.cpp:2474), already implemented. No real-weight parity yet: no llama.cpp
    // build on this host knows lfm3.
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "lfm3",
        Traits = new() { Hybrid = OpenTail.Stingray.Core.HybridKind.ShortConv },
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "LFM3-Omni-2B text path (`lfm3`)",
        EvidenceDoc = "docs/2-coverage/2026-11-24-lfm3-omni-2b-port-plan.md",
        // GatedSwa supplies the attention output gate and the SWA layer pattern. LFM3 applies RoPE on every layer and does not
        // scale the embedding, so those two Muse/afmoe defaults are overridden. The gate is per head.
        ApplyModelSemantics = ctx => FamilyModelSemantics.GatedSwa(ctx, muse: false) with
        {
            RopeOnlySwaLayers = false,
            EmbeddingScale = 1f,
            AttentionOutputGatePerHead = true,
        },
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "LFM3's per-head attention output gate is supported by the CPU pass only.",
        SupportsContinuousBatching = false,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
