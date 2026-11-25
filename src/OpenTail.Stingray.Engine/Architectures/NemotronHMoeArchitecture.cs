namespace OpenTail.Stingray.Engine;

internal static class NemotronHMoeArchitecture
{
    // nemotron_h_moe (NVIDIA Nemotron-3.5-Lightning 30B-A3B). PORTED, NOT ADMITTED (CLAUDE.md rule 14).
    // Ported from llama.cpp nemotron-h.cpp (MoE branch) and nemotron-h-moe.cpp (reference: the C:\Git checkout at
    // c0bc8591e and llama.cpp master 33aeb17, both register nemotron_h_moe). 52 single-sublayer blocks: Mamba-2
    // (head_count_kv 0, feed_forward_length 0), attention (kv 2, no RoPE), or MoE (feed_forward_length > 0, no KV).
    // MoE: sigmoid routing with a selection bias (ffn_exp_probs_b), top-6 of 128, sum-normalised weights, routed
    // scale 2.5; experts are non-gated ReLU² (ffn_up_exps / ffn_down_exps, no gate tensor) and read/write a latent
    // width when ffn_latent_down / ffn_latent_up exist; one shared non-gated ReLU² expert (expert_shared_feed_forward_length).
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "nemotron_h_moe",
        Traits = new() { Hybrid = OpenTail.Stingray.Core.HybridKind.Mamba2, SingleSublayerBlocks = true },
        ApplyModelSemantics = ctx => ctx.Baseline with
        {
            UsesReluSquared = true,
            NoRopeLayerStep = 1,
            ExpertGatingFunc = 2,
            NormalizeMoeTopKWeights = true,
        },
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-11-25-nemotron-3.5-lightning-port-plan.md",
        RefusalReason = "Nemotron-3.5-Lightning 30B-A3B (nemotron_h_moe) is ported from llama.cpp nemotron-h.cpp (2026-11-25) " +
            "but has no real-weight parity receipt yet. Synthetic and structural checks only.",
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "Nemotron-3.5's non-gated ReLU² MoE (with latent projections) is supported by the CPU pass only.",
        SupportsContinuousBatching = false,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
