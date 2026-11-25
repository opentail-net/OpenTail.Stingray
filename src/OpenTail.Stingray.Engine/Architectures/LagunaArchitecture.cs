namespace OpenTail.Stingray.Engine;

internal static class LagunaArchitecture
{
    // laguna (Poolside Laguna-XS-2.1 / S-2.1). PORTED, NOT ADMITTED (CLAUDE.md rule 14).
    // Ported from llama.cpp src/models/laguna.cpp (reference: the local checkout at OpenTail/examples/cpp/llama.cpp,
    // commit c0bc8591e, which also registers the architecture). Per-layer query head count from the array form of
    // attention.head_count (48 on full-attention layers, 64 on SWA layers; 8 KV heads everywhere); full attention on
    // layers i % 4 == 0 (no sliding_window_pattern key, so llama's dense-first period 4 applies); partial YaRN RoPE on
    // full layers (rope.dimension_count 64) and plain RoPE on SWA layers (rope.dimension_count_swa 128, freq base
    // 10000); QK-norm before RoPE; a per-head softplus attention output gate (attn_gate: [embDim -> numHeads_l]);
    // sigmoid-routed MoE with a score-correction bias, sum-normalised top-k, routed scale 2.5, one SiLU shared expert;
    // layer 0 dense (leading_dense_block_count 1); no embedding scale; NEOX RoPE.
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "laguna",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-11-25-laguna-xs-2.1-port-plan.md",
        RefusalReason = "Laguna-XS/S-2.1 (laguna) is ported from llama.cpp laguna.cpp but has no real-weight parity receipt yet; " +
            "synthetic and structural checks only.",
        UsesNeoxRope = true,
        ApplyModelSemantics = ApplyLaguna,
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "Laguna's per-head softplus attention output gate and per-layer query heads are supported by the CPU pass only.",
        SupportsContinuousBatching = false,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    private static ModelHyperparams ApplyLaguna(ModelArchitectureSemanticsContext ctx)
    {
        var hp = ctx.Baseline;
        int layers = hp.NumLayers;

        // Per-layer query heads (array form). NumHeads becomes the maximum, which sizes the scratch buffers.
        var heads = ctx.IntArray("attention.head_count");
        bool perLayerHeads = heads is { Count: > 0 };

        // Full attention on i % 4 == 0 (llama dense-first period 4); SWA on the other layers.
        var swa = new bool[layers];
        for (int i = 0; i < layers; i++) swa[i] = i % 4 != 0;

        // Full layers: rope.dimension_count (partial rotary). SWA layers: rope.dimension_count_swa (llama n_rot(il)).
        int fullRot = ctx.Int("rope.dimension_count");
        int swaRot = ctx.Int("rope.dimension_count_swa", fullRot);
        var dims = new int[layers];
        for (int i = 0; i < layers; i++) dims[i] = swa[i] ? swaRot : fullRot;

        return hp with
        {
            NumHeads = perLayerHeads ? heads!.Max() : hp.NumHeads,
            LayerNumHeads = perLayerHeads ? heads : null,
            AttentionOutputGate = true,
            AttentionOutputGatePerHead = true,
            AttentionOutputGateSoftplus = true,
            IsSwaLayer = swa,
            SlidingWindowSize = ctx.Int("attention.sliding_window"),
            RopeThetaSwa = ctx.Float("rope.freq_base_swa", 10_000f),
            RopeOnlySwaLayers = false,
            LayerRopeDim = dims,
            RopeWidthPerLayer = true,
            EmbeddingScale = 1f,
            NormalizeMoeTopKWeights = true,
        };
    }
}
