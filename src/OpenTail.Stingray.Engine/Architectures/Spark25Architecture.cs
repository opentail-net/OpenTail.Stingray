namespace OpenTail.Stingray.Engine;

internal static class Spark25Architecture
{
    // spark2_5 (Spark-X2.5 / Spark-X2.6 4B). PORTED, NOT ADMITTED (CLAUDE.md rule 14).
    // Ported from llama.cpp src/models/spark2-5.cpp (master, 2026-11-24; revision not pinned): dense RMS-norm
    // attention, NEOX RoPE (llama-model.cpp llama_model_rope_type), per-layer RoPE width (n_rot_swa on SWA layers,
    // n_rot_full otherwise, llama-hparams.cpp n_rot), sliding-window pattern from the GGUF, a per-head sigmoid
    // attention output gate (attn_gate: [embDim -> numHeads]), GELU-gated FFN (approximate GELU), no embedding
    // scale, no final logit softcap. Tokenizer pre-type "spark2_5" (llama-vocab.cpp). No real-weight parity: no
    // llama.cpp build on this host knows spark2_5.
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "spark2_5",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-11-24-spark-x2.6-port-plan.md",
        RefusalReason = "Spark-X2.5/2.6 (spark2_5) is ported from llama.cpp spark2-5.cpp (2026-11-24) but has no real-weight parity receipt yet; " +
            "no llama.cpp build on this host knows the architecture. Synthetic tests only.",
        UsesNeoxRope = true,
        ApplyModelSemantics = ApplySpark,
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "Spark-X's per-head attention output gate is supported by the CPU pass only.",
        SupportsContinuousBatching = false,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Muse-Glimmer and afmoe share GatedSwa, which also sets RoPE-only-on-SWA layers and a sqrt(embDim)
    // embedding scale, and sets one uniform RoPE width. Spark applies RoPE on every layer, does not scale
    // embeddings, uses GELU, and has two RoPE widths (rope.dimension_count for full layers, rope.dimension_count_swa
    // for SWA layers; llama-hparams.cpp n_rot(il)).
    private static ModelHyperparams ApplySpark(ModelArchitectureSemanticsContext ctx)
    {
        var hp = FamilyModelSemantics.GatedSwa(ctx, muse: false) with
        {
            RopeOnlySwaLayers = false,
            EmbeddingScale = 1f,
            FfnActivation = FfnActivation.GeluApprox,
            AttentionOutputGatePerHead = true,
        };
        int fullRot = ctx.Int("rope.dimension_count");
        int swaRot = ctx.Int("rope.dimension_count_swa", fullRot);
        var swa = hp.IsSwaLayer;
        var dims = new int[hp.NumLayers];
        for (int i = 0; i < dims.Length; i++)
            dims[i] = swa is not null && swa[i] ? swaRot : fullRot;
        return hp with { LayerRopeDim = dims, RopeWidthPerLayer = true };
    }
}
