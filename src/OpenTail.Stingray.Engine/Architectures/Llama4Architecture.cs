namespace OpenTail.Stingray.Engine;

internal static class Llama4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "llama4",
        ChatProtocolId = "llama4",
        // Meta reference: sigmoid gating weight-before-FFN; Llama4TextL2Norm QK-norm (unweighted, no attn_q_norm tensor);
        // NoPE every 4th layer (llama.cpp n_no_rope_layer_step = 4); chunked attention unless sliding_window is explicitly 0.
        ApplyModelSemantics = ctx =>
        {
            bool chunked = !(ctx.HasKey("attention.sliding_window") && ctx.Int("attention.sliding_window") == 0);
            return ctx.Baseline with
            {
                NoRopeLayerStep = 4,
                UseSigmoidGating = true,
                UseL2QkNorm = true,
                HasQkNorm = true,
                AttentionChunkSize = chunked ? 8192 : 0,
                AttnTempScale = chunked ? 0.1f : 0f,
            };
        },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Llama 4 Scout (`llama4`)",
        EvidenceDoc = "docs/STATUS.md",
        FallbackChat = FallbackChatFormat.Llama4,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
