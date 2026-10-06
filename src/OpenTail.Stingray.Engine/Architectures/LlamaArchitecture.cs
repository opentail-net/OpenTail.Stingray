namespace OpenTail.Stingray.Engine;

internal static class LlamaArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "llama",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        FallbackChat = FallbackChatFormat.Llama3,
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
        DetectFromProbe = probe =>
            probe.HasTensor("token_embd.weight") &&
            probe.HasTensor("output.weight") &&
            probe.HasTensor("output_norm.weight") &&
            !probe.HasTensor("output_norm.bias") &&
            probe.HasTensor("blk.0.attn_q.weight") &&
            probe.HasTensor("blk.0.attn_k.weight") &&
            probe.HasTensor("blk.0.attn_v.weight") &&
            probe.HasTensor("blk.0.attn_output.weight") &&
            probe.HasTensor("blk.0.attn_norm.weight") &&
            !probe.HasTensor("blk.0.attn_norm.bias") &&
            probe.HasTensor("blk.0.ffn_gate.weight") &&
            probe.HasTensor("blk.0.ffn_up.weight") &&
            probe.HasTensor("blk.0.ffn_down.weight") &&
            probe.HasTensor("blk.0.ffn_norm.weight") &&
            !probe.HasTensor("blk.0.ffn_norm.bias") &&
            !probe.HasTensor("blk.0.attn_q.bias") &&
            !probe.HasTensor("blk.0.attn_q_norm.weight") &&
            !probe.HasTensor("blk.0.post_attention_norm.weight") &&
            !probe.HasTensor("blk.0.ssm_in.weight") &&
            !probe.HasTensor("blk.0.attn_kv_a_mqa.weight"),
    };
}
