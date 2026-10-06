namespace OpenTail.Stingray.Engine;

internal static class PhiArchitectures
{
    // phimoe — was allowlisted without a receipt and produced word salad. Fixed 2026-09-26
    // (RMSNorm + bias instead of LayerNorm, output.bias, top-k weight renormalization,
    // LongRoPE short/long factors chosen by context size + rope.scaling.attn_factor).
    // Receipt: PhiMoeGreedyParityTests — Phi-3.5-MoE-instruct Q3_K_M vs llama-server, 24/24
    // exact greedy tokens at -c 4096 (short factors) and 32/32 on a 214-token prompt at
    // -c 8192 (long factors).
    public static readonly ArchitectureDescriptor Phi2 = new()
    {
        Id = "phi2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'phi2'.
    public static readonly ArchitectureDescriptor Phi3 = new()
    {
        Id = "phi3",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Phi-3 (`phi3`) on Vulkan",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'phi2'.
    public static readonly ArchitectureDescriptor Phimoe = new()
    {
        Id = "phimoe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Phi-3.5-MoE (`phimoe`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
