namespace OpenTail.Stingray.Engine;

internal static class DeepSeek4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "deepseek4",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Structurally complete alpha forward pass that has never run on a real checkpoint; MQA K==V, output-side rope_ext_back, and CSA overlap-gather are unverified, and the Hadamard rotation is only partially applied (indexer K only, not indexer Q or main-attention Q/kv/output, confirmed against the real llama.cpp reference).",
    };
}
