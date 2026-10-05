namespace OpenTail.Stingray.Engine;

internal static class DeepSeek4Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "deepseek4",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Structurally complete alpha forward pass that has never run on a real checkpoint; MQA K==V, output-side rope_ext_back, CSA overlap-gather and the Hadamard rotation are unverified or unimplemented.",
    };
}
