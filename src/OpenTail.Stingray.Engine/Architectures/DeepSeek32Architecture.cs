namespace OpenTail.Stingray.Engine;

internal static class DeepSeek32Architecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "deepseek32",
        Status = AdmissionStatus.NotAdmitted,
        EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        // Ported, not verified (CLAUDE.md rule 14); the table in EvidenceDoc is the internal record.
        RefusalReason = "Structurally complete alpha MLA + DSA forward pass, never run on a real DeepSeek-V3.2 GGUF; Hadamard rotation on the indexer and MTP are not implemented. Also carries the same unguarded Float32-cast risk confirmed on a real checkpoint for qwen4exp/deepseek4/deepseek41 (2026-11-07) -- this file's norm/scale/conv weight casts are scattered across many individual call sites rather than one helper, so the same fix wasn't ported here yet.",
    };
}
