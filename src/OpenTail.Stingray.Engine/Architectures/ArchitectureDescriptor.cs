namespace OpenTail.Stingray.Engine;

/// <summary>Whether a GGUF architecture may be served.</summary>
public enum AdmissionStatus
{
    /// <summary>Verified against an independent reference; evidence is required.</summary>
    Admitted,
    /// <summary>Ported but not verified (CLAUDE.md rule 14): refused with the descriptor's reason.</summary>
    NotAdmitted,
    /// <summary>Loadable only when <see cref="ArchitectureDescriptor.ExperimentalEnvVar"/> is "1".</summary>
    Experimental,
}

/// <summary>Prompt layout used when the GGUF ships no usable Jinja chat template.</summary>
public enum FallbackChatFormat
{
    ChatMl,
    Llama3,
    Llama4,
    Granite,
}

/// <summary>
/// One model family's declaration: admission status plus the per-family behaviour that used to be
/// spread over CLI/server <c>arch == "..."</c> checks. Add a family by adding one
/// <c>*Architecture.cs</c> beside this file and one line in <see cref="BuiltInArchitectures"/>.
/// </summary>
public sealed class ArchitectureDescriptor
{
    /// <summary>The GGUF <c>general.architecture</c> value.</summary>
    public required string Id { get; init; }

    public required AdmissionStatus Status { get; init; }

    /// <summary>Repo-relative doc holding the receipt (Admitted) or the missing work (NotAdmitted).</summary>
    public required string EvidenceDoc { get; init; }

    /// <summary>Shown to the user when the status is not <see cref="AdmissionStatus.Admitted"/>.</summary>
    public string? RefusalReason { get; init; }

    public string? ExperimentalEnvVar { get; init; }

    /// <summary>Reasoning stays off unless a request opts in (e.g. Gemma 4).</summary>
    public bool ThinkingDefaultOff { get; init; }

    public FallbackChatFormat FallbackChat { get; init; } = FallbackChatFormat.ChatMl;

    /// <summary>Whether the engine may run this architecture right now.</summary>
    public bool IsUsable() => Status switch
    {
        AdmissionStatus.Admitted => true,
        AdmissionStatus.Experimental => Environment.GetEnvironmentVariable(ExperimentalEnvVar ?? "") == "1",
        _ => false,
    };

    internal void Validate()
    {
        if (string.IsNullOrWhiteSpace(Id))
            throw new InvalidOperationException("Architecture descriptor has no Id.");
        if (string.IsNullOrWhiteSpace(EvidenceDoc))
            throw new InvalidOperationException($"Architecture '{Id}' has no EvidenceDoc.");
        if (Status != AdmissionStatus.Admitted && string.IsNullOrWhiteSpace(RefusalReason))
            throw new InvalidOperationException($"Architecture '{Id}' is {Status} but gives no RefusalReason.");
        if (Status == AdmissionStatus.Experimental && string.IsNullOrWhiteSpace(ExperimentalEnvVar))
            throw new InvalidOperationException($"Architecture '{Id}' is Experimental but names no ExperimentalEnvVar.");
    }
}
