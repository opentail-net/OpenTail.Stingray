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

[Flags]
public enum SupportedBackends
{
    Cpu = 1,
    Vulkan = 2,
    Cuda = 4,
    All = Cpu | Vulkan | Cuda,
}

public enum ForwardPassFamily
{
    Dense,
    HybridGdn,
    Rwkv,
    GptOss,
    DeepSeek2Mla,
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
/// <summary>The planner-facing capability view of an <see cref="ArchitectureDescriptor"/> (see <see cref="ArchitectureDescriptor.Capabilities"/>).</summary>
public sealed record ArchitectureCapabilities(
    SupportedBackends Backends,
    bool ContinuousBatching,
    bool HasBatchPredicate,
    bool ImageInput,
    bool AudioInput,
    ForwardPassFamily Family);

public sealed class ArchitectureDescriptor
{
    /// <summary>The GGUF <c>general.architecture</c> value.</summary>
    public required string Id { get; init; }

    /// <summary>Other GGUF architecture strings that map to this family (e.g. muse_glimmer).</summary>
    public IReadOnlyList<string> Aliases { get; init; } = [];

    public required AdmissionStatus Status { get; init; }

    /// <summary>Repo-relative doc holding the receipt (Admitted) or the missing work (NotAdmitted).</summary>
    public required string EvidenceDoc { get; init; }

    /// <summary>Substring identifying this family's row in the public status matrix.</summary>
    public string? StatusAnchor { get; init; }

    /// <summary>Why this admitted family has no dedicated row in the public status matrix.</summary>
    public string? StatusExemption { get; init; }

    /// <summary>Shown to the user when the status is not <see cref="AdmissionStatus.Admitted"/>.</summary>
    public string? RefusalReason { get; init; }

    public string? ExperimentalEnvVar { get; init; }

    /// <summary>Reasoning stays off unless a request opts in (e.g. Gemma 4).</summary>
    public bool ThinkingDefaultOff { get; init; }

    public FallbackChatFormat FallbackChat { get; init; } = FallbackChatFormat.ChatMl;

    /// <summary>
    /// Id of the <see cref="ChatProtocol"/> the engine API uses for this architecture (rendering, media placeholders, thinking
    /// markers, output parsing): <c>llama3</c>, <c>llama4</c>, <c>granite</c> or <c>gemma</c>. Null means the default
    /// <c>chatml</c> protocol. Distinct from <see cref="FallbackChat"/>, which only picks the prompt layout the CLI/server fall back
    /// to when a GGUF ships no usable Jinja template; the two differ today for the Gemma family and granitehybrid/granitemoe.
    /// </summary>
    public string? ChatProtocolId { get; init; }

    /// <summary>
    /// Structural facts the generic baseline parser needs while reading this architecture's file (hybrid layout, head-size source,
    /// multi-axis rope keys, ...). Null means none. Declared here so a new architecture needs no edit in Core; pinned to
    /// <see cref="OpenTail.Stingray.Core.ModelArchitectureTraits.Legacy"/> by a test while the by-name shim still exists.
    /// </summary>
    public OpenTail.Stingray.Core.ModelArchitectureTraits? Traits { get; init; }

    public ForwardPassFamily ForwardPassFamily { get; init; } = ForwardPassFamily.Dense;

    public SupportedBackends SupportedBackends { get; init; } = SupportedBackends.All;

    public string? BackendLimitation { get; init; }

    public string? DisplayName { get; init; }

    /// <summary>
    /// Recognizes a GGUF that declares NO architecture metadata (e.g. MiniMax-H3).
    /// </summary>
    public Func<ArchitectureProbe, bool>? DetectFromProbe { get; init; }

    /// <summary>
    /// Recognizes a model file labeled with a generic architecture (e.g. older Mistral models labeled as 'llama',
    /// or 'llama' with 'llama.expert_count' > 0). Receives declared architecture and probe.
    /// </summary>
    public Func<string, ArchitectureProbe, bool>? RecognizeRelabelledFile { get; init; }

    /// <summary>What <see cref="RecognizeRelabelledFile"/> accepts, for refusal/diagnostic messages.</summary>
    public string? RelabelledFileDescription { get; init; }

    /// <summary>
    /// RoPE pairing: true = NEOX (rotate-half), false = NORM (interleaved), mirroring llama.cpp llama_model_rope_type().
    /// A per-checkpoint <c>{arch}.rope.is_neox</c> metadata key still overrides.
    /// </summary>
    public bool UsesNeoxRope { get; init; }

    /// <summary>
    /// Optional load-time hook: generic baseline hyperparameters in, architecture-correct hyperparameters out.
    /// Pure and deterministic; must not touch execution plans or backends. Null means the baseline is already correct.
    /// </summary>
    public Func<OpenTail.Stingray.Core.ModelArchitectureSemanticsContext, ModelHyperparams>? ApplyModelSemantics { get; init; }

    public bool SupportsContinuousBatching { get; init; } = true;

    /// <summary>Optional fine-grained predicate checking (hyperparams, turboQuant) -> canBatch.</summary>
    public Func<ModelHyperparams, bool, bool>? CanBatchPredicate { get; init; }

    /// <summary>
    /// Checks whether continuous batching is supported for this model under the given hyperparams and quantization.
    /// </summary>
    public bool CanBatch(ModelHyperparams hp, bool turboQuant)
    {
        if (!SupportsContinuousBatching) return false;
        if (CanBatchPredicate is not null) return CanBatchPredicate(hp, turboQuant);
        return !hp.IsMoE && !turboQuant && hp.LayerHeadDim is null
            && !hp.AttentionOutputGate && !hp.InputEmbeddingRmsNorm;
    }

    public bool SupportsImageInput { get; init; }
    public bool SupportsAudioInput { get; init; }

    /// <summary>
    /// What this model CAN do, as one small read-only value. The planner reads this and nothing else from the descriptor:
    /// a descriptor says what the model supports ("can run on CUDA", "can batch", "takes image input"), never what to do on a
    /// given machine ("use CUDA"), which stays the planner's decision. A projection over the flat init fields so the 70-odd
    /// descriptors keep their declarative syntax (docs/2-coverage/2026-10-08-architecture-capabilities-plan.md, Phase 1).
    /// </summary>
    public ArchitectureCapabilities Capabilities => new(
        SupportedBackends, SupportsContinuousBatching, CanBatchPredicate is not null, SupportsImageInput, SupportsAudioInput, ForwardPassFamily);

    /// <summary>File-name patterns, tried in order, for finding this family's mmproj companion beside the model GGUF.</summary>
    public IReadOnlyList<string> ProjectorFileHints { get; init; } = [];

    /// <summary>
    /// Constructs the forward pass for this architecture according to the load context.
    /// </summary>
    public Func<ArchitectureLoadContext, OpenTail.Stingray.Core.IForwardPass>? CreateForwardPass { get; init; }

    /// <summary>Optional process-wide native tunables or pre-load hooks.</summary>
    public Action<ArchitectureLoadContext>? ApplyLoadSetup { get; init; }

    /// <summary>
    /// Universal construction pipeline: executes any pre-load setup hooks and invokes the factory.
    /// </summary>
    public OpenTail.Stingray.Core.IForwardPass ConstructForwardPass(ArchitectureLoadContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (CreateForwardPass is null)
            throw new InvalidOperationException($"Architecture '{Id}' has no CreateForwardPass factory.");
        ApplyLoadSetup?.Invoke(context);
        return CreateForwardPass(context);
    }

    /// <summary>Whether the engine may run this architecture right now.</summary>
    public bool IsUsable() => Status switch
    {
        AdmissionStatus.Admitted => true,
        AdmissionStatus.Experimental => Environment.GetEnvironmentVariable(ExperimentalEnvVar ?? "") == "1",
        _ => false,
    };

    public string GetRefusalMessage(string architecture) => Status switch
    {
        AdmissionStatus.NotAdmitted =>
            $"GGUF architecture '{architecture}' is not admitted by OpenTail.Stingray: {RefusalReason} " +
            $"(status {Status}; record: {EvidenceDoc}).",
        AdmissionStatus.Experimental =>
            $"GGUF architecture '{architecture}' is ported but not verified: {RefusalReason} " +
            $"Set {ExperimentalEnvVar}=1 to try it; outputs are unverified (record: {EvidenceDoc}).",
        _ => throw new InvalidOperationException($"Architecture '{Id}' is not refused."),
    };

    internal void Validate()
    {
        if (string.IsNullOrWhiteSpace(Id))
            throw new InvalidOperationException("Architecture descriptor has no Id.");
        if (string.IsNullOrWhiteSpace(EvidenceDoc))
            throw new InvalidOperationException($"Architecture '{Id}' has no EvidenceDoc.");
        if ((SupportedBackends & ~SupportedBackends.All) != 0 || SupportedBackends == 0)
            throw new InvalidOperationException($"Architecture '{Id}' has invalid SupportedBackends.");
        if (SupportedBackends != SupportedBackends.All && string.IsNullOrWhiteSpace(BackendLimitation))
            throw new InvalidOperationException($"Architecture '{Id}' restricts backends but gives no BackendLimitation.");
        if (SupportedBackends == SupportedBackends.All && BackendLimitation is not null)
            throw new InvalidOperationException($"Architecture '{Id}' gives BackendLimitation but supports every backend.");
        if (Status != AdmissionStatus.Admitted && string.IsNullOrWhiteSpace(RefusalReason))
            throw new InvalidOperationException($"Architecture '{Id}' is {Status} but gives no RefusalReason.");
        if (Status == AdmissionStatus.Experimental && string.IsNullOrWhiteSpace(ExperimentalEnvVar))
            throw new InvalidOperationException($"Architecture '{Id}' is Experimental but names no ExperimentalEnvVar.");
        if (Status == AdmissionStatus.Admitted)
        {
            bool hasAnchor = !string.IsNullOrWhiteSpace(StatusAnchor);
            bool hasExemption = !string.IsNullOrWhiteSpace(StatusExemption);
            if (hasAnchor == hasExemption)
                throw new InvalidOperationException(
                    $"Admitted architecture '{Id}' must have exactly one of StatusAnchor or StatusExemption.");
            if (CreateForwardPass == null)
                throw new InvalidOperationException($"Admitted architecture '{Id}' has no CreateForwardPass factory.");
            if (ForwardPassFamily != ForwardPassFamily.Dense && CreateForwardPass == CommonForwardPassFactory.CreateDense)
                throw new InvalidOperationException(
                    $"Admitted architecture '{Id}' has non-dense ForwardPassFamily '{ForwardPassFamily}' but uses CommonForwardPassFactory.CreateDense.");
            // The reverse direction (found 2026-10-08): qwen35/qwen35moe declared no family (so Dense) while using the dense factory
            // for a Gated-DeltaNet model, and nothing noticed. A hybrid factory now requires the hybrid family.
            if (CreateForwardPass == CommonForwardPassFactory.CreateHybridGdn && ForwardPassFamily != ForwardPassFamily.HybridGdn)
                throw new InvalidOperationException(
                    $"Admitted architecture '{Id}' uses CommonForwardPassFactory.CreateHybridGdn but declares ForwardPassFamily '{ForwardPassFamily}'; declare ForwardPassFamily.HybridGdn.");
        }
        else if (StatusAnchor is not null || StatusExemption is not null)
        {
            throw new InvalidOperationException(
                $"Architecture '{Id}' is {Status} and must not have StatusAnchor or StatusExemption.");
        }

        if (RecognizeRelabelledFile != null && string.IsNullOrWhiteSpace(RelabelledFileDescription))
        {
            throw new InvalidOperationException(
                $"Architecture '{Id}' recognises relabelled files but does not say which (RelabelledFileDescription is required).");
        }
    }
}
