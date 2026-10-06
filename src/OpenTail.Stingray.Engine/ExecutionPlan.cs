#nullable enable

using System.Collections.Immutable;
using System.Text.Json.Serialization;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Packaging;

namespace OpenTail.Stingray.Engine;

[JsonConverter(typeof(JsonStringEnumConverter<PlanDecisionDisposition>))]
public enum PlanDecisionDisposition
{
    Selected,
    Rejected,
    Ignored,
    Conditional
}

[JsonConverter(typeof(JsonStringEnumConverter<PlanDiagnosticSeverity>))]
public enum PlanDiagnosticSeverity
{
    Info,
    Warning,
    Error
}

[JsonConverter(typeof(JsonStringEnumConverter<BatchingMode>))]
public enum BatchingMode
{
    Sequential,
    Continuous
}

[JsonConverter(typeof(JsonStringEnumConverter<SpeculationMode>))]
public enum SpeculationMode
{
    None,
    DraftModel,
    PromptLookup,
    DSpark
}

/// <summary>Stable decision-code constants shared across CLI, server, and loaders.</summary>
public static class ExecutionPlanDecisionCodes
{
    public const string ModelCompatibility = "model.compatibility";
    public const string BackendSelection = "backend.selection";
    public const string KvTurboQuant = "kv.turbo_quant";
    public const string SnapKv = "kv.snapkv";
    public const string Speculation = "speculation.selection";
    public const string Batching = "batching.selection";
    public const string ToolGrammar = "tool_grammar.selection";
    public const string Configuration = "configuration";
}

public sealed record BackendPlan(
    ForwardPassBackend Backend,
    string DeviceName,
    int DeviceIndex,
    bool CudaAvailable,
    bool VulkanAvailable,
    int ThreadCount);

public sealed record PlacementPlan(
    int GpuLayers,
    int CpuLayers,
    int TotalLayers,
    long GpuWeightBytes,
    long CpuWeightBytes,
    long ExpertCacheBudgetBytes = 0,
    long MoeRoutedExpertBytes = 0,
    bool FixedWeightsOnCpu = false);

public sealed record StatePlan(
    string StateModel,
    int ContextLength,
    DType KvDType,
    bool TurboQuant = false,
    string TurboQuantMode = "none",
    int TurboQuantBits = 3,
    int TurboQuantFp32Window = 256,
    bool SnapKvEnabled = false,
    int SnapKvBudget = 0);

public sealed record BatchingPlan(
    BatchingMode Mode,
    int MaxBatchSize,
    int MaxConcurrentSessions);

public sealed record SpeculationPlan(
    SpeculationMode Mode,
    string? DraftModelPath = null,
    string? DSparkModelPath = null,
    int SpeculativeTokens = 0);

public sealed record ModalityPlan(
    bool SupportsVision,
    bool SupportsEmbeddingInput,
    string? MmprojPath = null);

public sealed record MemoryPlan(
    double EstimatedVramMb,
    double EstimatedRamMb,
    long ScratchBytes = 0,
    long PeakAllocationBytes = 0);

public sealed record PlanProvenance(
    string CreatedAtUtc,
    string PlannerVersion,
    string PrimaryModelPath,
    string TargetArchitecture,
    string Goal);

/// <summary>
/// Immutable execution plan model (Schema v2, §7.1 &amp; §5.3):
/// Serves as the authoritative, deterministic execution contract consumed by RuntimeInstance
/// and loaders to guarantee the executed plan matches the planned configuration without rediscovery.
/// </summary>
[method: JsonConstructor]
public sealed record ExecutionPlan(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("model_path")] string ModelPath,
    [property: JsonPropertyName("goal")] string Goal,
    [property: JsonPropertyName("backend")] string Backend,
    [property: JsonPropertyName("gpu_layers")] int GpuLayers,
    [property: JsonPropertyName("total_layers")] int TotalLayers,
    [property: JsonPropertyName("context_size")] int ContextSize,
    [property: JsonPropertyName("kv_dtype")] string KvDtype,
    [property: JsonPropertyName("estimated_vram_mb")] double EstimatedVramMb,
    [property: JsonPropertyName("estimated_ram_mb")] double EstimatedRamMb,
    [property: JsonPropertyName("decisions")] ImmutableArray<ExecutionPlanDecisionDetail> Decisions,
    [property: JsonPropertyName("warnings")] ImmutableArray<string> Warnings,
    [property: JsonPropertyName("request")] PlanRequest? Request = null,
    [property: JsonPropertyName("selected_backend")] string? SelectedBackend = null,
    [property: JsonPropertyName("cpu_layers")] int CpuLayers = 0,
    [property: JsonPropertyName("is_executable")] bool IsExecutable = true,
    [property: JsonPropertyName("plan_decisions")] ImmutableArray<ExecutionPlanDecision>? PlanDecisions = null,
    [property: JsonPropertyName("effective_configuration")] EffectiveConfigurationSnapshot? EffectiveConfiguration = null,
    [property: JsonPropertyName("model_format")] ModelFormat ModelFormat = ModelFormat.Gguf,
    [property: JsonPropertyName("package_identity")] ModelPackageIdentity? PackageIdentity = null,
    [property: JsonPropertyName("forward_pass_kind")] ForwardPassKind ForwardPassKind = ForwardPassKind.CpuDense,
    [property: JsonPropertyName("backend_plan")] BackendPlan? BackendPlan = null,
    [property: JsonPropertyName("placement_plan")] PlacementPlan? Placement = null,
    [property: JsonPropertyName("state_plan")] StatePlan? State = null,
    [property: JsonPropertyName("batching_plan")] BatchingPlan? Batching = null,
    [property: JsonPropertyName("speculation_plan")] SpeculationPlan? Speculation = null,
    [property: JsonPropertyName("modality_plan")] ModalityPlan? Modality = null,
    [property: JsonPropertyName("memory_plan")] MemoryPlan? Memory = null,
    [property: JsonPropertyName("provenance")] PlanProvenance? Provenance = null
)
{
    public ExecutionPlan(
        int SchemaVersion,
        string ModelPath,
        string Goal,
        string Backend,
        int GpuLayers,
        int TotalLayers,
        int ContextSize,
        string KvDtype,
        double EstimatedVramMb,
        double EstimatedRamMb,
        IEnumerable<ExecutionPlanDecisionDetail>? Decisions,
        IEnumerable<string>? Warnings,
        PlanRequest? Request = null,
        string? SelectedBackend = null,
        int CpuLayers = 0,
        bool IsExecutable = true,
        IEnumerable<ExecutionPlanDecision>? PlanDecisions = null,
        EffectiveConfigurationSnapshot? EffectiveConfiguration = null,
        ModelFormat ModelFormat = ModelFormat.Gguf,
        ModelPackageIdentity? PackageIdentity = null,
        ForwardPassKind ForwardPassKind = ForwardPassKind.CpuDense,
        BackendPlan? BackendPlan = null,
        PlacementPlan? Placement = null,
        StatePlan? State = null,
        BatchingPlan? Batching = null,
        SpeculationPlan? Speculation = null,
        ModalityPlan? Modality = null,
        MemoryPlan? Memory = null,
        PlanProvenance? Provenance = null)
        : this(
            SchemaVersion,
            ModelPath,
            Goal,
            Backend,
            GpuLayers,
            TotalLayers,
            ContextSize,
            KvDtype,
            EstimatedVramMb,
            EstimatedRamMb,
            Decisions?.ToImmutableArray() ?? [],
            Warnings?.ToImmutableArray() ?? [],
            Request,
            SelectedBackend ?? Backend,
            CpuLayers,
            IsExecutable,
            PlanDecisions != null ? PlanDecisions.ToImmutableArray() : null,
            EffectiveConfiguration,
            ModelFormat,
            PackageIdentity,
            ForwardPassKind,
            BackendPlan,
            Placement,
            State,
            Batching,
            Speculation,
            Modality,
            Memory,
            Provenance)
    {
    }

    public ExecutionPlan(
        int schemaVersion,
        PlanRequest request,
        string selectedBackend,
        int gpuLayers,
        int cpuLayers,
        int contextSize,
        bool isExecutable,
        IEnumerable<ExecutionPlanDecision> planDecisions,
        EffectiveConfigurationSnapshot effectiveConfiguration)
        : this(
            SchemaVersion: schemaVersion,
            ModelPath: request.Target,
            Goal: "auto",
            Backend: selectedBackend,
            GpuLayers: gpuLayers,
            TotalLayers: gpuLayers + cpuLayers,
            ContextSize: contextSize,
            KvDtype: request.KvType,
            EstimatedVramMb: 0,
            EstimatedRamMb: 0,
            Decisions: ImmutableArray<ExecutionPlanDecisionDetail>.Empty,
            Warnings: ImmutableArray<string>.Empty,
            Request: request,
            SelectedBackend: selectedBackend,
            CpuLayers: cpuLayers,
            IsExecutable: isExecutable,
            PlanDecisions: planDecisions != null ? planDecisions.ToImmutableArray() : null,
            EffectiveConfiguration: effectiveConfiguration,
            ModelFormat: ModelFormat.Gguf
        )
    {
    }

    /// <summary>
    /// Creates a schema v2 execution plan with full strongly typed sub-plans and defensive collection copies.
    /// </summary>
    public static ExecutionPlan CreateV2(
        ModelPackageIdentity packageIdentity,
        ForwardPassKind forwardPassKind,
        BackendPlan backendPlan,
        PlacementPlan placement,
        StatePlan state,
        BatchingPlan batching,
        SpeculationPlan speculation,
        ModalityPlan modality,
        MemoryPlan memory,
        PlanProvenance provenance,
        ImmutableArray<ExecutionPlanDecisionDetail> decisions,
        ImmutableArray<string> warnings,
        PlanRequest? request = null,
        ImmutableArray<ExecutionPlanDecision>? planDecisions = null,
        EffectiveConfigurationSnapshot? effectiveConfiguration = null,
        ModelFormat modelFormat = ModelFormat.Gguf,
        bool isExecutable = true)
    {
        string backendStr = backendPlan.Backend.ToString().ToLowerInvariant();

        return new ExecutionPlan(
            SchemaVersion: 2,
            ModelPath: provenance.PrimaryModelPath,
            Goal: provenance.Goal,
            Backend: backendStr,
            GpuLayers: placement.GpuLayers,
            TotalLayers: placement.TotalLayers,
            ContextSize: state.ContextLength,
            KvDtype: state.KvDType.ToString().ToLowerInvariant(),
            EstimatedVramMb: memory.EstimatedVramMb,
            EstimatedRamMb: memory.EstimatedRamMb,
            Decisions: decisions.IsDefault ? [] : decisions,
            Warnings: warnings.IsDefault ? [] : warnings,
            Request: request,
            SelectedBackend: backendStr,
            CpuLayers: placement.CpuLayers,
            IsExecutable: isExecutable,
            PlanDecisions: planDecisions,
            EffectiveConfiguration: effectiveConfiguration,
            ModelFormat: modelFormat,
            PackageIdentity: packageIdentity,
            ForwardPassKind: forwardPassKind,
            BackendPlan: backendPlan,
            Placement: placement,
            State: state,
            Batching: batching,
            Speculation: speculation,
            Modality: modality,
            Memory: memory,
            Provenance: provenance
        );
    }

    /// <summary>
    /// Translates a plan's <see cref="KvDtype"/> into the vocabulary <c>STINGRAY_KV_DTYPE</c> (and <c>--kv-type</c>) accepts.
    /// </summary>
    public static string KvDtypeToEnvValue(string planKvDtype) => planKvDtype.Trim().ToLowerInvariant() switch
    {
        "float32" or "f32" or "fp32" => "fp32",
        "bfloat16" or "bf16" => "bf16",
        "q8_0" or "q8" => "q8_0",
        _ => planKvDtype,
    };

    public string CompactSummary()
    {
        string backendUpper = Backend.ToUpperInvariant();
        string placement = TotalLayers > 0 && GpuLayers >= TotalLayers
            ? $"full {backendUpper} GPU weights ({GpuLayers}/{TotalLayers} layers)"
            : GpuLayers > 0
                ? $"hybrid GPU/CPU ({GpuLayers}/{TotalLayers} layers on {backendUpper})"
                : "CPU only";

        return $"[ExecutionPlan] Model: {System.IO.Path.GetFileName(ModelPath)} (ctx {ContextSize})\n" +
               $"[ExecutionPlan] Placement: {placement}, KV: {KvDtype}\n" +
               $"[ExecutionPlan] Goal: {Goal} (est. VRAM: {EstimatedVramMb:F1} MiB, RAM: {EstimatedRamMb:F1} MiB)";
    }
}

public sealed record ExecutionPlanDecisionDetail(
    [property: JsonPropertyName("code")] string Code,
    [property: JsonPropertyName("selected_value")] string SelectedValue,
    [property: JsonPropertyName("reason")] string Reason,
    [property: JsonPropertyName("source")] string Source
);

/// <summary>One stable, machine-readable explanation for a consequential planning decision.</summary>
public sealed record ExecutionPlanDecision(
    string Code,
    PlanDecisionDisposition Disposition,
    PlanDiagnosticSeverity Severity,
    string Reason);

public sealed record PlanRequest(
    string Target,
    string Backend,
    int GpuLayers,
    int ContextSize,
    bool TurboQuant,
    string KvType,
    string SpecType,
    int MaxBatchSize,
    bool ToolGrammar);

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(ExecutionPlan))]
[JsonSerializable(typeof(BackendPlan))]
[JsonSerializable(typeof(PlacementPlan))]
[JsonSerializable(typeof(StatePlan))]
[JsonSerializable(typeof(BatchingPlan))]
[JsonSerializable(typeof(SpeculationPlan))]
[JsonSerializable(typeof(ModalityPlan))]
[JsonSerializable(typeof(MemoryPlan))]
[JsonSerializable(typeof(PlanProvenance))]
[JsonSerializable(typeof(ModelPackageIdentity))]
[JsonSerializable(typeof(ModelPackageComponentIdentity))]
[JsonSerializable(typeof(ExecutionPlanDecisionDetail))]
[JsonSerializable(typeof(ExecutionPlanDecision))]
[JsonSerializable(typeof(PlanRequest))]
public partial class ExecutionPlanJsonContext : JsonSerializerContext
{
}
