using System.Text.Json.Serialization;

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

/// <summary>
/// Immutable execution plan model (§7.1 &amp; §5.3 of QoL plan):
/// Produced by <see cref="ExecutionPlanBuilder"/>, inspectable by CLI tools (<c>plan</c>, <c>run --explain</c>),
/// and consumed by <c>InferenceEngineLoader</c> to guarantee the executed plan matches the displayed plan.
/// </summary>
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
    [property: JsonPropertyName("decisions")] IReadOnlyList<ExecutionPlanDecisionDetail> Decisions,
    [property: JsonPropertyName("warnings")] IReadOnlyList<string> Warnings,
    [property: JsonPropertyName("request")] PlanRequest? Request = null,
    [property: JsonPropertyName("selected_backend")] string? SelectedBackend = null,
    [property: JsonPropertyName("cpu_layers")] int CpuLayers = 0,
    [property: JsonPropertyName("is_executable")] bool IsExecutable = true,
    [property: JsonPropertyName("plan_decisions")] IReadOnlyList<ExecutionPlanDecision>? PlanDecisions = null,
    [property: JsonPropertyName("effective_configuration")] EffectiveConfigurationSnapshot? EffectiveConfiguration = null,
    [property: JsonPropertyName("model_format")] ModelFormat ModelFormat = ModelFormat.Gguf
)
{
    public ExecutionPlan(
        int schemaVersion,
        PlanRequest request,
        string selectedBackend,
        int gpuLayers,
        int cpuLayers,
        int contextSize,
        bool isExecutable,
        IReadOnlyList<ExecutionPlanDecision> planDecisions,
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
            Decisions: new List<ExecutionPlanDecisionDetail>(),
            Warnings: new List<string>(),
            Request: request,
            SelectedBackend: selectedBackend,
            CpuLayers: cpuLayers,
            IsExecutable: isExecutable,
            PlanDecisions: planDecisions,
            EffectiveConfiguration: effectiveConfiguration,
            ModelFormat: ModelFormat.Gguf
        )
    {
    }

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
public partial class ExecutionPlanJsonContext : JsonSerializerContext
{
}
