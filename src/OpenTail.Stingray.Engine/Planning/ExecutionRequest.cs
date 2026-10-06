#nullable enable

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// Unified input request representing all execution preferences and pins passed to <see cref="ExecutionPlanner"/>.
/// Used identically across CLI, server, and embedded callers.
/// </summary>
public sealed record ExecutionRequest
{
    public string Goal { get; init; } = "balanced";
    public string? PinnedBackend { get; init; }
    public int? PinnedGpuLayers { get; init; }
    public int? PinnedContextSize { get; init; }
    public string? PinnedKvDtype { get; init; }
    public bool TurboQuant { get; init; }
    public string TurboQuantMode { get; init; } = "auto";
    public int TurboQuantBits { get; init; } = 3;
    public int TurboQuantFp32Window { get; init; } = 256;
    public bool FlashAttention { get; init; } = true;
    public float? RopeFrequencyBase { get; init; }
    public float? RopeFrequencyScale { get; init; }
    public int ThreadCount { get; init; } = 0;
    public BatchingMode BatchingMode { get; init; } = BatchingMode.Sequential;
    public int MaxBatchSize { get; init; } = 1;
    public SpeculationMode SpeculationMode { get; init; } = SpeculationMode.None;
    public string? DraftModelPath { get; init; }
    public string? DSparkModelPath { get; init; }
    public bool SnapKvEnabled { get; init; }
    public int SnapKvBudget { get; init; }
    public bool AllowUnverifiedArchitecture { get; init; }
    public bool NoGpuProbe { get; init; }
}
