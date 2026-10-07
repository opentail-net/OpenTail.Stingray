#nullable enable

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// Unified input request representing all execution preferences and pins passed to <see cref="ExecutionPlanner"/>.
/// Used identically across CLI, server, and embedded callers.
/// </summary>
public sealed record ExecutionRequest
{
    public string? ModelPath { get; init; }
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
    public string? MmprojPath { get; init; }
    public bool EnableSessions { get; init; }
    public bool SnapKvEnabled { get; init; }
    public int SnapKvBudget { get; init; }
    public bool AllowUnverifiedArchitecture { get; init; }
    public bool NoGpuProbe { get; init; }

    /// <summary>Concrete GPU device index to use (CUDA and Vulkan); -1 = driver default selection.</summary>
    public int DeviceIndex { get; init; } = -1;

    /// <summary>User pin for the DSpark draft head location: null/"auto", "gpu", "cpu" or "off".</summary>
    public string? DSparkPlace { get; init; }

    /// <summary>
    /// True when a configured DSpark head is mandatory: a placement of Off is then a planning failure. False (default)
    /// means optional: Off records a warning and the plan falls back to normal generation. Same request, same plan,
    /// whichever frontend asked.
    /// </summary>
    public bool DSparkRequired { get; init; }

    // MoE / expert execution (null = unspecified; engine default or inherited environment applies).
    public bool? CpuMoe { get; init; }
    public bool? GpuMoePrefill { get; init; }
    public int? MoeWarmPin { get; init; }
    public int? MoeWarmPinAfter { get; init; }
    public bool? MoePredictPrefetch { get; init; }
    public string? ExpertStatsPath { get; init; }

    // KV store layout, speculation, prefill and SnapKV (null = unspecified: the planner records the engine default).
    public string? KvStore { get; init; }
    public int? KvBf16MinTokens { get; init; }
    public bool? MtpEnabled { get; init; }
    public bool? BatchVerify { get; init; }
    public bool? SpecBatchVerify { get; init; }
    public int? MtpDraftN { get; init; }
    public float? MtpMinAccept { get; init; }
    public int? MtpBatchMax { get; init; }
    public bool? MtpBatchedMoeVerify { get; init; }
    public int? DSparkVerifyLen { get; init; }
    public float? DSparkMinConfidence { get; init; }
    public int? PrefillChunkTokens { get; init; }
    public int? PrefixSlots { get; init; }
    public int? PrefixScratchTokens { get; init; }
    public string? HybridCpuPrefill { get; init; }
    public string? CudaHybridCpuPrefill { get; init; }
    public string? GpuCpuPrefill { get; init; }
    public int? HybridCpuPrefillMinTokens { get; init; }
    public int? HybridCpuPrefillKvBudgetMb { get; init; }
    public bool? HybridCpuPrefillWarmExperts { get; init; }
    public int? SnapKvWindow { get; init; }
    public int? SnapKvRecency { get; init; }
    public bool SnapKvBudgetExplicit { get; init; }
}
