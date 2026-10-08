namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Vulkan;

/// <summary>
/// Execution parameters and hardware resources passed to an architecture's factory to construct its forward pass.
/// </summary>
public sealed class ArchitectureLoadContext
{
    public required ArchitectureProbe Probe { get; init; }

    /// <summary>Final, architecture-correct hyperparameters (after descriptor semantics). Factories read this, never the raw probe.</summary>
    public required ModelHyperparams Hyperparams { get; init; }
    public required ExecutionPlan Plan { get; init; }

    // Convenience forwarders reading from Plan
    public ForwardPassKind ForwardPassKind => Plan.ForwardPassKind;
    public ForwardPassDecision Decision => new(Plan.ForwardPassKind, Plan.BackendPlan?.Backend.ToString());
    public ForwardPassBackend Backend => Plan.BackendPlan?.Backend ?? ForwardPassBackend.Cpu;
    public int ContextSize => Plan.ContextSize;
    public int GpuLayers => Plan.GpuLayers;

    // Hardware layer placement computed by TierPlanner (for partial offloads: CudaHybrid, VulkanHybrid, VulkanLayerSplit)
    public LayerPlacement? Placement => Plan.Placement != null
        ? new LayerPlacement(
            Plan.Placement.GpuLayers,
            Plan.Placement.CpuLayers,
            Plan.Placement.GpuWeightBytes,
            0,
            Plan.ContextSize,
            Plan.Placement.ExpertCacheBudgetBytes,
            Plan.Placement.MoeRoutedExpertBytes,
            Plan.Placement.CpuWeightBytes)
        : null;

    // Configured runtime options forwarded from Plan
    public bool TurboQuant => Plan.State?.TurboQuant ?? false;
    public string TurboQuantMode => Plan.State?.TurboQuantMode ?? "manual";
    public int HeadDim => Plan.HeadDim;
    public TqQuantizer TqQuantizer => Plan.State?.TqQuantizer ?? TqQuantizer.LloydMax;
    public bool FlashAttention => Plan.State?.FlashAttention ?? true;
    public DType KvDType => Plan.State?.KvDType ?? DType.Float16;
    public long PrefillDequantCacheBytes => Plan.Memory?.PrefillDequantCacheBytes ?? 0;
    public bool PreferBatchingOverAutoSnapKv => Plan.Batching?.PreferBatchingOverAutoSnapKv ?? false;

    // Backend resources (orchestrator manages initialization and lifetime)
    public CpuBackend? CpuBackend { get; init; }
    public CudaBackend? CudaBackend { get; init; }
    public VulkanBackend? VulkanBackend { get; init; }

    // Baseline CPU dense pass (used by partial GPU passes: CudaHybridForwardPass / VulkanHybridForwardPass)
    public ForwardPass? CpuDensePass { get; init; }

    /// <summary>Instance-local execution settings resolved from <see cref="Plan"/>; pass to every pass constructor.</summary>
    public EngineSettings Settings => _settings ??= EngineSettings.FromPlan(Plan);
    private EngineSettings? _settings;

    // Track disposable resources created during forward pass setup
    public List<IDisposable> OwnedDisposables { get; } = [];

    public void TrackDisposable(IDisposable? disposable)
    {
        if (disposable is not null)
            OwnedDisposables.Add(disposable);
    }
}
