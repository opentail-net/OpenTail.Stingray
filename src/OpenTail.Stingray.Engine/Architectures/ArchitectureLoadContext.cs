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
    public required ForwardPassDecision Decision { get; init; }
    public required ForwardPassBackend Backend { get; init; }
    public required int ContextSize { get; init; }
    public required int GpuLayers { get; init; }

    // Hardware layer placement computed by TierPlanner (for partial offloads: CudaHybrid, VulkanHybrid, VulkanLayerSplit)
    public LayerPlacement? Placement { get; init; }

    // Configured runtime options
    public bool TurboQuant { get; init; }
    public string TurboQuantMode { get; init; } = "auto";
    public int HeadDim { get; init; }
    public TqQuantizer TqQuantizer { get; init; } = TqQuantizer.LloydMax;
    public bool FlashAttention { get; init; } = true;
    public DType KvDType { get; init; } = DType.Float16;
    public long PrefillDequantCacheBytes { get; init; }
    public bool PreferBatchingOverAutoSnapKv { get; init; }

    // Backend resources (orchestrator manages initialization and lifetime)
    public CpuBackend? CpuBackend { get; init; }
    public CudaBackend? CudaBackend { get; init; }
    public VulkanBackend? VulkanBackend { get; init; }

    // Baseline CPU dense pass (used by partial GPU passes: CudaHybridForwardPass / VulkanHybridForwardPass)
    public ForwardPass? CpuDensePass { get; init; }

    // Track disposable resources created during forward pass setup
    public List<IDisposable> OwnedDisposables { get; } = [];

    public void TrackDisposable(IDisposable? disposable)
    {
        if (disposable is not null)
            OwnedDisposables.Add(disposable);
    }
}
