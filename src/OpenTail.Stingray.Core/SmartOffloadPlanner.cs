
namespace OpenTail.Stingray.Core;

/// <summary>
/// Execution strategy recommended by the planner.
/// </summary>
public enum ExecutionStrategy
{
    DiscreteGpuFull,
    DiscreteGpuPartial,
    MultiGpuPooled,
    IntegratedGpuHybrid,
    CpuSimdOnly
}

/// <summary>
/// Specific layer allocation for an individual GPU accelerator in a multi-GPU setup.
/// </summary>
public sealed record DeviceLayerAllocation
{
    [JsonPropertyName("device_index")]
    public int DeviceIndex { get; init; }

    [JsonPropertyName("device_name")]
    public string DeviceName { get; init; } = "";

    [JsonPropertyName("backend")]
    public string Backend { get; init; } = "vulkan";

    [JsonPropertyName("start_layer")]
    public int StartLayer { get; init; }

    [JsonPropertyName("end_layer")]
    public int EndLayer { get; init; }

    [JsonPropertyName("layer_count")]
    public int LayerCount { get; init; }

    [JsonPropertyName("allocated_bytes")]
    public long AllocatedBytes { get; init; }
}

/// <summary>
/// Optimal execution plan for a specific model and hardware configuration.
/// </summary>
public sealed record OffloadPlan
{
    public ExecutionStrategy Strategy { get; init; }
    public string RecommendedBackend { get; init; } = "auto";
    public int GpuLayersToOffload { get; init; }
    public int RecommendedThreads { get; init; }
    public string Description { get; init; } = "";
    public List<DeviceLayerAllocation> DeviceAllocations { get; init; } = new();
}

/// <summary>
/// Intelligent hardware-aware compute and layer offload planner.
/// Automatically balances workloads across single/multi discrete GPUs, APU/Vega iGPUs, and CPU SIMD engines.
/// </summary>
public static class SmartOffloadPlanner
{
    /// <summary>
    /// Computes the optimal execution plan for a model given total model size and layer count.
    /// Supports single-card full/partial offload and multi-GPU VRAM pooling.
    /// </summary>
    public static OffloadPlan Plan(
        long modelSizeBytes,
        int totalLayers,
        int contextTokens = 2048,
        HardwareTopology? topology = null)
    {
        var topo = topology ?? HardwareCapabilities.Current;
        var cpu = topo.Cpu;
        int physicalThreads = Math.Max(1, cpu.PhysicalCores);

        // Filter discrete GPUs vs integrated APUs
        var discreteGpus = topo.Gpus.Where(g => !g.IsIntegrated && g.TotalMemoryBytes > 0).ToList();
        var primaryGpu = topo.Gpus.Count > 0 ? topo.Gpus[0] : null;

        // 1. Pure CPU Fallback (No GPU detected)
        if (topo.Gpus.Count == 0 || primaryGpu == null || primaryGpu.TotalMemoryBytes <= 0)
        {
            return new OffloadPlan
            {
                Strategy = ExecutionStrategy.CpuSimdOnly,
                RecommendedBackend = "cpu",
                GpuLayersToOffload = 0,
                RecommendedThreads = physicalThreads,
                Description = $"Pure CPU execution using {(cpu.HasAvx512 ? "AVX-512" : cpu.HasAvx2 ? "AVX2" : "SIMD")} with {physicalThreads} worker threads."
            };
        }

        // 2. Integrated GPU / APU (e.g. AMD Vega, Intel Iris, Apple UMA)
        if (discreteGpus.Count == 0 && primaryGpu.IsIntegrated)
        {
            return new OffloadPlan
            {
                Strategy = ExecutionStrategy.IntegratedGpuHybrid,
                RecommendedBackend = "vulkan",
                GpuLayersToOffload = 0, // Keep memory-bound GEMM on CPU SIMD to avoid memory bus thrash
                RecommendedThreads = physicalThreads,
                Description = $"APU/iGPU Hybrid: CPU {(cpu.HasAvx512 ? "AVX-512" : "AVX2")} for memory-bound token generation + {primaryGpu.Name} for compute-heavy VAE/Vision kernels."
            };
        }

        long contextVramBuffer = Math.Max(256L * 1024 * 1024, (long)contextTokens * 4096 * 4); // KV cache buffer
        long bytesPerLayer = totalLayers > 0 ? modelSizeBytes / totalLayers : modelSizeBytes;

        // 3. Multi-GPU Pooling (>= 2 Discrete GPUs)
        if (discreteGpus.Count >= 2)
        {
            var usableVramPerGpu = discreteGpus.Select(g => Math.Max(0, g.AvailableMemoryBytes - contextVramBuffer)).ToList();
            long totalPooledVram = usableVramPerGpu.Sum();

            if (totalPooledVram > 0 && bytesPerLayer > 0)
            {
                int totalLayersToOffload = Math.Min(totalLayers, (int)(totalPooledVram / bytesPerLayer));
                var allocations = new List<DeviceLayerAllocation>();
                int currentStartLayer = 0;

                for (int i = 0; i < discreteGpus.Count; i++)
                {
                    var gpu = discreteGpus[i];
                    long gpuVram = usableVramPerGpu[i];

                    // Proportional layer split
                    int gpuLayerCount = (int)Math.Round((double)gpuVram / totalPooledVram * totalLayersToOffload);
                    gpuLayerCount = Math.Min(gpuLayerCount, totalLayersToOffload - currentStartLayer);

                    if (i == discreteGpus.Count - 1 && currentStartLayer + gpuLayerCount < totalLayersToOffload)
                    {
                        // Assign any remainder to the last GPU if VRAM permits
                        gpuLayerCount = totalLayersToOffload - currentStartLayer;
                    }

                    if (gpuLayerCount > 0)
                    {
                        allocations.Add(new DeviceLayerAllocation
                        {
                            DeviceIndex = gpu.Index,
                            DeviceName = gpu.Name,
                            Backend = gpu.Backend,
                            StartLayer = currentStartLayer,
                            EndLayer = currentStartLayer + gpuLayerCount - 1,
                            LayerCount = gpuLayerCount,
                            AllocatedBytes = gpuLayerCount * bytesPerLayer
                        });
                        currentStartLayer += gpuLayerCount;
                    }
                }

                if (allocations.Count >= 2)
                {
                    return new OffloadPlan
                    {
                        Strategy = ExecutionStrategy.MultiGpuPooled,
                        RecommendedBackend = allocations[0].Backend,
                        GpuLayersToOffload = totalLayersToOffload,
                        RecommendedThreads = physicalThreads,
                        DeviceAllocations = allocations,
                        Description = $"Multi-GPU VRAM Pooled across {allocations.Count} accelerators: {string.Join(", ", allocations.Select(a => $"{a.DeviceName} ({a.LayerCount} layers)"))}."
                    };
                }
            }
        }

        // 4. Single Discrete GPU
        var targetGpu = discreteGpus.Count > 0 ? discreteGpus[0] : primaryGpu;
        long availableVram = Math.Max(0, targetGpu.AvailableMemoryBytes - contextVramBuffer);

        // Full Offload
        if (availableVram >= (long)(modelSizeBytes * 1.15))
        {
            return new OffloadPlan
            {
                Strategy = ExecutionStrategy.DiscreteGpuFull,
                RecommendedBackend = targetGpu.Backend,
                GpuLayersToOffload = totalLayers,
                RecommendedThreads = physicalThreads,
                DeviceAllocations =
                [
                    new DeviceLayerAllocation
                    {
                        DeviceIndex = targetGpu.Index,
                        DeviceName = targetGpu.Name,
                        Backend = targetGpu.Backend,
                        StartLayer = 0,
                        EndLayer = totalLayers - 1,
                        LayerCount = totalLayers,
                        AllocatedBytes = modelSizeBytes
                    }
                ],
                Description = $"Full GPU acceleration on {targetGpu.Name} (all {totalLayers} layers offloaded to VRAM)."
            };
        }

        // Partial Offload (Tight VRAM)
        int layersToOffload = bytesPerLayer > 0
            ? Math.Clamp((int)(availableVram / bytesPerLayer), 0, totalLayers)
            : 0;

        if (layersToOffload > 0)
        {
            return new OffloadPlan
            {
                Strategy = ExecutionStrategy.DiscreteGpuPartial,
                RecommendedBackend = targetGpu.Backend,
                GpuLayersToOffload = layersToOffload,
                RecommendedThreads = physicalThreads,
                DeviceAllocations =
                [
                    new DeviceLayerAllocation
                    {
                        DeviceIndex = targetGpu.Index,
                        DeviceName = targetGpu.Name,
                        Backend = targetGpu.Backend,
                        StartLayer = 0,
                        EndLayer = layersToOffload - 1,
                        LayerCount = layersToOffload,
                        AllocatedBytes = layersToOffload * bytesPerLayer
                    }
                ],
                Description = $"Partial VRAM offload on {targetGpu.Name}: {layersToOffload}/{totalLayers} layers on GPU, remainder on CPU SIMD."
            };
        }

        // VRAM insufficient for even 1 layer -> Fall back to CPU SIMD
        return new OffloadPlan
        {
            Strategy = ExecutionStrategy.CpuSimdOnly,
            RecommendedBackend = "cpu",
            GpuLayersToOffload = 0,
            RecommendedThreads = physicalThreads,
            Description = $"VRAM budget ({targetGpu.AvailableMemoryBytes / (1024 * 1024)}MB) insufficient for model ({modelSizeBytes / (1024 * 1024)}MB); smoothly routing to CPU SIMD."
        };
    }
}
