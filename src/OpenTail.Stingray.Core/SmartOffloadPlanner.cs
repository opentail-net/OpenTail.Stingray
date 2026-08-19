namespace OpenTail.Stingray.Core;

/// <summary>
/// Execution strategy recommended by the planner.
/// </summary>
public enum ExecutionStrategy
{
    DiscreteGpuFull,
    DiscreteGpuPartial,
    IntegratedGpuHybrid,
    CpuSimdOnly
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
}

/// <summary>
/// Intelligent hardware-aware compute and layer offload planner.
/// Automatically balances workloads across discrete GPUs, APU/Vega iGPUs, and CPU SIMD engines.
/// </summary>
public static class SmartOffloadPlanner
{
    /// <summary>
    /// Computes the optimal execution plan for a model given total model size and layer count.
    /// </summary>
    public static OffloadPlan Plan(
        long modelSizeBytes,
        int totalLayers,
        int contextTokens = 2048,
        HardwareTopology? topology = null)
    {
        var topo = topology ?? HardwareCapabilities.Current;
        var cpu = topo.Cpu;
        var primaryGpu = topo.Gpus.Count > 0 ? topo.Gpus[0] : null;

        int physicalThreads = Math.Max(1, cpu.PhysicalCores);

        // 1. Pure CPU Fallback (No GPU detected)
        if (primaryGpu == null || primaryGpu.TotalMemoryBytes <= 0)
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
        if (primaryGpu.IsIntegrated)
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

        // 3. Discrete GPU (Dedicated VRAM)
        long contextVramBuffer = Math.Max(256L * 1024 * 1024, (long)contextTokens * 4096 * 4); // KV cache buffer
        long availableVram = Math.Max(0, primaryGpu.AvailableMemoryBytes - contextVramBuffer);

        // Full Offload
        if (availableVram >= (long)(modelSizeBytes * 1.15))
        {
            return new OffloadPlan
            {
                Strategy = ExecutionStrategy.DiscreteGpuFull,
                RecommendedBackend = primaryGpu.Backend,
                GpuLayersToOffload = totalLayers,
                RecommendedThreads = physicalThreads,
                Description = $"Full GPU acceleration on {primaryGpu.Name} (all {totalLayers} layers offloaded to VRAM)."
            };
        }

        // Partial Offload (Tight VRAM)
        long bytesPerLayer = totalLayers > 0 ? modelSizeBytes / totalLayers : modelSizeBytes;
        int layersToOffload = bytesPerLayer > 0
            ? Math.Clamp((int)(availableVram / bytesPerLayer), 0, totalLayers)
            : 0;

        if (layersToOffload > 0)
        {
            return new OffloadPlan
            {
                Strategy = ExecutionStrategy.DiscreteGpuPartial,
                RecommendedBackend = primaryGpu.Backend,
                GpuLayersToOffload = layersToOffload,
                RecommendedThreads = physicalThreads,
                Description = $"Partial VRAM offload on {primaryGpu.Name}: {layersToOffload}/{totalLayers} layers on GPU, remainder on CPU SIMD."
            };
        }

        // VRAM insufficient for even 1 layer -> Fall back to CPU SIMD
        return new OffloadPlan
        {
            Strategy = ExecutionStrategy.CpuSimdOnly,
            RecommendedBackend = "cpu",
            GpuLayersToOffload = 0,
            RecommendedThreads = physicalThreads,
            Description = $"VRAM budget ({primaryGpu.AvailableMemoryBytes / (1024 * 1024)}MB) insufficient for model ({modelSizeBytes / (1024 * 1024)}MB); smoothly routing to CPU SIMD."
        };
    }
}
