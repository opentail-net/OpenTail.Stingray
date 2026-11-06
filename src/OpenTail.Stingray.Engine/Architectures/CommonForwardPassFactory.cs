namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Vulkan;

/// <summary>
/// Reusable forward-pass construction routines for standard transformer architectures and shared variants.
/// </summary>
public static class CommonForwardPassFactory
{
    /// <summary>
    /// <c>STINGRAY_TP_DEGREE</c>: the tensor-parallel degree flag for the CUDA dense path
    /// (docs/4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md). Reading the env var here,
    /// rather than threading a new field through <see cref="ArchitectureLoadContext"/>/<c>EngineSettings</c>,
    /// keeps this first slice's surface small; a CLI <c>--tp</c> option and a typed settings field are a
    /// later increment, not yet built.
    /// </summary>
    internal static void CheckTensorParallelDegree()
    {
        string? raw = Environment.GetEnvironmentVariable("STINGRAY_TP_DEGREE");
        if (string.IsNullOrEmpty(raw)) return;
        if (!int.TryParse(raw, out int degree) || degree < 1)
            throw new NotSupportedException($"STINGRAY_TP_DEGREE='{raw}' is not a positive integer.");
        if (degree == 1) return;

        // CudaTensorParallelGroup and CudaTensorParallelSharding (the weight-sharding/AllReduce math,
        // unit tested) exist; what does not exist yet is the dense-llama forward pass that actually
        // uploads sharded weights onto the group's ranks and calls AllReduceSumAcrossRanks at the two
        // reconvergence points (attention output, FFN down). Refusing clearly here, rather than
        // silently running the ordinary single-GPU CudaForwardPass and ignoring the flag, matches this
        // project's existing pattern for a requested-but-unimplemented path (e.g. Gemma 2's GPU
        // soft-capping refusal) instead of claiming a capability that is not there yet.
        throw new NotSupportedException(
            $"STINGRAY_TP_DEGREE={degree}: tensor-parallel GPU grouping (CudaTensorParallelGroup) and " +
            "weight-sharding math (CudaTensorParallelSharding) exist and are unit tested, but the dense-" +
            "llama forward pass that wires sharded uploads and AllReduce into CudaForwardPass's decode " +
            "loop is not built yet -- see docs/4-performance/gpu/2026-11-06-cuda-tensor-parallelism-plan.md. " +
            "Unset STINGRAY_TP_DEGREE (or set it to 1) to use the ordinary single-GPU path.");
    }

    /// <summary>
    /// Constructs a standard dense forward pass (CPU, SafeTensors, CUDA, Vulkan, or partial GPU offload).
    /// </summary>
    public static IForwardPass CreateDense(ArchitectureLoadContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        var hp = ctx.Hyperparams;

        switch (ctx.Plan.ForwardPassKind)
        {
            case ForwardPassKind.SafeTensorsCpu:
            case ForwardPassKind.CpuDense:
            {
                var cpuBackend = ctx.CpuBackend ?? new CpuBackend();
                if (ctx.CpuBackend is null) ctx.TrackDisposable(cpuBackend);

                var dense = new ForwardPass(ctx.Probe.TensorSource, cpuBackend, hp, maxContextLength: ctx.ContextSize,
                    prefillDequantCacheBytes: ctx.PrefillDequantCacheBytes,
                    settings: ctx.Settings);
                ctx.TrackDisposable(dense);

                if (ctx.TurboQuant)
                {
                    dense.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: ctx.TqQuantizer);
                }
                return dense;
            }

            case ForwardPassKind.CudaDense:
            {
                CheckTensorParallelDegree();

                var cuda = ctx.CudaBackend ?? CudaBackend.Create();
                if (ctx.CudaBackend is null) ctx.TrackDisposable(cuda);

                var cfwd = new CudaForwardPass(ctx.Probe.Gguf!, cuda, hp, ctx.ContextSize,
                    enableTurboQuant: ctx.TurboQuant,
                    tqQuantizer: ctx.TqQuantizer,
                    preferBatchingOverAutoSnapKv: ctx.PreferBatchingOverAutoSnapKv,
                    settings: ctx.Settings);
                ctx.TrackDisposable(cfwd);
                return cfwd;
            }

            case ForwardPassKind.CudaHybrid:
            {
                var cuda = ctx.CudaBackend ?? CudaBackend.Create();
                if (ctx.CudaBackend is null) ctx.TrackDisposable(cuda);

                var placement = ctx.Placement ?? throw new InvalidOperationException("LayerPlacement required for CudaHybrid.");
                var chybrid = new CudaHybridForwardPass(ctx.Probe.Gguf!, cuda, hp, placement, enableTq: ctx.TurboQuant, settings: ctx.Settings);
                ctx.TrackDisposable(chybrid);
                return chybrid;
            }

            case ForwardPassKind.VulkanDense:
            {
                var vk = ctx.VulkanBackend ?? new VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);

                var gfwd = new GpuForwardPass(ctx.Probe.Gguf!, vk, hp, ctx.ContextSize,
                    enableTurboQuant: ctx.TurboQuant,
                    kvDtype: ctx.KvDType,
                    settings: ctx.Settings);
                if (!ctx.FlashAttention)
                {
                    gfwd.DisableFlashAttention = true;
                }
                ctx.TrackDisposable(gfwd);
                return gfwd;
            }

            case ForwardPassKind.VulkanHybrid:
            {
                var vk = ctx.VulkanBackend ?? new VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);

                var placement = ctx.Placement ?? throw new InvalidOperationException("LayerPlacement required for VulkanHybrid.");
                var vhybrid = new HybridForwardPass(ctx.Probe.Gguf!, vk, hp, placement, enableTq: ctx.TurboQuant, settings: ctx.Settings);
                ctx.TrackDisposable(vhybrid);
                return vhybrid;
            }

            case ForwardPassKind.VulkanLayerSplit:
            {
                var vk = ctx.VulkanBackend ?? new VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);

                var vsplit = new VulkanLayerSplitForwardPass(ctx.Probe.Gguf!, vk, hp, ctx.ContextSize, ctx.GpuLayers, settings: ctx.Settings);
                ctx.TrackDisposable(vsplit);
                return vsplit;
            }

            default:
                throw new InvalidOperationException($"Unsupported forward pass kind '{ctx.Plan.ForwardPassKind}' for dense architecture.");
        }
    }

    /// <summary>
    /// Constructs a hybrid GDN forward pass (CPU, CUDA, or Vulkan).
    /// </summary>
    public static IForwardPass CreateHybridGdn(ArchitectureLoadContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        var hp = ctx.Hyperparams;

        // Mamba-2 (Granite-H, Nemotron-H) and short-conv (LFM2) hybrids are not Gated-DeltaNet: the planner routes them to
        // the dense passes, which carry their recurrent layers internally (hp.IsMamba2Layer / hp.IsShortConvLayer).
        // Found 2026-10-08: since the plan-driven migration (5c591e6a) granitehybrid crashed here with "Unsupported forward pass kind 'CpuDense'".
        if ((hp.IsMamba2Layer is not null || hp.IsShortConvLayer is not null) && ctx.Plan.ForwardPassKind is not
            (ForwardPassKind.CpuHybridGdn or ForwardPassKind.CudaHybridGdn or ForwardPassKind.VulkanHybridGdn))
            return CreateDense(ctx);

        switch (ctx.Plan.ForwardPassKind)
        {
            case ForwardPassKind.CpuHybridGdn:
            {
                var cpu = ctx.CpuBackend ?? new CpuBackend();
                if (ctx.CpuBackend is null) ctx.TrackDisposable(cpu);

                var cpuHybrid = new HybridGdnForwardPass(ctx.Probe.Gguf!, cpu, hp, settings: ctx.Settings);
                ctx.TrackDisposable(cpuHybrid);
                return cpuHybrid;
            }

            case ForwardPassKind.CudaHybridGdn:
            {
                var cuda = ctx.CudaBackend ?? CudaBackend.Create();
                if (ctx.CudaBackend is null) ctx.TrackDisposable(cuda);

                var placement = ctx.Placement ?? new LayerPlacement(
                    GpuLayers: hp.NumLayers,
                    CpuLayers: 0,
                    GpuWeightBytes: 0,
                    GpuKvBytes: 0,
                    RecommendedCtxSize: ctx.ContextSize > 0 ? ctx.ContextSize : Math.Min(hp.ContextLength, 4096));

                var chgdn = new CudaHybridGdnForwardPass(ctx.Probe.Gguf!, cuda, hp, placement, settings: ctx.Settings);
                ctx.TrackDisposable(chgdn);
                return chgdn;
            }

            case ForwardPassKind.VulkanHybridGdn:
            {
                var vk = ctx.VulkanBackend ?? new VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);

                var placement = ctx.Placement ?? new LayerPlacement(
                    GpuLayers: hp.NumLayers,
                    CpuLayers: 0,
                    GpuWeightBytes: 0,
                    GpuKvBytes: 0,
                    RecommendedCtxSize: ctx.ContextSize > 0 ? ctx.ContextSize : Math.Min(hp.ContextLength, 4096));

                var vhgdn = new VulkanHybridGdnForwardPass(ctx.Probe.Gguf!, vk, hp, placement, settings: ctx.Settings);
                ctx.TrackDisposable(vhgdn);
                return vhgdn;
            }

            default:
                throw new InvalidOperationException($"Unsupported forward pass kind '{ctx.Plan.ForwardPassKind}' for hybrid GDN architecture.");
        }
    }
}
