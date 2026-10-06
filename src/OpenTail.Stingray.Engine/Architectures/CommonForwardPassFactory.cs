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
    /// Constructs a standard dense forward pass (CPU, SafeTensors, CUDA, Vulkan, or partial GPU offload).
    /// </summary>
    public static IForwardPass CreateDense(ArchitectureLoadContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        var hp = ctx.Probe.Hyperparams;

        switch (ctx.Decision.Kind)
        {
            case ForwardPassKind.SafeTensorsCpu:
            case ForwardPassKind.CpuDense:
            {
                var cpuBackend = ctx.CpuBackend ?? new CpuBackend();
                if (ctx.CpuBackend is null) ctx.TrackDisposable(cpuBackend);

                var dense = new ForwardPass(ctx.Probe.TensorSource, cpuBackend, hp, maxContextLength: ctx.ContextSize);
                ctx.TrackDisposable(dense);

                if (ctx.TurboQuant)
                {
                    dense.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: ctx.TqQuantizer);
                }
                return dense;
            }

            case ForwardPassKind.CudaDense:
            {
                var cuda = ctx.CudaBackend ?? CudaBackend.Create();
                if (ctx.CudaBackend is null) ctx.TrackDisposable(cuda);

                var cfwd = new CudaForwardPass(ctx.Probe.Gguf!, cuda, hp, ctx.ContextSize,
                    enableTurboQuant: ctx.TurboQuant,
                    tqQuantizer: ctx.TqQuantizer);
                ctx.TrackDisposable(cfwd);
                return cfwd;
            }

            case ForwardPassKind.CudaHybrid:
            {
                var cuda = ctx.CudaBackend ?? CudaBackend.Create();
                if (ctx.CudaBackend is null) ctx.TrackDisposable(cuda);

                var placement = ctx.Placement ?? throw new InvalidOperationException("LayerPlacement required for CudaHybrid.");
                var chybrid = new CudaHybridForwardPass(ctx.Probe.Gguf!, cuda, hp, placement, enableTq: ctx.TurboQuant);
                ctx.TrackDisposable(chybrid);
                return chybrid;
            }

            case ForwardPassKind.VulkanDense:
            {
                var vk = ctx.VulkanBackend ?? new VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);

                var gfwd = new GpuForwardPass(ctx.Probe.Gguf!, vk, hp, ctx.ContextSize,
                    enableTurboQuant: ctx.TurboQuant,
                    kvDtype: ctx.KvDType);
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
                var vhybrid = new HybridForwardPass(ctx.Probe.Gguf!, vk, hp, placement, enableTq: ctx.TurboQuant);
                ctx.TrackDisposable(vhybrid);
                return vhybrid;
            }

            case ForwardPassKind.VulkanLayerSplit:
            {
                var vk = ctx.VulkanBackend ?? new VulkanBackend();
                if (ctx.VulkanBackend is null) ctx.TrackDisposable(vk);

                var vsplit = new VulkanLayerSplitForwardPass(ctx.Probe.Gguf!, vk, hp, ctx.ContextSize, ctx.GpuLayers);
                ctx.TrackDisposable(vsplit);
                return vsplit;
            }

            default:
                throw new InvalidOperationException($"Unsupported forward pass kind '{ctx.Decision.Kind}' for dense architecture.");
        }
    }

    /// <summary>
    /// Constructs a hybrid GDN forward pass (CPU, CUDA, or Vulkan).
    /// </summary>
    public static IForwardPass CreateHybridGdn(ArchitectureLoadContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        var hp = ctx.Probe.Hyperparams;

        switch (ctx.Decision.Kind)
        {
            case ForwardPassKind.CpuHybridGdn:
            {
                var cpu = ctx.CpuBackend ?? new CpuBackend();
                if (ctx.CpuBackend is null) ctx.TrackDisposable(cpu);

                var cpuHybrid = new HybridGdnForwardPass(ctx.Probe.Gguf!, cpu, hp);
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

                var chgdn = new CudaHybridGdnForwardPass(ctx.Probe.Gguf!, cuda, hp, placement);
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

                var vhgdn = new VulkanHybridGdnForwardPass(ctx.Probe.Gguf!, vk, hp, placement);
                ctx.TrackDisposable(vhgdn);
                return vhgdn;
            }

            default:
                throw new InvalidOperationException($"Unsupported forward pass kind '{ctx.Decision.Kind}' for hybrid GDN architecture.");
        }
    }
}
