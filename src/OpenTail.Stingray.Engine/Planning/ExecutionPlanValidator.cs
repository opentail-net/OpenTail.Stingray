#nullable enable

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// Enforces structural and cross-field semantic invariants on <see cref="ExecutionPlan"/>.
/// Ensures that runtime execution never encounters ambiguous, "auto", or conflicting directives.
/// </summary>
public static class ExecutionPlanValidator
{
    /// <summary>
    /// Validates an <see cref="ExecutionPlan"/>, throwing <see cref="InvalidOperationException"/>
    /// if any invariant is violated.
    /// </summary>
    public static void Validate(ExecutionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);

        if (string.Equals(plan.Backend, "auto", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(plan.SelectedBackend, "auto", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Execution plan backend must be resolved and cannot remain 'auto'.");
        }

        if (plan.GpuLayers < 0)
        {
            throw new InvalidOperationException($"Execution plan GpuLayers cannot be negative (got {plan.GpuLayers}).");
        }

        if (plan.CpuLayers < 0)
        {
            throw new InvalidOperationException($"Execution plan CpuLayers cannot be negative (got {plan.CpuLayers}).");
        }

        if (plan.ContextSize <= 0)
        {
            throw new InvalidOperationException($"Execution plan ContextSize must be positive (got {plan.ContextSize}).");
        }

        if (plan.BackendPlan != null)
        {
            switch (plan.ForwardPassKind)
            {
                case ForwardPassKind.VulkanDense:
                case ForwardPassKind.VulkanHybrid:
                case ForwardPassKind.VulkanHybridGdn:
                case ForwardPassKind.VulkanLayerSplit:
                case ForwardPassKind.DeepSeek2Vulkan:
                case ForwardPassKind.GptOssVulkan:
                    if (plan.BackendPlan.Backend != ForwardPassBackend.Vulkan)
                    {
                        throw new InvalidOperationException(
                            $"Forward pass kind '{plan.ForwardPassKind}' requires Vulkan backend, but plan specifies '{plan.BackendPlan.Backend}'.");
                    }
                    break;

                case ForwardPassKind.CudaDense:
                case ForwardPassKind.CudaHybrid:
                case ForwardPassKind.CudaHybridGdn:
                    if (plan.BackendPlan.Backend != ForwardPassBackend.Cuda)
                    {
                        throw new InvalidOperationException(
                            $"Forward pass kind '{plan.ForwardPassKind}' requires CUDA backend, but plan specifies '{plan.BackendPlan.Backend}'.");
                    }
                    break;

                case ForwardPassKind.SafeTensorsCpu:
                    if (plan.BackendPlan.Backend != ForwardPassBackend.Cpu)
                    {
                        throw new InvalidOperationException(
                            $"Forward pass kind SafeTensorsCpu requires CPU backend, but plan specifies '{plan.BackendPlan.Backend}'.");
                    }
                    break;
            }
        }

        if (plan.Batching != null)
        {
            if (plan.Batching.Mode == BatchingMode.Continuous)
            {
                if (plan.Batching.MaxBatchSize < 1)
                {
                    throw new InvalidOperationException(
                        $"Continuous batching requires MaxBatchSize >= 1 (got {plan.Batching.MaxBatchSize}).");
                }

                if (plan.ForwardPassKind is ForwardPassKind.RwkvCpu or ForwardPassKind.SafeTensorsCpu)
                {
                    throw new InvalidOperationException(
                        $"Forward pass kind '{plan.ForwardPassKind}' does not support continuous batching.");
                }
            }
        }

        if (plan.Speculation != null && plan.Speculation.Mode != SpeculationMode.None)
        {
            if (plan.ForwardPassKind is ForwardPassKind.RwkvCpu or ForwardPassKind.CpuHybridGdn or ForwardPassKind.CudaHybridGdn or ForwardPassKind.VulkanHybridGdn)
            {
                throw new InvalidOperationException(
                    $"Forward pass kind '{plan.ForwardPassKind}' does not support speculative decoding.");
            }
        }

        if (plan.State != null && plan.State.TurboQuant)
        {
            if (plan.ForwardPassKind is ForwardPassKind.RwkvCpu or ForwardPassKind.CpuHybridGdn or ForwardPassKind.CudaHybridGdn or ForwardPassKind.VulkanHybridGdn)
            {
                throw new InvalidOperationException(
                    $"Forward pass kind '{plan.ForwardPassKind}' does not support TurboQuant KV cache compression.");
            }
        }
    }
}
