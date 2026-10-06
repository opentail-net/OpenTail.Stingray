#nullable enable

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine.Runtime;

/// <summary>
/// Authoritative concrete execution resource boundary for an ExecutionPlan (§5.1 of plan).
/// Owns the active backend instances, forward pass, tokenizer, and inference engine.
/// Guarantees that once an ExecutionPlan is resolved, execution never rediscovers policy.
/// </summary>
public sealed class RuntimeInstance : IDisposable
{
    private readonly List<IDisposable> _ownedDisposables;
    private bool _disposed;

    public ExecutionPlan Plan { get; }
    public Model Model { get; }
    public IInferenceEngine Engine { get; }
    public IForwardPass ForwardPass { get; }
    public ITokenizer Tokenizer { get; }
    public CpuBackend? CpuBackend { get; }
    public CudaBackend? CudaBackend { get; }
    public VulkanBackend? VulkanBackend { get; }
    public IReadOnlyList<IDisposable> OwnedDisposables => _ownedDisposables;

    private RuntimeInstance(
        ExecutionPlan plan,
        Model model,
        IInferenceEngine engine,
        IForwardPass forwardPass,
        ITokenizer tokenizer,
        CpuBackend? cpuBackend,
        CudaBackend? cudaBackend,
        VulkanBackend? vulkanBackend,
        List<IDisposable> ownedDisposables)
    {
        Plan = plan;
        Model = model;
        Engine = engine;
        ForwardPass = forwardPass;
        Tokenizer = tokenizer;
        CpuBackend = cpuBackend;
        CudaBackend = cudaBackend;
        VulkanBackend = vulkanBackend;
        _ownedDisposables = ownedDisposables;
    }

    /// <summary>
    /// Constructs a RuntimeInstance from a pre-loaded Model and ExecutionPlan.
    /// Strictly validates plan invariants, backend operational status, and architecture admission.
    /// Never reads execution policy from <see cref="Model.Parameters"/>.
    /// </summary>
    public static RuntimeInstance Create(ExecutionPlan plan, Model model, ModelHyperparams? hyperparamsOverride = null)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(model);

        if (plan.BackendPlan?.ThreadCount > 0)
        {
            SimdKernels.CpuThreads = plan.BackendPlan.ThreadCount;
        }

        var effectiveHp = hyperparamsOverride ?? model.Hyperparams;

        // 1. Validate plan executability
        if (!plan.IsExecutable)
        {
            throw new PlanNotExecutableException($"Plan for model '{plan.ModelPath}' is marked as non-executable.", plan);
        }

        // 2. Validate format
        if (plan.ModelFormat == ModelFormat.SafeTensors && model.IsGguf)
        {
            throw new PlanNotExecutableException($"Plan specifies SafeTensors format, but model '{model.ModelPath}' is GGUF.", plan);
        }
        if (plan.ModelFormat == ModelFormat.Gguf && !model.IsGguf)
        {
            throw new PlanNotExecutableException($"Plan specifies GGUF format, but model '{model.ModelPath}' is not GGUF.", plan);
        }

        // 3. Validate architecture descriptor and admission
        var descriptor = ArchitectureRegistry.Find(model.Architecture)
            ?? throw new PlanNotExecutableException($"Architecture '{model.Architecture}' is not supported.", plan);

        if (!descriptor.IsUsable())
        {
            throw new PlanNotExecutableException(descriptor.GetRefusalMessage(model.Architecture), plan);
        }

        if (!string.IsNullOrEmpty(plan.Provenance?.TargetArchitecture)
            && !string.Equals(plan.Provenance.TargetArchitecture, model.Architecture, StringComparison.OrdinalIgnoreCase))
        {
            throw new PlanNotExecutableException(
                $"Plan target architecture '{plan.Provenance.TargetArchitecture}' does not match model architecture '{model.Architecture}'.", plan);
        }

        // 4. Validate backend operational availability (Zero silent fallbacks!)
        var backend = plan.BackendPlan?.Backend
            ?? (Enum.TryParse<ForwardPassBackend>(plan.Backend, true, out var b) ? b : ForwardPassBackend.Cpu);

        if (backend == ForwardPassBackend.Cuda && !CudaBackend.IsAvailable())
        {
            throw new PlanNotExecutableException("Planned CUDA backend is not operational or available on this system.", plan);
        }

        var ownedDisposables = new List<IDisposable>();
        CpuBackend? cpuBackend = null;
        CudaBackend? cudaBackend = null;
        VulkanBackend? vulkanBackend = null;
        ForwardPass? cpuDensePass = null;

        try
        {
            // 5. Allocate required backends
            bool needsCpu = backend == ForwardPassBackend.Cpu
                || plan.ForwardPassKind is ForwardPassKind.CpuDense or ForwardPassKind.SafeTensorsCpu or ForwardPassKind.CpuHybridGdn
                || (plan.Placement != null && plan.Placement.CpuLayers > 0)
                || plan.ForwardPassKind is ForwardPassKind.CudaHybrid or ForwardPassKind.VulkanHybrid;

            if (needsCpu)
            {
                cpuBackend = new CpuBackend();
                ownedDisposables.Add(cpuBackend);
            }

            if (backend == ForwardPassBackend.Cuda
                || plan.ForwardPassKind is ForwardPassKind.CudaDense or ForwardPassKind.CudaHybrid or ForwardPassKind.CudaHybridGdn)
            {
                try
                {
                    cudaBackend = CudaBackend.Create();
                    ownedDisposables.Add(cudaBackend);
                }
                catch (Exception ex)
                {
                    throw new PlanNotExecutableException("Planned CUDA backend failed to initialize or is not operational on this system.", ex, plan);
                }
            }

            if (backend == ForwardPassBackend.Vulkan
                || plan.ForwardPassKind is ForwardPassKind.VulkanDense or ForwardPassKind.VulkanHybrid or ForwardPassKind.VulkanLayerSplit or ForwardPassKind.VulkanHybridGdn)
            {
                try
                {
                    vulkanBackend = new VulkanBackend();
                    ownedDisposables.Add(vulkanBackend);
                }
                catch (Exception ex)
                {
                    throw new PlanNotExecutableException("Planned Vulkan backend failed to initialize or is not operational on this system.", ex, plan);
                }
            }

            // 6. Allocate baseline CPU dense pass for hybrid configurations if needed
            if (plan.ForwardPassKind is ForwardPassKind.CudaHybrid or ForwardPassKind.VulkanHybrid)
            {
                if (!effectiveHp.IsHybridSsm && model.IsGguf)
                {
                    cpuDensePass = new ForwardPass(
                        model.Gguf,
                        cpuBackend ?? new CpuBackend(),
                        effectiveHp,
                        maxContextLength: plan.ContextSize,
                        prefillDequantCacheBytes: 0);
                    ownedDisposables.Add(cpuDensePass);
                }
            }

            // 7. Construct forward pass via plan-driven ArchitectureLoadContext
            var probe = new ArchitectureProbe
            {
                Path = model.ModelPath,
                Architecture = model.Architecture,
                TensorSource = model.TensorSource,
                Hyperparams = effectiveHp,
                IsGguf = model.IsGguf,
                Gguf = model.IsGguf ? model.Gguf : null,
            };

            var loadContext = new ArchitectureLoadContext
            {
                Probe = probe,
                Plan = plan,
                CpuBackend = cpuBackend,
                CudaBackend = cudaBackend,
                VulkanBackend = vulkanBackend,
                CpuDensePass = cpuDensePass,
            };

            var forwardPass = descriptor.ConstructForwardPass(loadContext);
            ownedDisposables.AddRange(loadContext.OwnedDisposables);

            // 8. Construct tokenizer
            ITokenizer tokenizer;
            if (model.IsGguf)
            {
                tokenizer = GgufTokenizer.FromGgufModel(model.Gguf);
            }
            else
            {
                var tokResult = HuggingFaceTokenizerSource.Load(model.ModelPath);
                if (!tokResult.IsUsable || tokResult.Source is null)
                {
                    throw new PlanNotExecutableException($"Failed to load tokenizer from '{model.ModelPath}'.", plan);
                }
                tokenizer = GgufTokenizer.FromSource(tokResult.Source);
            }

            var (thinkTokenId, endThinkTokenId) = tokenizer.ReasoningTokens;

            // 9. Construct inference engine
            IInferenceEngine engine;
            if (plan.Batching?.Mode == BatchingMode.Continuous)
            {
                if (forwardPass is not IBatchedForwardPass batchedPass)
                {
                    throw new PlanNotExecutableException(
                        $"Continuous batching planned but forward pass '{forwardPass.GetType().Name}' does not implement IBatchedForwardPass.", plan);
                }
                int batchSize = plan.Batching.MaxBatchSize > 0 ? plan.Batching.MaxBatchSize : 8;
                engine = new ContinuousBatchingEngine(batchedPass, tokenizer, model.Architecture, batchSize, thinkTokenId, endThinkTokenId);
            }
            else
            {
                engine = new InferenceEngine(forwardPass, tokenizer, model.Architecture, thinkTokenId, endThinkTokenId, owned: [.. ownedDisposables]);
            }

            return new RuntimeInstance(
                plan,
                model,
                engine,
                forwardPass,
                tokenizer,
                cpuBackend,
                cudaBackend,
                vulkanBackend,
                ownedDisposables);
        }
        catch
        {
            foreach (var d in ownedDisposables)
            {
                try { d.Dispose(); } catch { }
            }
            throw;
        }
    }

    /// <summary>
    /// Loads the model from <see cref="ExecutionPlan.ModelPath"/> and creates a RuntimeInstance.
    /// </summary>
    public static RuntimeInstance Create(ExecutionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        var model = Model.Load(plan.ModelPath);
        return Create(plan, model);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        (Engine as IDisposable)?.Dispose();
        ForwardPass.Dispose();

        foreach (var d in _ownedDisposables)
        {
            try { d.Dispose(); } catch { }
        }
    }
}
