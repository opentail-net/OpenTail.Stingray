#nullable enable

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.TurboQuant;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray;

/// <summary>
/// Execution context allocated over a loaded <see cref="Model"/>.
/// Manages tokenizer access, context configuration, and execution state.
/// </summary>
public sealed class ModelContext : IModelContext
{
    private readonly Model _model;
    private readonly ContextParams _params;
    private readonly ITokenizer _tokenizer;
    private readonly object _lock = new();
    private IInferenceEngine? _engine;
    private bool _hasExplicitEngine;
    private bool _disposed;

    /// <inheritdoc/>
    public IModel Model => _model;

    /// <summary>Strongly-typed reference to the parent <see cref="OpenTail.Stingray.Model"/>.</summary>
    public Model ModelConcrete => _model;

    /// <inheritdoc/>
    public ITokenizer Tokenizer => _tokenizer;

    /// <inheritdoc/>
    public int ContextSize => _params.ContextSize > 0 ? (int)_params.ContextSize : _model.ContextLength;

    /// <summary>Context parameters applied to this execution context.</summary>
    public ContextParams Parameters => _params;

    /// <summary>Whether an explicit engine was configured via <see cref="SetEngine"/>.</summary>
    public bool HasExplicitEngine => _hasExplicitEngine;

    /// <summary>
    /// Gets the underlying <see cref="IInferenceEngine"/> powering this context.
    /// Lazily initialized with runtime-selected backend and forward pass if not explicitly configured.
    /// </summary>
    public IInferenceEngine Engine
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                return _engine ??= CreateDefaultEngine();
            }
        }
    }

    internal ModelContext(Model model, ContextParams contextParams)
    {
        _model = model;
        _params = contextParams;
        if (model.IsGguf)
        {
            _tokenizer = GgufTokenizer.FromGgufModel(model.Gguf);
        }
        else
        {
            var tokResult = HuggingFaceTokenizerSource.Load(model.ModelPath);
            if (!tokResult.IsUsable || tokResult.Source is null)
                throw new InvalidOperationException($"Failed to load tokenizer from '{model.ModelPath}'.");
            _tokenizer = GgufTokenizer.FromSource(tokResult.Source);
        }
    }

    /// <summary>
    /// Configures or replaces the underlying <see cref="IInferenceEngine"/> for this context.
    /// </summary>
    public void SetEngine(IInferenceEngine engine)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ThrowIfDisposed();
        lock (_lock)
        {
            if (_engine is IDisposable oldDisposable && !ReferenceEquals(_engine, engine))
            {
                oldDisposable.Dispose();
            }
            _engine = engine;
            _hasExplicitEngine = true;
        }
    }

    private IInferenceEngine CreateDefaultEngine()
    {
        var (fwd, _, owned) = BuildForwardPass(isContinuousBatching: false);
        var (thinkTokenId, endThinkTokenId) = _tokenizer.ReasoningTokens;
        return new InferenceEngine(fwd, _tokenizer, _model.Architecture, thinkTokenId, endThinkTokenId, owned: [.. owned]);
    }

    private (IForwardPass Fwd, ForwardPassKind Kind, List<IDisposable> Owned) BuildForwardPass(bool isContinuousBatching)
    {
        var owned = new List<IDisposable>();
        var hp = _model.Hyperparams;
        if (_params.RopeFrequencyBase is { } rfb and > 0)
        {
            hp = hp with { RopeTheta = rfb };
        }
        if (_params.RopeFrequencyScale is { } rfs and > 0)
        {
            hp = hp with { RopeTheta = hp.RopeTheta / rfs };
        }

        if (_params.ThreadCount > 0)
        {
            SimdKernels.CpuThreads = (int)_params.ThreadCount;
        }

        string backendStr = (_model.Parameters.Backend ?? "auto").Trim().ToLowerInvariant();
        int nGpuLayers = _model.Parameters.GpuLayerCount;
        bool isGguf = _model.IsGguf;
        string arch = _model.Architecture;

        ForwardPassBackend backend = backendStr switch
        {
            "cpu" => ForwardPassBackend.Cpu,
            "cuda" => ForwardPassBackend.Cuda,
            "vulkan" => ForwardPassBackend.Vulkan,
            "auto" or "" => ForwardPassBackend.Auto,
            _ => ForwardPassBackend.Auto,
        };

        string? unsupportedBackend = backendStr is not ("auto" or "" or "cpu" or "cuda" or "vulkan")
            ? _model.Parameters.Backend
            : null;

        var desc = ArchitectureRegistry.Find(arch);
        bool archSupported = desc?.IsUsable() ?? false;
        string? archRefusal = desc is not null && !desc.IsUsable()
            ? desc.GetRefusalMessage(arch)
            : (desc is null ? $"GGUF architecture '{arch}' is not admitted by OpenTail.Stingray." : null);

        bool turboQuant = !string.IsNullOrWhiteSpace(_params.TurboQuantMode);
        string tqMode = turboQuant ? _params.TurboQuantMode! : "auto";
        bool snapKvEnabled = SnapKvConfig.FromEnvironment().Enabled;

        bool unsupportedGpuPath = isGguf && nGpuLayers != 0 && GpuForwardPass.UnsupportedReason(_model.Gguf, hp) is not null;
        bool unsupportedPartialCudaPath = isGguf && GpuForwardPass.PartialOffloadUnsupportedReason(_model.Gguf, hp) is not null;
        bool unsupportedPartialVulkanPath = isGguf && GpuForwardPass.PartialOffloadUnsupportedReason(_model.Gguf, hp) is not null;

        bool hasMlaTensors = isGguf
            && _model.Gguf.FindTensor("blk.0.attn_kv_b.weight") is not null
            && _model.Gguf.FindTensor("blk.0.attn_q_a.weight") is null;

        bool isSafeTensorsGpu = !isGguf && (nGpuLayers != 0 && backendStr != "cpu");

        int plannedGpuLayers = hp.NumLayers;
        if (nGpuLayers != 0 && backend != ForwardPassBackend.Cpu && isGguf)
        {
            try
            {
                bool wantCuda = backend == ForwardPassBackend.Cuda || (backend == ForwardPassBackend.Auto && CudaBackend.IsAvailable());
                if (wantCuda && CudaBackend.IsAvailable())
                {
                    var hw = HardwareProfile.Detect();
                    plannedGpuLayers = TierPlanner.Plan(_model.Gguf, hp, hw, turboQuant, requestedCtxSize: ContextSize,
                        kvDtype: CudaForwardPass.ResolveConfiguredKvDType()).GpuLayers;
                }
                else
                {
                    var hw = HardwareProfile.Detect();
                    plannedGpuLayers = TierPlanner.Plan(_model.Gguf, hp, hw, turboQuant, requestedCtxSize: ContextSize).GpuLayers;
                }
            }
            catch
            {
                plannedGpuLayers = hp.NumLayers;
            }
        }

        int headDim = _params.TurboQuantHeadDim is { } tqHd and > 0 ? tqHd : hp.HeadDim;

        var request = new ForwardPassRequest
        {
            Frontend = ForwardPassFrontend.Cli,
            Architecture = arch,
            UnsupportedBackendName = unsupportedBackend,
            IsSafeTensors = !isGguf,
            PackageSupported = true,
            IsSafeTensorsGpuRequested = isSafeTensorsGpu,
            IsHybridSsm = hp.IsHybridSsm,
            HasHybridGdnLayers = hp.IsHybridSsm,
            HasCpuHybridGdnPass = hp.IsHybridSsm,
            IsMoE = hp.IsMoE,
            KvLoraRank = hp.KvLoraRank,
            HasMlaTensors = hasMlaTensors,
            ArchitectureSupported = archSupported,
            AllowUnverifiedArchitecture = _model.Parameters.AllowUnverifiedArch,
            ArchitectureRefusal = archRefusal,
            NumLayers = hp.NumLayers,
            GpuLayers = nGpuLayers,
            PlannedGpuLayers = plannedGpuLayers,
            Backend = backend,
            CudaAvailable = CudaBackend.IsAvailable(),
            UnsupportedGpuPath = unsupportedGpuPath,
            UnsupportedPartialCudaPath = unsupportedPartialCudaPath,
            UnsupportedPartialVulkanPath = unsupportedPartialVulkanPath,
            LayerHeadDim = hp.LayerHeadDim is not null,
            HasLayerHeadDim = hp.LayerHeadDim is not null,
            TurboQuant = turboQuant,
            TurboQuantMode = tqMode,
            HeadDim = headDim,
            IsContinuousBatching = isContinuousBatching,
            KVarNSnapKvBlocked = snapKvEnabled,
            KVarNCudaMoeBlocked = hp.IsMoE,
            KVarNPartialCudaBlocked = nGpuLayers > 0 && nGpuLayers < hp.NumLayers,
            PlannedGpuLayersForKVarN = plannedGpuLayers,
            LayerSplitOnly = unsupportedPartialVulkanPath || hp.LayerHeadDim is not null,
            IsGemma4 = hp.LayerHeadDim is not null,
        };

        var decision = ForwardPassSelection.Select(request);
        if (decision.IsRefused)
        {
            throw new NotSupportedException(decision.Refusal);
        }

        var cpuBackend = new CpuBackend();
        owned.Add(cpuBackend);

        TqQuantizer ResolveTq(string? kvarnBlocked)
        {
            if (!turboQuant) return TqQuantizer.LloydMax;
            kvarnBlocked ??= snapKvEnabled ? TqSupport.SnapKvReason : null;
            bool tqModeIsAuto = string.IsNullOrEmpty(tqMode) || tqMode.Equals("auto", StringComparison.OrdinalIgnoreCase);
            if (kvarnBlocked is null)
                return tqModeIsAuto ? TqQuantizer.KVarN : (tqMode.Equals("kvarn", StringComparison.OrdinalIgnoreCase) ? TqQuantizer.KVarN : TqQuantizer.LloydMax);
            if (!tqModeIsAuto && tqMode.Equals("kvarn", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"TqMode=kvarn is not supported on this path: {kvarnBlocked}.");
            if (!TqSupport.IsLloydMaxHeadDim(headDim))
                throw new InvalidOperationException(
                    $"TurboQuant with head dim {headDim} requires KVarN ({kvarnBlocked}), but Lloyd-Max — the " +
                    "only codec available on this path — ships codebooks for head dim 128/256 only. " +
                    "Use nGpuLayers=0 for the CPU KVarN path.");
            return TqQuantizer.LloydMax;
        }

        IForwardPass fwd;
        switch (decision.Kind!.Value)
        {
            case ForwardPassKind.SafeTensorsCpu:
                var sfwd = new ForwardPass(_model.TensorSource, cpuBackend, hp, maxContextLength: ContextSize);
                owned.Add(sfwd);
                fwd = sfwd;
                break;

            case ForwardPassKind.CpuDense:
                var dfwd = new ForwardPass(_model.TensorSource, cpuBackend, hp, maxContextLength: ContextSize);
                if (turboQuant)
                    dfwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: ResolveTq(null));
                owned.Add(dfwd);
                fwd = dfwd;
                break;

            case ForwardPassKind.CpuHybridGdn:
                var gdnFwd = new HybridGdnForwardPass(_model.Gguf, cpuBackend, hp, maxContextLength: ContextSize);
                owned.Add(gdnFwd);
                fwd = gdnFwd;
                break;

            case ForwardPassKind.RwkvCpu:
                var rwkvFwd = RwkvForwardPassBase.Create(_model.Gguf);
                owned.Add(rwkvFwd);
                fwd = rwkvFwd;
                break;

            case ForwardPassKind.GptOssCpu:
                var gptOssHp = GptOssHyperparams.FromModel(_model.Gguf);
                var gptOssFwd = new GptOssForwardPass(_model.Gguf, gptOssHp);
                owned.Add(gptOssFwd);
                fwd = gptOssFwd;
                break;

            case ForwardPassKind.GptOssVulkan:
                var vkGpt = new VulkanBackend();
                owned.Add(vkGpt);
                var gptOssHpVk = GptOssHyperparams.FromModel(_model.Gguf);
                var gptOssGpu = new GptOssGpuForwardPass(_model.Gguf, vkGpt, gptOssHpVk, maxContextLength: ContextSize);
                owned.Add(gptOssGpu);
                fwd = gptOssGpu;
                break;

            case ForwardPassKind.DeepSeek2Vulkan:
                var vkDs = new VulkanBackend();
                owned.Add(vkDs);
                var dsFwd = new DeepSeek2GpuForwardPass(_model.Gguf, vkDs, hp, maxContextLength: ContextSize);
                owned.Add(dsFwd);
                fwd = dsFwd;
                break;

            case ForwardPassKind.CudaDense:
                var cuda = CudaBackend.Create();
                owned.Add(cuda);
                var cfwd = new CudaForwardPass(_model.Gguf, cuda, hp, ContextSize,
                    enableTurboQuant: turboQuant,
                    tqQuantizer: ResolveTq(hp.IsMoE ? TqSupport.CudaMoeReason
                        : !TqSupport.IsKVarNCudaHeadDim(headDim) ? TqSupport.CudaHeadDimReason(headDim)
                        : null));
                owned.Add(cfwd);
                fwd = cfwd;
                break;

            case ForwardPassKind.CudaHybridGdn:
                var cudaGdn = CudaBackend.Create();
                owned.Add(cudaGdn);
                var placementGdn = new LayerPlacement(
                    GpuLayers: hp.NumLayers,
                    CpuLayers: 0,
                    GpuWeightBytes: 0,
                    GpuKvBytes: 0,
                    RecommendedCtxSize: ContextSize);
                var chgdn = new CudaHybridGdnForwardPass(_model.Gguf, cudaGdn, hp, placementGdn);
                owned.Add(chgdn);
                fwd = chgdn;
                break;

            case ForwardPassKind.CudaHybrid:
                var cudaHyb = CudaBackend.Create();
                owned.Add(cudaHyb);
                int effCudaLayers = nGpuLayers > 0 ? nGpuLayers : plannedGpuLayers;
                var hwProfileCuda = HardwareProfile.Detect(cudaHyb);
                var planCuda = TierPlanner.Plan(_model.Gguf, hp, hwProfileCuda, turboQuant, requestedCtxSize: ContextSize,
                    kvDtype: CudaForwardPass.ResolveConfiguredKvDType(), pinGpuLayers: effCudaLayers);
                var chfwd = new CudaHybridForwardPass(_model.Gguf, cudaHyb, hp, planCuda, turboQuant);
                owned.Add(chfwd);
                fwd = chfwd;
                break;

            case ForwardPassKind.VulkanDense:
                var vk = new VulkanBackend();
                owned.Add(vk);
                _ = ResolveTq(TqSupport.VulkanReason);
                var gfwd = new GpuForwardPass(_model.Gguf, vk, hp, ContextSize, enableTurboQuant: turboQuant,
                    kvDtype: CudaForwardPass.ResolveConfiguredKvDType());
                if (!_params.FlashAttention)
                {
                    gfwd.DisableFlashAttention = true;
                }
                owned.Add(gfwd);
                fwd = gfwd;
                break;

            case ForwardPassKind.VulkanHybridGdn:
                var vkGdn = new VulkanBackend();
                owned.Add(vkGdn);
                var placementVkGdn = new LayerPlacement(
                    GpuLayers: hp.NumLayers,
                    CpuLayers: 0,
                    GpuWeightBytes: 0,
                    GpuKvBytes: 0,
                    RecommendedCtxSize: ContextSize);
                var vhgdn = new VulkanHybridGdnForwardPass(_model.Gguf, vkGdn, hp, placementVkGdn);
                owned.Add(vhgdn);
                fwd = vhgdn;
                break;

            case ForwardPassKind.VulkanHybrid:
                var vkHyb = new VulkanBackend();
                owned.Add(vkHyb);
                _ = ResolveTq(TqSupport.VulkanReason);
                int effVkLayers = nGpuLayers > 0 ? nGpuLayers : plannedGpuLayers;
                var hwProfileVk = HardwareProfile.Detect(vkHyb);
                var planVk = TierPlanner.Plan(_model.Gguf, hp, hwProfileVk, turboQuant, requestedCtxSize: ContextSize,
                    pinGpuLayers: effVkLayers);
                var hfwd = new HybridForwardPass(_model.Gguf, vkHyb, hp, planVk, enableTq: turboQuant);
                owned.Add(hfwd);
                fwd = hfwd;
                break;

            case ForwardPassKind.VulkanLayerSplit:
                var vkSplit = new VulkanBackend();
                owned.Add(vkSplit);
                int effSplitLayers = nGpuLayers > 0 ? nGpuLayers : plannedGpuLayers;
                int split = Math.Min(effSplitLayers, VulkanLayerSplitForwardPass.MaxGpuLayers(hp));
                var sfwdSplit = new VulkanLayerSplitForwardPass(_model.Gguf, vkSplit, hp, split, ContextSize);
                owned.Add(sfwdSplit);
                fwd = sfwdSplit;
                break;

            default:
                throw new InvalidOperationException($"Unhandled forward pass kind: {decision.Kind}");
        }

        return (fwd, decision.Kind!.Value, owned);
    }

    /// <summary>
    /// Creates a <see cref="ContinuousBatchingEngine"/> instance configured for this context.
    /// </summary>
    public ContinuousBatchingEngine CreateContinuousBatchingEngine(int? maxBatchSize = null)
    {
        ThrowIfDisposed();
        int batchSize = maxBatchSize ?? (_params.BatchSize > 0 ? (int)_params.BatchSize : 8);
        var hp = _model.Hyperparams;
        var (fwd, kind, owned) = BuildForwardPass(isContinuousBatching: true);

        bool batchOk = false;
        if (fwd is IBatchedForwardPass)
        {
            if (kind is ForwardPassKind.CpuDense or ForwardPassKind.SafeTensorsCpu)
            {
                bool turboQuant = !string.IsNullOrWhiteSpace(_params.TurboQuantMode);
                batchOk = !hp.IsMoE && !turboQuant && hp.LayerHeadDim is null
                    && !hp.AttentionOutputGate && !hp.InputEmbeddingRmsNorm;
            }
            else if (kind == ForwardPassKind.CudaDense && fwd is CudaForwardPass cfwd)
            {
                batchOk = cfwd.SupportsContinuousBatching;
            }
        }

        if (!batchOk || fwd is not IBatchedForwardPass batchPass)
        {
            foreach (var d in owned) d.Dispose();
            throw new NotSupportedException($"The architecture '{_model.Architecture}' with forward pass '{kind}' does not support continuous batching.");
        }

        var (thinkTokenId, endThinkTokenId) = _tokenizer.ReasoningTokens;
        return new ContinuousBatchingEngine(batchPass, _tokenizer, _model.Architecture, batchSize, thinkTokenId, endThinkTokenId);
    }

    /// <summary>
    /// Ensures that this context has an active <see cref="ContinuousBatchingEngine"/> instance.
    /// If no explicit engine is configured, creates one via <see cref="CreateContinuousBatchingEngine"/>.
    /// </summary>
    public void EnsureContinuousBatchingEngine(int? maxBatchSize = null)
    {
        ThrowIfDisposed();
        lock (_lock)
        {
            if (_engine is ContinuousBatchingEngine) return;
            var batchEngine = CreateContinuousBatchingEngine(maxBatchSize);
            SetEngine(batchEngine);
        }
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ThrowIfDisposed();
        lock (_lock)
        {
            _engine?.Reset();
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            if (_engine is IDisposable disposableEngine)
            {
                disposableEngine.Dispose();
            }
            _engine = null;
            _model.UnregisterContext(this);
        }
    }
}
