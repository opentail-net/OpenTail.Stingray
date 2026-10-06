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
        bool useGpu = nGpuLayers != 0 && backendStr != "cpu";

        var cpuBackend = new CpuBackend();
        owned.Add(cpuBackend);

        IForwardPass fwd;
        if (useGpu && _model.IsGguf)
        {
            bool isCuda = backendStr == "cuda" || (backendStr == "auto" && CudaBackend.IsAvailable());
            bool isVulkan = backendStr == "vulkan" || (backendStr == "auto" && !isCuda);

            if (isCuda)
            {
                if (!CudaBackend.IsAvailable() && backendStr == "cuda")
                {
                    throw new PlatformNotSupportedException("CUDA backend was requested but CUDA is not available on this system.");
                }
                var cuda = CudaBackend.Create();
                owned.Add(cuda);
                bool enableTq = !string.IsNullOrEmpty(_params.TurboQuantMode);
                var tqQuantizer = _params.TurboQuantMode?.ToLowerInvariant() == "lloydmax"
                    ? TqQuantizer.LloydMax
                    : TqQuantizer.KVarN;

                if (nGpuLayers < 0 || nGpuLayers >= hp.NumLayers)
                {
                    var cudaFwd = new CudaForwardPass(_model.Gguf, cuda, hp,
                        maxContextLength: ContextSize,
                        enableTurboQuant: enableTq,
                        tqQuantizer: tqQuantizer);
                    owned.Add(cudaFwd);
                    fwd = cudaFwd;
                }
                else
                {
                    var placement = new LayerPlacement(
                        GpuLayers: nGpuLayers,
                        CpuLayers: hp.NumLayers - nGpuLayers,
                        GpuWeightBytes: 0,
                        GpuKvBytes: 0,
                        RecommendedCtxSize: ContextSize);
                    var hybridCuda = new CudaHybridForwardPass(_model.Gguf, cuda, hp,
                        placement,
                        enableTq: enableTq);
                    owned.Add(hybridCuda);
                    fwd = hybridCuda;
                }
            }
            else if (isVulkan)
            {
                var vk = new VulkanBackend();
                owned.Add(vk);
                bool enableTq = !string.IsNullOrEmpty(_params.TurboQuantMode);
                if (nGpuLayers < 0 || nGpuLayers >= hp.NumLayers)
                {
                    var vkFwd = new GpuForwardPass(_model.Gguf, vk, hp,
                        maxContextLength: ContextSize,
                        enableTurboQuant: enableTq);
                    owned.Add(vkFwd);
                    fwd = vkFwd;
                }
                else
                {
                    var placement = new LayerPlacement(
                        GpuLayers: nGpuLayers,
                        CpuLayers: hp.NumLayers - nGpuLayers,
                        GpuWeightBytes: 0,
                        GpuKvBytes: 0,
                        RecommendedCtxSize: ContextSize);
                    var hybridVk = new HybridForwardPass(_model.Gguf, vk, hp,
                        placement,
                        enableTq: enableTq);
                    owned.Add(hybridVk);
                    fwd = hybridVk;
                }
            }
            else
            {
                fwd = CreateCpuForwardPass(cpuBackend, hp, owned);
            }
        }
        else
        {
            fwd = CreateCpuForwardPass(cpuBackend, hp, owned);
        }

        var (thinkTokenId, endThinkTokenId) = _tokenizer.ReasoningTokens;
        return new InferenceEngine(fwd, _tokenizer, _model.Architecture, thinkTokenId, endThinkTokenId, owned: [.. owned]);
    }

    private IForwardPass CreateCpuForwardPass(CpuBackend cpuBackend, ModelHyperparams hp, List<IDisposable> owned)
    {
        if (hp.IsHybridSsm && _model.IsGguf)
        {
            var gdn = new HybridGdnForwardPass(_model.Gguf, cpuBackend, hp, maxContextLength: ContextSize);
            owned.Add(gdn);
            return gdn;
        }

        var cpuFwd = new ForwardPass(_model.TensorSource, cpuBackend, hp, maxContextLength: ContextSize);
        if (!string.IsNullOrEmpty(_params.TurboQuantMode))
        {
            var quantizer = _params.TurboQuantMode.ToLowerInvariant() == "lloydmax"
                ? TqQuantizer.LloydMax
                : TqQuantizer.KVarN;
            cpuFwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: quantizer);
        }
        owned.Add(cpuFwd);
        return cpuFwd;
    }

    /// <summary>
    /// Creates a <see cref="ContinuousBatchingEngine"/> instance configured for this context.
    /// </summary>
    public ContinuousBatchingEngine CreateContinuousBatchingEngine(int? maxBatchSize = null)
    {
        ThrowIfDisposed();
        int batchSize = maxBatchSize ?? (_params.BatchSize > 0 ? (int)_params.BatchSize : 8);
        var owned = new List<IDisposable>();
        var cpuBackend = new CpuBackend();
        owned.Add(cpuBackend);
        var hp = _model.Hyperparams;

        if (hp.IsMoE || !string.IsNullOrEmpty(_params.TurboQuantMode) || hp.LayerHeadDim is not null)
        {
            throw new NotSupportedException($"The architecture '{_model.Architecture}' or configuration does not support continuous batching.");
        }

        var fwd = new ForwardPass(_model.TensorSource, cpuBackend, hp, maxContextLength: ContextSize);
        owned.Add(fwd);
        var (thinkTokenId, endThinkTokenId) = _tokenizer.ReasoningTokens;
        return new ContinuousBatchingEngine(fwd, _tokenizer, _model.Architecture, batchSize, thinkTokenId, endThinkTokenId);
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
