#nullable enable

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Planning;
using OpenTail.Stingray.Engine.Runtime;

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
    private RuntimeInstance? _runtimeInstance;
    private IInferenceEngine? _engine;
    private ExecutionPlan? _plan;
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
    /// Gets the authoritative execution plan governing this context's runtime.
    /// If an engine has already been instantiated, returns the active plan.
    /// Otherwise, lazily plans the default execution configuration.
    /// </summary>
    public ExecutionPlan ExecutionPlan
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                if (_runtimeInstance is not null)
                {
                    return _runtimeInstance.Plan;
                }
                return _plan ??= Plan(isContinuousBatching: false);
            }
        }
    }

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
            if (_runtimeInstance is not null && !ReferenceEquals(_runtimeInstance.Engine, engine))
            {
                _runtimeInstance.Dispose();
                _runtimeInstance = null;
            }
            else if (_engine is IDisposable oldDisposable && !ReferenceEquals(_engine, engine))
            {
                oldDisposable.Dispose();
            }
            _engine = engine;
            _hasExplicitEngine = true;
        }
    }

    private IInferenceEngine CreateDefaultEngine()
    {
        var plan = _plan ?? Plan(isContinuousBatching: false);
        _runtimeInstance = RuntimeInstance.Create(plan, _model);
        _plan = _runtimeInstance.Plan;
        return _runtimeInstance.Engine;
    }

    /// <summary>
    /// Creates a <see cref="ContinuousBatchingEngine"/> instance configured for this context.
    /// </summary>
    public ContinuousBatchingEngine CreateContinuousBatchingEngine(int? maxBatchSize = null)
    {
        ThrowIfDisposed();
        int batchSize = maxBatchSize ?? (_params.BatchSize > 0 ? (int)_params.BatchSize : 8);
        var plan = Plan(isContinuousBatching: true, maxBatchSize: batchSize);
        var instance = RuntimeInstance.Create(plan, _model);

        if (instance.Engine is not ContinuousBatchingEngine batchEngine)
        {
            instance.Dispose();
            throw new NotSupportedException($"The architecture '{_model.Architecture}' with forward pass '{plan.ForwardPassKind}' does not support continuous batching.");
        }

        lock (_lock)
        {
            if (_runtimeInstance is not null && !ReferenceEquals(_runtimeInstance, instance))
            {
                _runtimeInstance.Dispose();
            }
            else if (_engine is IDisposable oldDisposable && !ReferenceEquals(_engine, batchEngine))
            {
                oldDisposable.Dispose();
            }

            _runtimeInstance = instance;
            _engine = batchEngine;
            _plan = plan;
        }

        return batchEngine;
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

    private ExecutionPlan Plan(bool isContinuousBatching, int? maxBatchSize = null)
    {
        int batchSize = maxBatchSize ?? (_params.BatchSize > 0 ? (int)_params.BatchSize : (isContinuousBatching ? 8 : 1));
        bool turboQuant = !string.IsNullOrWhiteSpace(_params.TurboQuantMode);
        string tqMode = turboQuant ? _params.TurboQuantMode! : "auto";
        var request = new ExecutionRequest
        {
            Goal = "balanced",
            PinnedBackend = _model.Parameters.Backend,
            PinnedGpuLayers = _model.Parameters.GpuLayerCount,
            PinnedContextSize = ContextSize,
            TurboQuant = turboQuant,
            TurboQuantMode = tqMode,
            FlashAttention = _params.FlashAttention,
            RopeFrequencyBase = _params.RopeFrequencyBase,
            RopeFrequencyScale = _params.RopeFrequencyScale,
            ThreadCount = (int)_params.ThreadCount,
            BatchingMode = isContinuousBatching ? BatchingMode.Continuous : BatchingMode.Sequential,
            MaxBatchSize = batchSize,
            AllowUnverifiedArchitecture = _model.Parameters.AllowUnverifiedArch,
        };

        var modelDesc = ModelDescription.FromModel(_model);
        var capabilities = BackendCapabilities.Detect();
        return ExecutionPlanner.Plan(modelDesc, ExecutionRequestEnvironment.ApplyTo(request), capabilities);
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
            if (_runtimeInstance is not null)
            {
                _runtimeInstance.Dispose();
                _runtimeInstance = null;
            }
            else if (_engine is IDisposable disposableEngine)
            {
                disposableEngine.Dispose();
            }
            _engine = null;
            _model.UnregisterContext(this);
        }
    }
}
