#nullable enable

using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

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

    /// <summary>
    /// Gets the underlying <see cref="IInferenceEngine"/> powering this context.
    /// Lazily initialized with default CPU forward pass if not explicitly configured.
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
        _tokenizer = GgufTokenizer.FromGgufModel(model.Gguf);
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
        }
    }

    private IInferenceEngine CreateDefaultEngine()
    {
        var cpuBackend = new CpuBackend();
        var fwd = new ForwardPass(_model.Gguf, cpuBackend, _model.Hyperparams, maxContextLength: ContextSize);
        var (thinkTokenId, endThinkTokenId) = _tokenizer.ReasoningTokens;
        return new InferenceEngine(fwd, _tokenizer, _model.Architecture, thinkTokenId, endThinkTokenId, owned: [fwd, cpuBackend]);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ThrowIfDisposed();
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
