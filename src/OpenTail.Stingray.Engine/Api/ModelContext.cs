#nullable enable

using OpenTail.Stingray.Core;

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

    internal ModelContext(Model model, ContextParams contextParams)
    {
        _model = model;
        _params = contextParams;
        _tokenizer = GgufTokenizer.FromGgufModel(model.Gguf);
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
            _model.UnregisterContext(this);
        }
    }
}
