using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// One loaded embedding pipeline serving many callers. The encoder forward pass is not re-entrant, so calls run
/// one at a time in arrival order, but waiting never blocks a thread and a caller that goes away while queued
/// leaves the queue at once. A call that has started runs to completion: <see cref="IEmbeddingPipeline.Embed"/>
/// takes no cancellation token, so cancelling it would only abandon the result.
/// </summary>
public sealed class ResidentEmbedder : IDisposable
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private volatile bool _disposed;

    /// <summary>Wraps <paramref name="pipeline"/>; this object owns it from now on.</summary>
    public ResidentEmbedder(IEmbeddingPipeline pipeline) => Pipeline = pipeline;

    /// <summary>The wrapped pipeline. Use <see cref="EmbedAsync"/> rather than calling it directly while shared.</summary>
    public IEmbeddingPipeline Pipeline { get; }

    /// <summary>Calls waiting for or holding the encoder.</summary>
    public int InFlight => Volatile.Read(ref _inFlight);
    private int _inFlight;

    /// <exception cref="OperationCanceledException">The token fired before the call started.</exception>
    /// <exception cref="ObjectDisposedException">The model was unloaded.</exception>
    public async Task<EmbeddingResult> EmbedAsync(EmbeddingRequest request, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Interlocked.Increment(ref _inFlight);
        try
        {
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                cancellationToken.ThrowIfCancellationRequested();
                return await Task.Run(() => Pipeline.Embed(request), CancellationToken.None).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }
        finally
        {
            Interlocked.Decrement(ref _inFlight);
        }
    }

    /// <summary>Waits for the call in progress, then releases the model. Queued callers get <see cref="ObjectDisposedException"/>.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gate.Wait();
        try { Pipeline.Dispose(); }
        finally { _gate.Release(); }
    }
}
