using System.Threading.Channels;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Target abstraction for <see cref="MoEPrefetcher"/>. Implemented by both
/// <see cref="ExpertSlotManager"/> (Vulkan) and <see cref="CudaExpertSlotManager"/>
/// (CUDA) so the same background prefetcher drives both backends.
/// </summary>
public interface IExpertPrefetchTarget
{
    /// <summary>Pre-load <paramref name="expertId"/> in <paramref name="layer"/> if not already resident.</summary>
    void Preload(int layer, int expertId);
}

/// <summary>
/// Background prefetcher for MoE expert weights.
/// After the router selects experts for the current token, callers enqueue
/// the predicted experts for the next token (or next layer). The background
/// worker calls <see cref="IExpertPrefetchTarget.Preload"/> so experts are
/// GPU-resident before they are needed, hiding upload latency.
/// </summary>
public sealed class MoEPrefetcher : IDisposable
{
    private readonly IExpertPrefetchTarget _slotManager;
    private readonly Channel<PrefetchBatch> _channel;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _worker;

    public MoEPrefetcher(IExpertPrefetchTarget slotManager, int queueDepth = 8)
    {
        _slotManager = slotManager;
        _channel = Channel.CreateBounded<PrefetchBatch>(new BoundedChannelOptions(queueDepth)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false
        });
        _worker = Task.Run(RunAsync);
    }

    /// <summary>
    /// Enqueue a prefetch request for <paramref name="expertIds"/> in
    /// <paramref name="layer"/>. Non-blocking: drops oldest if queue is full.
    /// </summary>
    public void EnqueuePrefetch(int layer, ReadOnlySpan<int> expertIds)
    {
        var batch = new PrefetchBatch(layer, expertIds.ToArray());
        _channel.Writer.TryWrite(batch); // DropOldest if full — safe to ignore
    }

    private async Task RunAsync()
    {
        try
        {
            await foreach (var batch in _channel.Reader.ReadAllAsync(_cts.Token))
            {
                foreach (int expertId in batch.ExpertIds)
                {
                    if (_cts.Token.IsCancellationRequested) return;
                    try
                    {
                        _slotManager.Preload(batch.Layer, expertId);
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException)
                    {
                        // A single Preload failure (transient cudaMalloc / upload
                        // fault / stale predicted expert) must not silently kill
                        // the worker: prefetch is best-effort, but losing it mid-run
                        // degrades throughput with no log to grep for.
                        Console.Error.WriteLine(
                            $"[MoEPrefetcher] Preload(layer={batch.Layer}, expert={expertId}) failed: {ex.Message}");
                    }
                }
            }
        }
        catch (OperationCanceledException) { }
    }

    /// <summary>
    /// How long <see cref="Dispose"/> waits for the worker. Cancellation stops the queue at the worker's next checkpoint,
    /// so at most one <see cref="IExpertPrefetchTarget.Preload"/> is still running; the caller frees the target as soon as
    /// Dispose returns, so the wait has to cover that call. The bound is only a backstop against a wedged upload.
    /// </summary>
    private static readonly TimeSpan DisposeDrainTimeout = TimeSpan.FromSeconds(30);

    public void Dispose()
    {
        _cts.Cancel();
        _channel.Writer.TryComplete();
        bool drained;
        try { drained = _worker.Wait(DisposeDrainTimeout); }
        catch (AggregateException) { drained = true; }   // the worker faulted, so it has stopped
        if (!drained)
        {
            // Keep the token source alive for the still-running worker; the target may be freed while this Preload runs.
            Console.Error.WriteLine(
                $"[MoEPrefetcher] worker still running {DisposeDrainTimeout.TotalSeconds:0} s after dispose; " +
                "a Preload may be using a target that is being freed.");
            return;
        }
        _cts.Dispose();
    }
}

/// <summary>A batch of experts to prefetch for a specific transformer layer.</summary>
public readonly record struct PrefetchBatch(int Layer, int[] ExpertIds);
