using System.Threading.Channels;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Pipeline;

/// <summary>
/// Asynchronous weight prefetcher. Analyses the upcoming layer schedule and issues DMA transfers
/// before weights are needed, hiding I/O latency.
///
/// <para><b>Internal alongside <see cref="MemoryHierarchy"/>.</b> Every request it dequeues calls
/// <c>MemoryHierarchy.PromoteToGpuAsync</c>, which throws <see cref="NotImplementedException"/>, so
/// this cannot function either — it was public only because the type it depends on was. The
/// implemented expert prefetcher is <c>MoEPrefetcher</c> in OpenTail.Stingray.Engine.</para>
/// </summary>
internal sealed class Prefetcher : IDisposable
{
    private readonly MemoryHierarchy _memory;
    private readonly Channel<PrefetchRequest> _queue;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _worker;

    public Prefetcher(MemoryHierarchy memory, int queueDepth = 4)
    {
        _memory = memory;
        _queue = System.Threading.Channels.Channel.CreateBounded<PrefetchRequest>(queueDepth);
        _worker = Task.Run(RunAsync);
    }

    public ValueTask EnqueueAsync(PrefetchRequest request, CancellationToken ct = default) =>
        _queue.Writer.WriteAsync(request, ct);

    /// <summary>
    /// Completion of the background prefetch worker. A fault here is also rethrown by
    /// <see cref="Dispose"/>, while normal disposal completes successfully after cancellation.
    /// </summary>
    public Task Completion => _worker;

    private async Task RunAsync()
    {
        try
        {
            await foreach (var req in _queue.Reader.ReadAllAsync(_cts.Token))
            {
                await _memory.PromoteToGpuAsync(req.TensorName, _cts.Token);
            }
        }
        catch (OperationCanceledException) when (_cts.IsCancellationRequested)
        {
            // Expected shutdown path — Dispose() cancels the token to stop the loop.
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _worker.Wait();
        _cts.Dispose();
    }
}

internal readonly record struct PrefetchRequest(string TensorName, int Priority = 0);
