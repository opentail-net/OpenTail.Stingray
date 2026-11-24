using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// <see cref="MoEPrefetcher.Dispose"/> must not return while a <see cref="IExpertPrefetchTarget.Preload"/> is still running:
/// the caller frees the target (expert slots and their device buffers) straight after Dispose. A Preload that outlasts a short
/// fixed wait used to let Dispose return with the call still in flight.
/// </summary>
public sealed class MoEPrefetcherDisposeTests
{
    /// <summary>A target whose first Preload blocks until <see cref="Release"/>; later calls return at once.</summary>
    private sealed class BlockingTarget : IExpertPrefetchTarget
    {
        private readonly ManualResetEventSlim _release = new(false);
        private int _calls;
        private int _inFlight;

        public int Calls => Volatile.Read(ref _calls);
        public bool InFlight => Volatile.Read(ref _inFlight) != 0;
        public void Release() => _release.Set();

        public void Preload(int layer, int expertId)
        {
            int call = Interlocked.Increment(ref _calls);
            Interlocked.Exchange(ref _inFlight, 1);
            try
            {
                if (call == 1) _release.Wait(TimeSpan.FromSeconds(30));
            }
            finally
            {
                Interlocked.Exchange(ref _inFlight, 0);
            }
        }
    }

    [Fact]
    public async Task Dispose_WaitsForAPreloadThatIsStillRunning_BeforeItReturns()
    {
        var target = new BlockingTarget();
        var prefetcher = new MoEPrefetcher(target);
        prefetcher.EnqueuePrefetch(layer: 0, expertIds: [1, 2, 3]);

        // Wait until the first Preload is inside the target.
        var started = System.Diagnostics.Stopwatch.StartNew();
        while (!target.InFlight && started.Elapsed < TimeSpan.FromSeconds(10)) Thread.Sleep(5);
        Assert.True(target.InFlight, "the first Preload never started");

        // The blocked Preload finishes 3 s from now: longer than the 2 s wait Dispose used to have.
        var unblock = Task.Delay(TimeSpan.FromSeconds(3)).ContinueWith(_ => target.Release());

        try
        {
            prefetcher.Dispose();
            bool inFlightAtReturn = target.InFlight;
            Assert.False(inFlightAtReturn,
                "Dispose returned while a Preload was still running; the caller would free the target under it.");

            // Cancellation stops the queue: the batch's other experts are not preloaded after Dispose.
            Assert.Equal(1, target.Calls);
        }
        finally
        {
            target.Release();
            await unblock;
        }
    }
}
