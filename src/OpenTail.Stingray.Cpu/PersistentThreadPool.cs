namespace OpenTail.Stingray.Cpu;

/// <summary>
/// A small persistent-worker-thread pool modeled directly on OpenBLAS's own thread server
/// (<c>examples/cpp/OpenBLAS/driver/others/blas_server_win32.c</c>, BSD-3-Clause, © 2009-2010
/// The University of Texas at Austin / The OpenBLAS Project — read directly from the vendored
/// source before writing this, not from memory). OpenBLAS spawns N threads ONCE
/// (<c>CreateThread</c> at <c>blas_thread_init</c>/<c>goto_set_num_threads</c>), each looping
/// forever, blocking on <c>WaitForMultipleObjects</c> for a queued work item, executing it, then
/// signaling a per-item completion event — idle workers are truly asleep (an OS wait, not a
/// spin), so they cost nothing and don't compete for cores/cache with active work.
///
/// This does the same thing in .NET terms — a real per-worker <see cref="AutoResetEvent"/> wait,
/// not a spin loop — as a static work-partition fork-join (equal contiguous chunks, one per
/// worker) rather than OpenBLAS's linked work queue, closer to how ggml's own thread pool
/// partitions a single op's iteration space.
///
/// Two other designs were tried and measured worse for this specific workload (see
/// docs/real-avx2-gemm-port-plan.md's "PersistentThreadPool experiment" section for full
/// numbers): a pure spin-wait version (busy-looping between calls, roughly tied with or worse
/// than plain <c>Parallel.For</c> — spinning workers burn a full core each even while "idle,"
/// self-competing with the active ones on a shared box), and a dynamic-chunk-claiming version
/// with caller-thread participation (all workers atomically claiming ranges from one shared
/// <c>Interlocked.Add</c> cursor instead of a static up-front partition — architecturally more
/// sophisticated and better suited to genuinely uneven workloads, but measured consistently
/// *worse* here: this kernel's 256 row-groups are near-identical in cost, so there was no
/// load-balancing gain to offset the shared-cache-line contention every worker incurred hitting
/// one claim counter). Static per-worker partitioning, with no field shared across workers at
/// dispatch time, wins for this specific near-uniform-cost workload.
///
/// NOT thread-safe for concurrent <see cref="For"/> calls from multiple threads at once (single
/// shared generation counter per worker, no call-level locking) — fine for a single-threaded
/// benchmark driver; would need a lock or per-call worker set before use from e.g. concurrent
/// request handling.
/// </summary>
internal static class PersistentThreadPool
{
    private sealed class Worker
    {
        public readonly AutoResetEvent WakeEvent = new(false);
        public Action<int, int>? Body;
        public int From;
        public int To;
    }

    private static readonly int s_poolSize = Environment.ProcessorCount;
    private static readonly Worker[] s_workers = BuildWorkers();
    private static int s_pendingCount;
    private static readonly ManualResetEventSlim s_doneEvent = new(false);

    private static Worker[] BuildWorkers()
    {
        var workers = new Worker[s_poolSize];
        for (int i = 0; i < s_poolSize; i++)
        {
            var w = new Worker();
            workers[i] = w;
            var t = new Thread(() => WorkerLoop(w)) { IsBackground = true, Name = "OpenTailPersistentWorker" };
            t.Start();
        }
        return workers;
    }

    private static void WorkerLoop(Worker w)
    {
        while (true)
        {
            w.WakeEvent.WaitOne(); // real OS block -- no CPU cost while idle, matching OpenBLAS's WaitForMultipleObjects.

            w.Body!(w.From, w.To);

            if (Interlocked.Decrement(ref s_pendingCount) == 0)
                s_doneEvent.Set();
        }
    }

    /// <summary>
    /// Static-partition fork-join over [0, count): splits into up to <see cref="Environment.ProcessorCount"/>
    /// equal contiguous chunks, one per persistent worker, and blocks until all chunks finish.
    /// </summary>
    internal static void For(int count, Action<int, int> body)
    {
        if (count <= 0) return;

        int workerCount = Math.Min(s_poolSize, count);
        if (workerCount <= 1) { body(0, count); return; }

        s_doneEvent.Reset();
        s_pendingCount = workerCount;

        int chunk = (count + workerCount - 1) / workerCount;
        for (int i = 0; i < workerCount; i++)
        {
            var w = s_workers[i];
            int from = i * chunk;
            int to = Math.Min(from + chunk, count);
            w.Body = body;
            w.From = from;
            w.To = to;
            w.WakeEvent.Set();
        }

        s_doneEvent.Wait();
    }
}
