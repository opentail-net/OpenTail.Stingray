// Ported from TensorSharp (https://github.com/zhongkaifu/TensorSharp), TensorSharp.Models/CpuWorkerPool.cs,
// Copyright (c) Zhongkai Fu, BSD 3-Clause License (see THIRD_PARTY_NOTICES.md). Adapted for OpenTail.Stingray:
// sized from SimdKernels.CpuThreads, STINGRAY_CPU_SPIN, nullable-annotated, no cancellation overload.

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Persistent spin-then-park worker pool for the CPU kernels: an alternative to one
/// <see cref="Parallel.For(int,int,ParallelOptions,Action{int})"/> per matvec, selected with
/// <c>STINGRAY_CPU_POOL=spin</c> (<see cref="SimdKernels.SpinPoolEnabled"/>).
///
/// <para>Why it might help: a decoded token issues a few hundred small matvecs, each its own fork/join.
/// The ThreadPool wakes parked threads through the kernel for each one, and on a tiny model
/// (SmolLM2-135M, a few microseconds of work per matvec) that wakeup is the same order as the work.
/// Workers here stay alive and spin on a generation counter, so submitting a job is one interlocked
/// write and the submitting thread works too. Blocks are claimed from an atomic counter. Workers spin
/// for a bounded interval (<c>STINGRAY_CPU_SPIN</c> rounds of <c>Thread.SpinWait(64)</c>, default 4096)
/// and only then park on a monitor, under the same lock the submitter pulses, so a wakeup cannot be
/// missed and an idle process stops burning cores.</para>
///
/// <para>Not the same experiment as the closed <see cref="PersistentThreadPool"/> one: that pool woke
/// workers through an OS event on every call, and its pure-spin variant spun without bound; both were
/// measured on SmolLM2-1.7B shapes (docs/done/perf-loop-progress.md, iteration 2).</para>
///
/// <para>One job at a time: a nested call runs inline on the calling thread. A call while another
/// thread's job is in flight runs inline through <see cref="For"/>, or returns false through
/// <see cref="TryFor"/> so the caller can use its own parallel loop (what SimdKernels does).</para>
///
/// <para>Measured 2026-10-02 (docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md §1): with
/// coarse blocks and the decode-path loops routed here, dense-model decode +19..27%, Mistral-7B tie,
/// MoE prefill -18..19%. So it is opt-in, and prefill/batched kernels never use it.</para>
/// </summary>
internal sealed class SpinParkWorkerPool : IDisposable
{
    private static readonly int SpinsBeforePark =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_CPU_SPIN"), out int v) && v >= 0 ? v : 4096;

    // Set on any thread inside a job, the submitter included: Monitor is recursive, so a nested For()
    // from the submitting thread would otherwise re-enter _submitLock and clobber the job in flight.
    [ThreadStatic] private static bool t_inJob;

    private readonly object _submitLock = new();
    private readonly object _sleepLock = new();
    private readonly int _workers;
    private readonly Thread[] _threads;

    private Action<int>? _body;
    private int _blockCount;
    private int _nextBlock;
    private int _activeCount;
    private int _parked;
    private int _ready;
    private bool _started;   // guarded by _submitLock
    private long _generation;
    private Exception? _error;
    private volatile bool _shutdown;

    /// <param name="totalThreads">Threads that run a job, including the submitting one.</param>
    internal SpinParkWorkerPool(int totalThreads)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(totalThreads, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(totalThreads, 512);
        _workers = totalThreads - 1;
        _threads = new Thread[_workers];
        for (int i = 0; i < _workers; i++)
        {
            var t = new Thread(WorkerLoop, 512 * 1024) { IsBackground = true, Name = "stingray-cpu-" + i };
            _threads[i] = t;
            t.Start();
        }
        // The workers are not waited for here (the handshake happens on the first For()): a worker
        // that touches this type's statics while a static initializer is still running this
        // constructor would block on it.
    }

    /// <summary>Threads that run a job, including the submitting one.</summary>
    public int ThreadCount => _workers + 1;

    /// <summary>Run <paramref name="body"/> for every block in [0, <paramref name="blockCount"/>) across
    /// this thread and the pool's; returns once all have completed. A failure is rethrown as an
    /// <see cref="AggregateException"/>. When another thread's job is in flight this runs inline on the
    /// calling thread; <see cref="TryFor"/> lets a caller fall back to its own parallel loop instead.</summary>
    public void For(int blockCount, Action<int> body)
    {
        if (!TryFor(blockCount, body))
            RunInline(blockCount, body);
    }

    /// <summary>
    /// <see cref="For"/>, except that when another thread's job is already in flight it returns false
    /// without running anything, so a concurrent caller (a second request on a server) can use
    /// <see cref="Parallel.For(int,int,Action{int})"/> instead of being serialized onto one thread.
    /// Nested calls from inside a job, and single-block calls, still run inline and return true.
    /// </summary>
    public bool TryFor(int blockCount, Action<int> body)
    {
        ObjectDisposedException.ThrowIf(_shutdown, this);
        if (blockCount <= 0) return true;

        if (blockCount == 1 || _workers == 0 || t_inJob)
        {
            RunInline(blockCount, body);
            return true;
        }
        if (!Monitor.TryEnter(_submitLock)) return false;

        t_inJob = true;
        try
        {
            ObjectDisposedException.ThrowIf(_shutdown, this);
            // Every worker must have latched a generation before the first job is published; one
            // that started late would latch the in-flight generation, skip the job and never
            // decrement the completion count, hanging the submitter.
            if (!_started)
            {
                var startSpin = new SpinWait();
                while (Volatile.Read(ref _ready) != _workers) startSpin.SpinOnce(-1);
                _started = true;
            }

            _body = body;
            _blockCount = blockCount;
            _error = null;
            Volatile.Write(ref _nextBlock, 0);
            Volatile.Write(ref _activeCount, _workers);

            // Full fence: publishes the fields above to every worker that then observes the new generation.
            Interlocked.Increment(ref _generation);

            // Only pay for a wakeup when someone is parked. A worker publishes _parked before
            // re-checking the generation, so one this misses is guaranteed to see the new generation.
            if (Volatile.Read(ref _parked) > 0)
            {
                lock (_sleepLock) Monitor.PulseAll(_sleepLock);
            }

            RunBlocks();

            var spin = new SpinWait();
            while (Volatile.Read(ref _activeCount) != 0) spin.SpinOnce(-1);

            _body = null;
            Exception? error = _error;
            if (error != null)
            {
                _error = null;
                throw new AggregateException(error);
            }
        }
        finally
        {
            t_inJob = false;
            Monitor.Exit(_submitLock);
        }
        return true;
    }

    private static void RunInline(int blockCount, Action<int> body)
    {
        bool wasInJob = t_inJob;
        t_inJob = true;
        try { for (int i = 0; i < blockCount; i++) body(i); }
        finally { t_inJob = wasInJob; }
    }

    public void Dispose()
    {
        if (t_inJob) throw new InvalidOperationException("Cannot dispose a CPU worker pool from a running pool job.");
        lock (_submitLock)
        {
            if (_shutdown) return;
            _shutdown = true;
            lock (_sleepLock) Monitor.PulseAll(_sleepLock);
            foreach (var thread in _threads) thread.Join();
        }
    }

    private void RunBlocks()
    {
        Action<int> body = _body!;
        int count = _blockCount;
        try
        {
            while (true)
            {
                int i = Interlocked.Increment(ref _nextBlock) - 1;
                if (i >= count) return;
                body(i);
            }
        }
        catch (Exception ex)
        {
            // Claim the rest of the range so the other threads wind down; hand the first failure back.
            Interlocked.Exchange(ref _nextBlock, count);
            Interlocked.CompareExchange(ref _error, ex, null);
        }
    }

    private void WorkerLoop()
    {
        long seen = Volatile.Read(ref _generation);
        Interlocked.Increment(ref _ready);
        t_inJob = true;
        int spins = 0;

        while (!_shutdown)
        {
            long gen = Volatile.Read(ref _generation);
            if (gen != seen)
            {
                seen = gen;
                RunBlocks();
                Interlocked.Decrement(ref _activeCount);
                spins = 0;
                continue;
            }

            if (++spins < SpinsBeforePark)
            {
                Thread.SpinWait(64);
                continue;
            }

            // Park under the lock the submitter pulses, so check-then-wait cannot race a publish.
            Interlocked.Increment(ref _parked);
            lock (_sleepLock)
            {
                if (Volatile.Read(ref _generation) == seen && !_shutdown)
                    Monitor.Wait(_sleepLock);
            }
            Interlocked.Decrement(ref _parked);
            spins = 0;
        }
    }
}
