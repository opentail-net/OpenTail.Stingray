namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Base for every real-model test class in this project. Skipped by default so local iteration
/// only pays for classes that build entirely synthetic weights; a real checkpoint won't load
/// unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c> is set (before this, most classes here gated only on
/// checkpoint-path presence, so they'd silently no-op rather than actually skip -- this makes that
/// gate explicit and uniform, matching Audio/ForwardPass/Vulkan/Sessions).
///
/// <para>Also holds a machine-wide, cross-process mutex for the whole lifetime of the test class:
/// two heavy-test processes launched independently (two manual `dotnet test` runs in separate
/// terminals, or a CI matrix running Tests.Audio and Tests.Diffusion concurrently) can each hold
/// 10+ GB of real weights at once with nothing stopping them. The mutex makes any heavy-test
/// process across the whole machine -- this project or any of its siblings that use the same mutex
/// name -- wait its turn instead of piling up memory concurrently.</para>
/// </summary>
public abstract class HeavyTestBase : IDisposable
{
    private static readonly Mutex Gate = new(initiallyOwned: false, @"Global\OpenTailStingray.HeavyTests");
    private readonly ManualResetEventSlim _releaseGate = new(false);
    private readonly Thread? _gateOwnerThread;

    protected HeavyTestBase()
    {
        Assert.SkipUnless(
            Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "Heavy/serial suite skipped by default for fast local iteration. Set STINGRAY_RUN_HEAVY_TESTS=1 to run it.");

        // A Mutex is thread-affine. Async test methods resume on another worker thread, so the
        // test instance cannot release a mutex acquired on its constructor thread. Keep a small
        // owner thread alive for the class lifetime and signal it from Dispose instead.
        using var acquired = new ManualResetEventSlim(false);
        bool gateOwned = false;
        _gateOwnerThread = new Thread(() =>
        {
            bool ownsGate;
            // An abandoned mutex means the previous process died; this thread now owns it.
            try { ownsGate = Gate.WaitOne(TimeSpan.FromHours(2)); }
            catch (AbandonedMutexException) { ownsGate = true; }

            gateOwned = ownsGate;
            acquired.Set();
            if (!ownsGate)
                return;

            _releaseGate.Wait();
            Gate.ReleaseMutex();
        })
        {
            IsBackground = true,
            Name = "OpenTail Stingray heavy-test gate owner",
        };
        _gateOwnerThread.Start();
        acquired.Wait();

        // Fail closed: a timed-out wait means ANOTHER heavy-test process still holds the machine-wide gate. Carrying on would run two
        // real-weight suites at once, which is exactly what the gate exists to prevent.
        if (!gateOwned)
        {
            _gateOwnerThread.Join();
            _releaseGate.Dispose();
            throw new TimeoutException("Timed out after 2 hours waiting for the machine-wide heavy-test gate (Global\\OpenTailStingray.HeavyTests); another heavy-test process is still running.");
        }
    }

    public virtual void Dispose()
    {
        _releaseGate.Set();
        _gateOwnerThread?.Join();
        _releaseGate.Dispose();
        GC.SuppressFinalize(this);
    }
}
