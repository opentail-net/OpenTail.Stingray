namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Contract for <see cref="SpinParkWorkerPool"/> (the <c>STINGRAY_CPU_POOL=spin</c> scheduler,
/// ported from TensorSharp's CpuWorkerPool) and for <see cref="SimdKernels.KernelFor"/> routing
/// through it.
///
/// <para>The pool is a scheduling change only: each block computes what it did under
/// <c>Parallel.For</c>. So the kernel tests demand bit equality against the pool-off arm, and the
/// pool tests demand that every block runs exactly once whatever the width, nesting, concurrency or
/// park state. A pool that skips or repeats a block produces plausible numbers in the wrong rows, so
/// "close enough" is not a pass.</para>
/// </summary>
public sealed unsafe class SpinParkWorkerPoolTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(2, 1000)]
    [InlineData(4, 3)]
    [InlineData(8, 7)]
    [InlineData(8, 8)]
    [InlineData(8, 1023)]
    public void For_RunsEveryBlockExactlyOnce(int threads, int blocks)
    {
        using var pool = new SpinParkWorkerPool(threads);
        Assert.Equal(threads, pool.ThreadCount);
        for (int round = 0; round < 50; round++)
        {
            var hits = new int[blocks];
            pool.For(blocks, i => Interlocked.Increment(ref hits[i]));
            Assert.All(hits, h => Assert.Equal(1, h));
        }
    }

    [Fact]
    public void For_ZeroBlocks_DoesNothing()
    {
        using var pool = new SpinParkWorkerPool(4);
        pool.For(0, _ => throw new InvalidOperationException("must not run"));
    }

    [Fact]
    public void NestedFor_RunsInline_AndCompletes()
    {
        using var pool = new SpinParkWorkerPool(4);
        var hits = new int[16 * 16];
        pool.For(16, outer => pool.For(16, inner => Interlocked.Increment(ref hits[outer * 16 + inner])));
        Assert.All(hits, h => Assert.Equal(1, h));
    }

    [Fact]
    public void Exception_IsRethrown_AndPoolStaysUsable()
    {
        using var pool = new SpinParkWorkerPool(4);
        var ex = Assert.Throws<AggregateException>(() =>
            pool.For(100, i => { if (i == 37) throw new InvalidOperationException("block 37"); }));
        Assert.Contains(ex.InnerExceptions, e => e.Message == "block 37");

        var hits = new int[100];
        pool.For(100, i => Interlocked.Increment(ref hits[i]));
        Assert.All(hits, h => Assert.Equal(1, h));
    }

    [Fact]
    public void ConcurrentSubmitters_AllComplete()
    {
        // One job at a time: a second submitter runs its job inline. Every job must still finish
        // with every block run exactly once.
        using var pool = new SpinParkWorkerPool(4);
        const int submitters = 6, rounds = 40, blocks = 200;
        var failures = 0;
        var threads = Enumerable.Range(0, submitters).Select(_ => new Thread(() =>
        {
            for (int r = 0; r < rounds; r++)
            {
                var hits = new int[blocks];
                pool.For(blocks, i => Interlocked.Increment(ref hits[i]));
                if (hits.Any(h => h != 1)) Interlocked.Increment(ref failures);
            }
        })).ToArray();
        foreach (var t in threads) t.Start();
        foreach (var t in threads) Assert.True(t.Join(TimeSpan.FromSeconds(60)), "submitter hung");
        Assert.Equal(0, failures);
    }

    [Fact]
    public void ParallelForHelpers_EmptyRange_AreNoOps_OnBothSchedulers()
    {
        // Parallel.For treats from == to as a no-op; the spin-pool path must too (it once divided by a
        // zero block count here, caught by RepackedGemmPath2Tests at batch sizes that produce no tiles).
        bool prev = SimdKernels.SpinPoolEnabled;
        try
        {
            foreach (bool spin in new[] { false, true })
            {
                SimdKernels.SpinPoolEnabled = spin;
                SimdKernels.ParallelForCapped(5, 5, _ => throw new InvalidOperationException("ran"));
                SimdKernels.ParallelForUncapped(0, 0, _ => throw new InvalidOperationException("ran"));
            }
        }
        finally { SimdKernels.SpinPoolEnabled = prev; }
    }

    [Fact]
    public void ParallelForHelpers_SpinPool_RunEveryIndexOnce_FromNonZeroStart()
    {
        bool prev = SimdKernels.SpinPoolEnabled;
        SimdKernels.SpinPoolEnabled = true;
        try
        {
            var hits = new int[1000];
            SimdKernels.ParallelForCapped(17, 1000, i => Interlocked.Increment(ref hits[i]));
            SimdKernels.ParallelForUncapped(0, 17, i => Interlocked.Increment(ref hits[i]));
            Assert.All(hits, h => Assert.Equal(1, h));
        }
        finally { SimdKernels.SpinPoolEnabled = prev; }
    }

    [Fact]
    public void TryFor_ReturnsFalseWhileAnotherThreadsJobIsInFlight()
    {
        // KernelFor relies on this to send a concurrent request's matvec to Parallel.For instead of
        // serializing it onto one thread.
        using var pool = new SpinParkWorkerPool(4);
        using var started = new ManualResetEventSlim(false);
        using var release = new ManualResetEventSlim(false);
        var owner = new Thread(() => pool.For(4, i => { started.Set(); release.Wait(TimeSpan.FromSeconds(30)); }));
        owner.Start();
        Assert.True(started.Wait(TimeSpan.FromSeconds(30)), "owner job never started");

        bool ran = false;
        Assert.False(pool.TryFor(8, _ => ran = true));
        Assert.False(ran);

        release.Set();
        Assert.True(owner.Join(TimeSpan.FromSeconds(30)), "owner job hung");
        var hits = new int[8];
        Assert.True(pool.TryFor(8, i => Interlocked.Increment(ref hits[i])));
        Assert.All(hits, h => Assert.Equal(1, h));
    }

    [Fact]
    public void ConcurrentMatVec_OnSpinPool_StaysBitIdentical()
    {
        const int rows = 1024, cols = 512;
        var rng = new Random(11);
        var w = BuildQ4K(rows, cols, rng);
        var x = new float[cols];
        for (int i = 0; i < cols; i++) x[i] = (float)((rng.NextDouble() - 0.5) * 4);
        var reference = RunMatVec(w, x, rows, cols, DType.Q4_K, spin: false);

        bool prev = SimdKernels.SpinPoolEnabled;
        SimdKernels.SpinPoolEnabled = true;
        try
        {
            int mismatches = 0;
            var threads = Enumerable.Range(0, 4).Select(_ => new Thread(() =>
            {
                for (int r = 0; r < 50; r++)
                {
                    var y = new float[rows];
                    fixed (byte* wp = w) fixed (float* xp = x) fixed (float* yp = y)
                        SimdKernels.MatVec(yp, wp, xp, rows, cols, DType.Q4_K);
                    for (int i = 0; i < rows; i++)
                        if (BitConverter.SingleToInt32Bits(y[i]) != BitConverter.SingleToInt32Bits(reference[i]))
                        { Interlocked.Increment(ref mismatches); break; }
                }
            })).ToArray();
            foreach (var t in threads) t.Start();
            foreach (var t in threads) Assert.True(t.Join(TimeSpan.FromSeconds(60)), "matvec thread hung");
            Assert.Equal(0, mismatches);
        }
        finally { SimdKernels.SpinPoolEnabled = prev; }
    }

    [Fact]
    public void ParkedWorkers_WakeForTheNextJob()
    {
        // Idle long enough for every worker to exhaust its spin budget and park, then submit: the job
        // must complete (a missed wakeup would leave the submitter spinning on _activeCount forever).
        using var pool = new SpinParkWorkerPool(4);
        pool.For(8, _ => { });
        for (int round = 0; round < 3; round++)
        {
            Thread.Sleep(300);
            var hits = new int[64];
            var job = new Thread(() => pool.For(64, i => Interlocked.Increment(ref hits[i])));
            job.Start();
            Assert.True(job.Join(TimeSpan.FromSeconds(30)), "job after park hung");
            Assert.All(hits, h => Assert.Equal(1, h));
        }
    }

    [Fact]
    public void Dispose_JoinsWorkers_AndForAfterwardThrows()
    {
        var pool = new SpinParkWorkerPool(4);
        pool.For(10, _ => { });
        pool.Dispose();
        pool.Dispose(); // idempotent
        Assert.Throws<ObjectDisposedException>(() => pool.For(10, _ => { }));
    }

    // ── KernelFor routing: real kernels, pool on vs off, bit equality ──

    private static byte[] BuildQ4K(int rows, int cols, Random rng)
    {
        int bpr = cols / 256 * 144;
        var b = new byte[rows * bpr];
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < cols / 256; k++)
            {
                int o = r * bpr + k * 144;
                ushort d = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 0.05 + 0.005));
                ushort m = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 0.03 + 0.005));
                b[o] = (byte)d; b[o + 1] = (byte)(d >> 8); b[o + 2] = (byte)m; b[o + 3] = (byte)(m >> 8);
                for (int i = 4; i < 144; i++) b[o + i] = (byte)rng.Next(256);
            }
        return b;
    }

    private static byte[] BuildQ8_0(int rows, int cols, Random rng)
    {
        int bpr = cols / 32 * 34;
        var b = new byte[rows * bpr];
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < cols / 32; k++)
            {
                int o = r * bpr + k * 34;
                ushort d = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 0.02 + 0.001));
                b[o] = (byte)d; b[o + 1] = (byte)(d >> 8);
                for (int i = 2; i < 34; i++) b[o + i] = (byte)rng.Next(256);
            }
        return b;
    }

    private static byte[] BuildF32(int rows, int cols, Random rng)
    {
        var f = new float[rows * cols];
        for (int i = 0; i < f.Length; i++) f[i] = (float)((rng.NextDouble() - 0.5) * 0.5);
        var b = new byte[f.Length * 4];
        Buffer.BlockCopy(f, 0, b, 0, b.Length);
        return b;
    }

    private static float[] RunMatVec(byte[] w, float[] x, int rows, int cols, DType dt, bool spin)
    {
        bool prev = SimdKernels.SpinPoolEnabled;
        SimdKernels.SpinPoolEnabled = spin;
        try
        {
            var y = new float[rows];
            fixed (byte* wp = w) fixed (float* xp = x) fixed (float* yp = y)
                SimdKernels.MatVec(yp, wp, xp, rows, cols, dt);
            return y;
        }
        finally { SimdKernels.SpinPoolEnabled = prev; }
    }

    [Theory]
    [InlineData(DType.Q4_K, 512, 1024)]
    [InlineData(DType.Q4_K, 2049, 512)]
    [InlineData(DType.Q8_0, 576, 576)]
    [InlineData(DType.Q8_0, 1537, 256)]
    [InlineData(DType.Float32, 300, 128)]
    public void MatVec_SpinPool_IsBitIdenticalToParallelFor(DType dt, int rows, int cols)
    {
        var rng = new Random(rows * 31 + cols);
        byte[] w = dt switch
        {
            DType.Q4_K => BuildQ4K(rows, cols, rng),
            DType.Q8_0 => BuildQ8_0(rows, cols, rng),
            _ => BuildF32(rows, cols, rng),
        };
        var x = new float[cols];
        for (int i = 0; i < cols; i++) x[i] = (float)((rng.NextDouble() - 0.5) * 4);

        var reference = RunMatVec(w, x, rows, cols, dt, spin: false);
        long before = Interlocked.Read(ref SimdKernels.SpinPoolDispatches);
        var spun = RunMatVec(w, x, rows, cols, dt, spin: true);
        Assert.True(Interlocked.Read(ref SimdKernels.SpinPoolDispatches) > before,
            "the spin pool was never dispatched: the A/B arm did not exercise it");

        for (int i = 0; i < rows; i++)
            Assert.True(BitConverter.SingleToInt32Bits(reference[i]) == BitConverter.SingleToInt32Bits(spun[i]),
                $"row {i}: {reference[i]:R} (Parallel.For) vs {spun[i]:R} (spin pool)");
    }

    [Fact]
    public void MatVecDual_SpinPool_IsBitIdenticalToParallelFor()
    {
        const int rows = 1024, cols = 512;
        var rng = new Random(7);
        var w1 = BuildQ4K(rows, cols, rng); var w2 = BuildQ4K(rows, cols, rng);
        var x = new float[cols];
        for (int i = 0; i < cols; i++) x[i] = (float)((rng.NextDouble() - 0.5) * 4);

        float[] Run(bool spin)
        {
            bool prev = SimdKernels.SpinPoolEnabled;
            SimdKernels.SpinPoolEnabled = spin;
            try
            {
                var y = new float[2 * rows];
                fixed (byte* a = w1) fixed (byte* b = w2) fixed (float* xp = x) fixed (float* yp = y)
                    SimdKernels.MatVecDual(yp, a, yp + rows, b, xp, rows, cols, DType.Q4_K, DType.Q4_K);
                return y;
            }
            finally { SimdKernels.SpinPoolEnabled = prev; }
        }

        var reference = Run(false);
        var spun = Run(true);
        for (int i = 0; i < reference.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(reference[i]), BitConverter.SingleToInt32Bits(spun[i]));
    }
}
