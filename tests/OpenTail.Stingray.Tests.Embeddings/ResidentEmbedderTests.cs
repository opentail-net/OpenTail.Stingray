using OpenTail.Stingray.Core.Embeddings;
using OpenTail.Stingray.Engine.Encoders;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Runtime behaviour of <see cref="ResidentEmbedder"/>: serialised calls in arrival order, cancellation of queued
/// callers, unload, and (when MiniLM is present) concurrent real requests agreeing with single calls.
/// </summary>
public sealed class ResidentEmbedderTests
{
    private sealed class FakePipeline : IEmbeddingPipeline
    {
        private int _active;
        public int MaxActive;
        public int Calls;
        public bool Disposed;
        public readonly ManualResetEventSlim Release = new(initialState: true);
        public readonly List<string> Order = [];

        public string ModelName => "fake";
        public int EmbeddingDimensions => 2;
        public PoolingType DefaultPooling => PoolingType.Mean;

        public EmbeddingResult Embed(EmbeddingRequest request)
        {
            int now = Interlocked.Increment(ref _active);
            int seen;
            while ((seen = Volatile.Read(ref MaxActive)) < now && Interlocked.CompareExchange(ref MaxActive, now, seen) != seen) { }
            Interlocked.Increment(ref Calls);
            lock (Order) Order.Add(request.Inputs[0]);
            Release.Wait();
            Interlocked.Decrement(ref _active);
            return new EmbeddingResult("fake",
                [.. request.Inputs.Select((t, i) => new EmbeddingData { Index = i, Vector = [t.Length, i] })], request.Inputs.Count, request.Inputs.Count);
        }

        public void Dispose() => Disposed = true;
    }

    private static EmbeddingRequest Req(string text) => new() { Inputs = [text] };

    [Fact]
    public async Task ConcurrentCallers_RunOneAtATime_AndEachGetsItsOwnResult()
    {
        var fake = new FakePipeline();
        using var embedder = new ResidentEmbedder(fake);
        var tasks = Enumerable.Range(1, 16).Select(n => embedder.EmbedAsync(Req(new string('x', n)), TestContext.Current.CancellationToken)).ToArray();
        var results = await Task.WhenAll(tasks);
        Assert.Equal(1, fake.MaxActive); // never two inside the encoder
        Assert.Equal(16, fake.Calls);
        for (int i = 0; i < 16; i++) Assert.Equal(i + 1, results[i].Data[0].Vector[0]); // own length, not a neighbour's
        Assert.Equal(0, embedder.InFlight);
    }

    [Fact]
    public async Task QueuedCaller_CancelsPromptly_WhileAnotherCallIsRunning_AndOthersStillComplete()
    {
        var fake = new FakePipeline();
        fake.Release.Reset(); // the first call holds the encoder
        using var embedder = new ResidentEmbedder(fake);
        var running = embedder.EmbedAsync(Req("first"), TestContext.Current.CancellationToken);
        while (fake.Calls == 0) await Task.Delay(5, TestContext.Current.CancellationToken);

        using var cts = new CancellationTokenSource();
        var cancelled = embedder.EmbedAsync(Req("cancelled"), cts.Token);
        var survivor = embedder.EmbedAsync(Req("survivor"), TestContext.Current.CancellationToken);
        await Task.Delay(50, TestContext.Current.CancellationToken);
        cts.Cancel();

        // The cancelled caller completes while the encoder is still busy.
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
        Assert.False(running.IsCompleted);

        fake.Release.Set();
        await running;
        await survivor;
        Assert.Equal(["first", "survivor"], fake.Order); // the cancelled request never reached the encoder
        Assert.Equal(0, embedder.InFlight);
    }

    [Fact]
    public async Task AlreadyCancelledToken_NeverStartsTheEncoder()
    {
        var fake = new FakePipeline();
        using var embedder = new ResidentEmbedder(fake);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => embedder.EmbedAsync(Req("x"), cts.Token));
        Assert.Equal(0, fake.Calls);
    }

    [Fact]
    public async Task Dispose_WaitsForTheRunningCall_ReleasesTheModel_AndRejectsLaterCalls()
    {
        var fake = new FakePipeline();
        fake.Release.Reset();
        var embedder = new ResidentEmbedder(fake);
        var running = embedder.EmbedAsync(Req("busy"), TestContext.Current.CancellationToken);
        while (fake.Calls == 0) await Task.Delay(5, TestContext.Current.CancellationToken);

        var dispose = Task.Run(embedder.Dispose, TestContext.Current.CancellationToken);
        await Task.Delay(100, TestContext.Current.CancellationToken);
        Assert.False(dispose.IsCompleted);
        Assert.False(fake.Disposed); // not freed under a running call

        fake.Release.Set();
        await running;
        await dispose;
        Assert.True(fake.Disposed);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => embedder.EmbedAsync(Req("late"), TestContext.Current.CancellationToken));
    }

    // ── Real model ───────────────────────────────────────────────────────────

    private static string? FindMiniLm() =>
        new[] { Environment.GetEnvironmentVariable("STINGRAY_TEST_MINILM_GGUF"), @"F:\_models\all-MiniLM-L6-v2-Q8_0.gguf", @"E:\_models\embeddings\minilm-q8.gguf" }
            .FirstOrDefault(p => p is { Length: > 0 } && File.Exists(p));

    [Fact]
    public async Task RealMiniLm_ConcurrentRequests_MatchSingleCalls_AndUnloadReloads()
    {
        string? gguf = FindMiniLm();
        Assert.SkipUnless(gguf != null, "all-MiniLM-L6-v2 Q8_0 GGUF not found");

        string[] texts = ["alpha beta", "the quick brown fox", "Ünïcödé café", "one", "embeddings are vectors", "last text here"];
        var embedder = EncoderPipelineFactory.GetResidentEmbedding(gguf!);
        try
        {
            var single = new float[texts.Length][];
            for (int i = 0; i < texts.Length; i++)
                single[i] = (await embedder.EmbedAsync(Req(texts[i]), TestContext.Current.CancellationToken)).Data[0].Vector;

            // 4 rounds of all texts at once, interleaved from many tasks.
            var jobs = Enumerable.Range(0, texts.Length * 4)
                .Select(n => embedder.EmbedAsync(Req(texts[n % texts.Length]), TestContext.Current.CancellationToken)).ToArray();
            var all = await Task.WhenAll(jobs);
            for (int n = 0; n < all.Length; n++)
                Assert.Equal(single[n % texts.Length], all[n].Data[0].Vector); // deterministic: bit-identical to the single call

            // A batch inside one request agrees with the single calls too.
            var batch = await embedder.EmbedAsync(new EmbeddingRequest { Inputs = texts }, TestContext.Current.CancellationToken);
            for (int i = 0; i < texts.Length; i++)
            {
                float dot = 0;
                for (int k = 0; k < single[i].Length; k++) dot += single[i][k] * batch.Data[i].Vector[k];
                Assert.InRange(dot, 0.9999f, 1.0001f);
            }

            Assert.True(EncoderPipelineFactory.UnloadEmbedding(gguf!));
            Assert.False(EncoderPipelineFactory.UnloadEmbedding(gguf!));
            await Assert.ThrowsAsync<ObjectDisposedException>(() => embedder.EmbedAsync(Req("x"), TestContext.Current.CancellationToken));

            var reloaded = EncoderPipelineFactory.GetResidentEmbedding(gguf!);
            Assert.NotSame(embedder, reloaded);
            var again = (await reloaded.EmbedAsync(Req(texts[0]), TestContext.Current.CancellationToken)).Data[0].Vector;
            Assert.Equal(single[0], again);
        }
        finally
        {
            EncoderPipelineFactory.UnloadEmbedding(gguf!);
        }
    }
}
