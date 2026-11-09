using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 2 (Qwen 3.8 plan): on a real qwen4exp request, a cancelled request and a finished request each dispose their
/// sequence cache and release their KV reservation. Real checkpoint; skipped when it is absent.
/// </summary>
public sealed class Qwen4ExpCancellationProbeTests
{
    private const string Prompt = "The capital of France is";

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        Qwen4ExpTestSupport.TrunkFileName);

    [Fact]
    public async Task CancelledRealRequest_DisposesItsCache_AndReleasesItsReservation()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var fixture = QwenProbeFixture.Open(path!);
        using var engine = new ContinuousBatchingEngine(fixture.Probe, fixture.Tokenizer, "qwen4exp-probe", maxBatchSize: 2);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 200 };

        using var cancel = new CancellationTokenSource(TimeSpan.FromMinutes(20));
        int seen = 0;
        try
        {
            await foreach (var _ in engine.GenerateChunksAsync(Prompt, sp, cancel.Token))
            {
                if (++seen == 2) cancel.Cancel();
            }
        }
        catch (OperationCanceledException) { }

        await WaitForReleaseAsync(fixture.Probe, engine);
        Assert.True(fixture.Probe.Created > 0, "The request never created a cache, so the probe tests nothing.");
        Assert.Equal(fixture.Probe.Created, fixture.Probe.Disposed);
        Assert.Equal(0, engine.CommittedKvTokens);
    }

    [Fact]
    public async Task FinishedRealRequest_DisposesItsCache_AndReleasesItsReservation()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var fixture = QwenProbeFixture.Open(path!);
        using var engine = new ContinuousBatchingEngine(fixture.Probe, fixture.Tokenizer, "qwen4exp-probe", maxBatchSize: 2);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        var sb = new StringBuilder();
        await foreach (var chunk in engine.GenerateAsync(Prompt, sp, cts.Token)) sb.Append(chunk);

        await WaitForReleaseAsync(fixture.Probe, engine);
        Assert.True(fixture.Probe.Created > 0, "The request never created a cache, so the probe tests nothing.");
        Assert.Equal(fixture.Probe.Created, fixture.Probe.Disposed);
        Assert.Equal(0, engine.CommittedKvTokens);
    }

    /// <summary>A poll with a timeout: a slow release passes, and a leak fails at the timeout rather than hanging.</summary>
    private static async Task WaitForReleaseAsync(CountingBatchedPass probe, ContinuousBatchingEngine engine)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(60);
        while (DateTime.UtcNow < deadline
               && !(probe.Created == probe.Disposed && engine.CommittedKvTokens == 0))
            await Task.Delay(20);
    }

    /// <summary>The real Qwen pass, loaded once per test, with the counting probe wrapped around it.</summary>
    private sealed class QwenProbeFixture : IDisposable
    {
        private readonly GgufModel _model;
        private readonly Qwen4ExpForwardPass _pass;
        public CountingBatchedPass Probe { get; }
        public ITokenizer Tokenizer { get; }

        private QwenProbeFixture(GgufModel model, Qwen4ExpForwardPass pass, Qwen4ExpHyperparams hp, ITokenizer tokenizer, int totalLayers)
        {
            _model = model;
            _pass = pass;
            Tokenizer = tokenizer;
            Probe = new CountingBatchedPass(new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers));
        }

        public static QwenProbeFixture Open(string path)
        {
            var model = GgufModel.Open(path);
            int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
            var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
            var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
            var pass = new Qwen4ExpForwardPass(model, hp, tensors);
            return new QwenProbeFixture(model, pass, hp, GgufTokenizer.FromGgufModel(model), totalLayers);
        }

        public void Dispose()
        {
            _pass.Dispose();
            _model.Dispose();
        }
    }

    /// <summary>
    /// Wraps a batched pass and hands out counting caches. It delegates every call, unwrapping its caches first, so the
    /// real Qwen pass runs unchanged. It does not implement the prefix-cache interface, so the engine runs without one.
    /// </summary>
    internal sealed class CountingBatchedPass(IBatchedForwardPass inner) : IBatchedForwardPass
    {
        private int _created, _disposed;
        public int Created => Volatile.Read(ref _created);
        public int Disposed => Volatile.Read(ref _disposed);

        private sealed class CountingCache(ISequenceKvCache inner, CountingBatchedPass owner) : ISequenceKvCache
        {
            public ISequenceKvCache Inner { get; } = inner;

            public void Dispose()
            {
                Inner.Dispose();
                Interlocked.Increment(ref owner._disposed);
            }
        }

        private static ISequenceKvCache[] Unwrap(ISequenceKvCache[] caches)
        {
            var result = new ISequenceKvCache[caches.Length];
            for (int i = 0; i < caches.Length; i++) result[i] = ((CountingCache)caches[i]).Inner;
            return result;
        }

        public bool SnapKvEnabled => inner.SnapKvEnabled;
        public long KvBytesPerToken => inner.KvBytesPerToken;
        public long FixedBytesPerSequence => inner.FixedBytesPerSequence;
        public bool KeepsTurnSnapshot => inner.KeepsTurnSnapshot;
        public int MaxSeqLen => inner.MaxSeqLen;
        public bool PrefillDequantCacheActive => inner.PrefillDequantCacheActive;
        public bool SupportsBatchedGpuArgmax => inner.SupportsBatchedGpuArgmax;
        public void BindToCurrentThread() => inner.BindToCurrentThread();

        public ISequenceKvCache CreateCache()
        {
            Interlocked.Increment(ref _created);
            return new CountingCache(inner.CreateCache(), this);
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0) =>
            inner.PrefillWithCache(tokens, ((CountingCache)cache).Inner, startPos);

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            inner.PrefillPackedMulti(chunks, startPos, Unwrap(caches), wantLogits);

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches) =>
            inner.BatchForwardMulti(tokens, positions, Unwrap(caches));
    }
}
