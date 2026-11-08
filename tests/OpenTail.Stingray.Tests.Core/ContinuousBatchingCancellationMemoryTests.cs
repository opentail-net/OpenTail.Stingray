using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 2 memory checks (Qwen 3.8 plan): a cancelled request must dispose its sequence cache and release its KV
/// reservation, and so must a finished one. Counted with a fake pass, so no checkpoint is needed.
/// </summary>
public sealed class ContinuousBatchingCancellationMemoryTests
{
    private const int ByteBase = 10;

    [Fact]
    public async Task CancelledRequest_DisposesItsCacheAndReleasesItsReservation()
    {
        var fwd = new CountingForwardPass();
        using var engine = new ContinuousBatchingEngine(fwd, new ByteTokenizer(), "cancel-memory", maxBatchSize: 2);
        // Never stops on its own: the request runs until it is cancelled.
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4000 };

        using var cancel = new CancellationTokenSource();
        int seen = 0;
        try
        {
            await foreach (var _ in engine.GenerateChunksAsync("Q", sp, cancel.Token))
            {
                if (++seen == 3) cancel.Cancel();
            }
        }
        catch (OperationCanceledException) { }

        await WaitUntilAsync(() => fwd.Created == fwd.Disposed && engine.CommittedKvTokens == 0);
        Assert.True(fwd.Created > 0, "The request never created a cache, so the check tests nothing.");
        Assert.Equal(fwd.Created, fwd.Disposed);
        Assert.Equal(0, engine.CommittedKvTokens);
    }

    [Fact]
    public async Task FinishedRequest_DisposesItsCacheAndReleasesItsReservation()
    {
        var fwd = new CountingForwardPass(stopAfter: 3);
        using var engine = new ContinuousBatchingEngine(fwd, new ByteTokenizer(), "finish-memory", maxBatchSize: 2);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 100 };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        var sb = new StringBuilder();
        await foreach (var chunk in engine.GenerateAsync("Q", sp, cts.Token)) sb.Append(chunk);

        await WaitUntilAsync(() => fwd.Created == fwd.Disposed && engine.CommittedKvTokens == 0);
        Assert.True(fwd.Created > 0);
        Assert.Equal(fwd.Created, fwd.Disposed);
        Assert.Equal(0, engine.CommittedKvTokens);
    }

    /// <summary>A poll rather than a fixed sleep, so a slow release passes and a leak still fails at the timeout.</summary>
    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (!condition() && DateTime.UtcNow < deadline)
            await Task.Delay(10);
    }

    internal sealed class ByteTokenizer : ITokenizer
    {
        public const int Pad = 0, Bos = 1, Eos = 2;
        public int VocabSize => ByteBase + 256;
        public int BosTokenId => Bos;
        public int EosTokenId => Eos;
        public int UnknownTokenId => Pad;
        public int PadTokenId => Pad;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text) => Encoding.UTF8.GetBytes(text).Select(b => ByteBase + b).ToArray();

        public string Decode(IEnumerable<int> tokens) =>
            Encoding.UTF8.GetString(tokens.Where(t => t >= ByteBase).Select(t => (byte)(t - ByteBase)).ToArray());

        public byte[] DecodeBytes(int token) => token >= ByteBase ? [(byte)(token - ByteBase)] : [];
    }

    /// <summary>Counts cache creation and disposal. Emits 'a' each step, or end-of-generation after <paramref name="stopAfter"/> steps.</summary>
    internal sealed class CountingForwardPass(int? stopAfter = null) : IBatchedForwardPass
    {
        private const int Token = ByteBase + 'a';
        private int _created, _disposed;
        public int Created => Volatile.Read(ref _created);
        public int Disposed => Volatile.Read(ref _disposed);

        private sealed class Cache(CountingForwardPass owner) : ISequenceKvCache
        {
            public int Steps;
            public void Dispose() => Interlocked.Increment(ref owner._disposed);
        }

        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 1 << 20;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache()
        {
            Interlocked.Increment(ref _created);
            return new Cache(this);
        }

        private float[] Next(int steps)
        {
            var row = new float[ByteBase + 256];
            bool stop = stopAfter is int n && steps >= n;
            row[stop ? ByteTokenizer.Eos : Token] = 1f;
            return row;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0) =>
            Next(0);

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        {
            var result = new float[]?[chunks.Length];
            for (int i = 0; i < chunks.Length; i++)
                result[i] = wantLogits[i] ? Next(0) : null;
            return result;
        }

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var result = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
            {
                var c = (Cache)caches[i];
                c.Steps++;
                result[i] = Next(c.Steps);
            }
            return result;
        }
    }
}
