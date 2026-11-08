using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 3 memory accounting (Qwen 3.8 plan): a retained prefix's fixed sequence state (recurrent conv and state, PLE
/// history) is charged to the prefix cache, not only the per-token KV. Synthetic, no checkpoint.
/// </summary>
public sealed class ContinuousBatchingFixedStateTests
{
    private const long FixedBytes = 1000;

    [Fact]
    public async Task CachedPrefix_ChargesTheFixedStateToTheCacheBudget()
    {
        var tok = new FixedStateTokenizer();
        var fwd = new FixedStateForwardPass(FixedBytes);
        using var engine = new ContinuousBatchingEngine(fwd, tok, "fixed-state", maxBatchSize: 1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        // Two prompt tokens: the engine always leaves the last prompt token uncached, so one token is cacheable.
        await foreach (var _ in engine.GenerateAsync("QR", sp, cts.Token)) { }

        Assert.True(engine.PrefixCacheEntries >= 1, "The prompt's prefix was not cached, so the test checks nothing.");
        // Per-token KV is 1 byte here, so without the fixed charge the used bytes would be a few bytes, not >= 1000.
        Assert.True(engine.PrefixCacheUsedBytes >= FixedBytes,
            $"prefix cache used {engine.PrefixCacheUsedBytes} bytes; the fixed state of {FixedBytes} bytes was not charged.");
    }

    [Fact]
    public async Task ConcurrentPrompts_EvictingEachOthersPrefix_DoNotThrowWhenAPromptFinishes()
    {
        // Property check: with a budget for one entry (1001 bytes), concurrent prompts evict each other's snapshots and
        // must still complete. Checked: this does NOT reproduce the Qwen crash (the exact-boundary guard in RetainPrefix
        // was removed and this still passed). The regression guard for that crash is Qwen4ExpBatchingTests (real checkpoint).
        var tok = new FixedStateTokenizer();
        var fwd = new FixedStateForwardPass(FixedBytes);
        using var engine = new ContinuousBatchingEngine(
            fwd, tok, "fixed-state", maxBatchSize: 2, prefillChunkTokens: 1, prefixCacheBytes: 1500);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        async Task<string> RunAsync(string prompt)
        {
            var sb = new StringBuilder();
            await foreach (var chunk in engine.GenerateAsync(prompt, sp, cts.Token)) sb.Append(chunk);
            return sb.ToString();
        }

        // Three prompt tokens each: one token is uncached, so the boundary is two tokens and prefill takes several steps.
        await Task.WhenAll(RunAsync("QRS"), RunAsync("TUV"));
    }

    [Fact]
    public void FixedFloatBytes_MatchesTheArraysAStateAllocates()
    {
        const int numLayers = 5, convLen = 3, stateLen = 4, pleLen = 6;
        static bool Recurrent(int l) => l % 2 == 0;

        var state = new Qwen4ExpSequenceState(numLayers, Recurrent, convLen, stateLen, pleLen, pleHasher: null);
        long actualFloats = state.PleConvHistory.Length;
        for (int l = 0; l < numLayers; l++)
            actualFloats += state.GdnConvState[l].Length + state.GdnState[l].Length;

        Assert.Equal(actualFloats * sizeof(float),
            Qwen4ExpSequenceState.FixedFloatBytes(numLayers, Recurrent, convLen, stateLen, pleLen));
    }

    [Fact]
    public void FixedFloatBytes_ZeroLengthBuffersStillCountOneFloat()
    {
        // The constructor allocates at least one float for each recurrent buffer; the sizing must agree.
        static bool AllRecurrent(int l) => true;
        var state = new Qwen4ExpSequenceState(2, AllRecurrent, 0, 0, 0, pleHasher: null);
        long actualFloats = state.PleConvHistory.Length;
        for (int l = 0; l < 2; l++)
            actualFloats += state.GdnConvState[l].Length + state.GdnState[l].Length;

        Assert.Equal(actualFloats * sizeof(float), Qwen4ExpSequenceState.FixedFloatBytes(2, AllRecurrent, 0, 0, 0));
    }

    [Fact]
    public void BackendsWithoutFixedState_ReportZero()
    {
        IBatchedForwardPass backend = new NoFixedStateForwardPass();
        Assert.Equal(0, backend.FixedBytesPerSequence);
    }

    /// <summary>Tokenizer with one character token and an end-of-generation token.</summary>
    private sealed class FixedStateTokenizer : ITokenizer
    {
        public const int Pad = 0, Bos = 1, Eos = 2, ByteBase = 10;
        public int VocabSize => ByteBase + 256;
        public int BosTokenId => Bos;
        public int EosTokenId => Eos;
        public int UnknownTokenId => Pad;
        public int PadTokenId => Pad;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text) =>
            Encoding.UTF8.GetBytes(text).Select(b => ByteBase + b).ToArray();

        public string Decode(IEnumerable<int> tokens) =>
            Encoding.UTF8.GetString(tokens.Where(t => t >= ByteBase).Select(t => (byte)(t - ByteBase)).ToArray());

        public byte[] DecodeBytes(int token) => token >= ByteBase ? [(byte)(token - ByteBase)] : [];
    }

    private sealed class FixedStateCache : ISequenceKvCache
    {
        public void Dispose() { }
    }

    /// <summary>Predicts end-of-generation at once, so each request is just a prefill. Prefix-cacheable, with a fixed size.</summary>
    private sealed class FixedStateForwardPass(long fixedBytes) : IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public long FixedBytesPerSequence => fixedBytes;
        public int MaxSeqLen => 4096;
        public bool PrefillDequantCacheActive => false;

        public int PrefixCacheBlockSize => 1;
        public bool RequiresExactPrefixBoundary => true;

        public ISequenceKvCache CreateCache() => new FixedStateCache();
        public ISequenceKvCache CapturePrefix(ISequenceKvCache cache, int prefixLength) => new FixedStateCache();
        public ISequenceKvCache ForkPrefix(ISequenceKvCache prefix) => new FixedStateCache();

        private static float[] Eos()
        {
            var row = new float[FixedStateTokenizer.ByteBase + 256];
            row[FixedStateTokenizer.Eos] = 1f;
            return row;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0) => Eos();

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        {
            var result = new float[]?[chunks.Length];
            for (int i = 0; i < chunks.Length; i++)
                result[i] = wantLogits[i] ? Eos() : null;
            return result;
        }

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches) =>
            tokens.Select(_ => Eos()).ToArray();
    }

    /// <summary>Stand-in whose only point is that the interface default is zero.</summary>
    private sealed class NoFixedStateForwardPass : IBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 16;
        public bool PrefillDequantCacheActive => false;
        public ISequenceKvCache CreateCache() => new FixedStateCache();
        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0) => default;
        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            new float[]?[chunks.Length];
        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches) => [];
    }
}
