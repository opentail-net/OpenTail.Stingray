using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Fast (no real model), page-boundary-focused regression tests for the layer between
/// <see cref="PagedKvCache"/> (proven correct at page boundaries — see
/// <c>PagedKvCacheTests.AppendAt_ExplicitPositionsAcrossPageBoundary_RoundTripCorrectly</c>) and
/// the full <c>HotSessionGreedyReplayTests</c> real-model oracle (which currently FAILS at exactly
/// this boundary). These use a page-agnostic fake <see cref="IBatchedForwardPass"/> to isolate one
/// specific remaining suspect: whether <c>ContinuousBatchingEngine.RunPrefillStep</c>'s chunked
/// admission produces a correct, gap-free, non-overlapping position sequence when a retained
/// session's second turn is (a) admitted starting exactly where a prior turn left off, and (b) that
/// turn's own prompt is itself split into multiple prefill chunks.
///
/// <para>The fake cannot reproduce a wrong SAMPLED TOKEN (it has no real numerics) — only a wrong
/// POSITION handed to the forward pass, or a KV cache write that lands on the wrong slot. That is a
/// narrower claim than the real-model oracle makes, but it is exactly the class of bug this
/// investigation is hunting, and it runs in milliseconds instead of ~20 seconds per real-model run.
/// </para>
/// </summary>
public sealed class HotSessionChunkedPrefillBoundaryTests
{
    private const int PageSize = 16; // mirrors PagedKvCache.PageSize; not read from it (fake is page-agnostic)
    private const int NonStopToken = 5;
    private const int Eos = 1;

    [Fact]
    public async Task RetainedTurn_ChunkedAcrossPriorPageBoundary_WritesContiguousPositionsWithNoGapsOrOverlaps()
    {
        var fwd = new ChunkAwareFakeForwardPass();
        // Small chunk size forces turn 2's prompt to be split across multiple RunPrefillStep calls,
        // with the first chunk starting exactly at the page boundary the previous turn left behind.
        using var engine = new ContinuousBatchingEngine(fwd, new CharTokenizer(), "test",
            maxBatchSize: 1, prefillChunkTokens: 4);
        var runtime = new HotSessionRuntime(engine, new CharTokenizer());
        using var session = runtime.Create();

        // Turn 1: a PageSize-1 (15) token seed + MaxNewTokens=1 lands the retained cache at
        // exactly position 16 — the same geometry HotSessionGreedyReplayTests uses, chosen so the
        // boundary is landed on exactly rather than merely crossed.
        var t1 = await session.RunTurnAsync(new string('a', PageSize - 1),
            new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
            SessionRevision.Initial, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue("seed"));
        Assert.Equal(SessionOperationState.Completed, t1.Operation.State);
        Assert.Equal(PageSize, t1.Cursor.MaterializedPositionCount);

        // Turn 2: a 6-token prompt (chunked 4 + 2 at chunk size 4) + 3 generated tokens, all on the
        // cache retained from turn 1. This is the untested combination: chunking AND resuming a
        // retained cache together, with the first chunk boundary coinciding with the page boundary.
        var t2 = await session.RunTurnAsync(new string('b', 6),
            new SamplingParams { Temperature = 0f, MaxNewTokens = 3 },
            t1.Operation.CommittedRevision!.Value, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue("across"));
        Assert.Equal(SessionOperationState.Completed, t2.Operation.State);

        const int expectedFinalPosition = PageSize + 6 + 3; // 25
        Assert.Equal(expectedFinalPosition, t2.Cursor.MaterializedPositionCount);

        Assert.Empty(fwd.Violations);
        var cache = Assert.Single(fwd.CachesCreated);
        var expected = Enumerable.Range(0, expectedFinalPosition).ToHashSet();
        Assert.True(expected.SetEquals(cache.WrittenPositions),
            $"expected positions 0..{expectedFinalPosition - 1} written exactly once each; got " +
            $"[{string.Join(",", cache.WrittenPositions.OrderBy(p => p))}]");

        // The chunking actually happened as designed -- otherwise this test isn't exercising what
        // it claims to. If this fails, _prefillChunkTokens/admission stopped chunking a 6-token
        // prompt and the test needs a different chunk size, not a shrug.
        Assert.True(fwd.PrefillCalls.Count(c => c.StartPos >= PageSize) >= 2,
            $"expected turn 2's prompt to be split into 2+ chunks; got calls: " +
            $"[{string.Join(",", fwd.PrefillCalls.Select(c => $"({c.StartPos},{c.Count})"))}]");
    }

    private sealed class PageAwareFakeCache : IRewindableSequenceKvCache
    {
        public int LogicalPosition { get; set; }
        public HashSet<int> WrittenPositions { get; } = [];
        public List<string> Violations { get; } = [];

        public void RecordWrite(int position)
        {
            if (!WrittenPositions.Add(position))
                Violations.Add($"position {position} written more than once");
        }

        public bool CanRewindTo(int logicalPosition) => logicalPosition >= 0 && logicalPosition <= LogicalPosition;

        public void RewindTo(int logicalPosition)
        {
            LogicalPosition = logicalPosition;
            WrittenPositions.RemoveWhere(p => p >= logicalPosition);
        }

        public void Dispose() { }
    }

    private sealed class ChunkAwareFakeForwardPass : IBatchedForwardPass
    {
        public List<(int StartPos, int Count)> PrefillCalls { get; } = [];
        public List<PageAwareFakeCache> CachesCreated { get; } = [];
        public List<string> Violations { get; } = [];

        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 4096;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache()
        {
            var cache = new PageAwareFakeCache();
            CachesCreated.Add(cache);
            return cache;
        }

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var c = Assert.IsType<PageAwareFakeCache>(cache);
            if (startPos != c.LogicalPosition)
                Violations.Add($"PrefillWithCache startPos={startPos} but cache.LogicalPosition={c.LogicalPosition}");
            PrefillCalls.Add((startPos, tokens.Count));
            for (int i = 0; i < tokens.Count; i++)
                c.RecordWrite(startPos + i);
            c.LogicalPosition = startPos + tokens.Count;
            Violations.AddRange(c.Violations);
            c.Violations.Clear();
            return CreateLogits(NonStopToken);
        }

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException("Single-sequence retained sessions take the sCount==1 PrefillWithCache path.");

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var result = new float[tokens.Length][];
            for (int i = 0; i < caches.Length; i++)
            {
                var c = Assert.IsType<PageAwareFakeCache>(caches[i]);
                if (positions[i] != c.LogicalPosition)
                    Violations.Add($"BatchForwardMulti position={positions[i]} but cache.LogicalPosition={c.LogicalPosition}");
                c.RecordWrite(positions[i]);
                c.LogicalPosition = positions[i] + 1;
                Violations.AddRange(c.Violations);
                c.Violations.Clear();
                result[i] = CreateLogits(NonStopToken);
            }
            return result;
        }

        private static float[] CreateLogits(int token)
        {
            var logits = new float[64];
            logits[token] = 1f;
            return logits;
        }
    }

    private sealed class CharTokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        // One token per character -- lets the test build exact-length prompts via string length.
        public IReadOnlyList<int> Encode(string text)
        {
            var ids = new int[text.Length];
            for (int i = 0; i < text.Length; i++)
                ids[i] = 10 + (text[i] % 40); // stay clear of Eos/NonStopToken/reserved low ids
            return ids;
        }

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }
}
