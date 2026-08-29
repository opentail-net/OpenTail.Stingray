
namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// docs/028 Phase 3: the real-model proof <see cref="HotSessionForkTests"/> (Sessions.Fast) can't
/// give — that <see cref="HotSessionRuntime.Fork"/> makes each branch genuinely share physical KV
/// pages with the parent (not merely produce matching output), and that copy-on-write correctly
/// isolates a branch's own divergence from its siblings and the parent. Correct output alone can't
/// distinguish genuine physical sharing from a coincidentally-correct independent computation, so
/// this checks both: the engine's own <see cref="ContinuousBatchingEngine.CrossSessionPrefixTokensShared"/>
/// counter (direct evidence the fork path ran for every branch, not a silent no-op) and a
/// precision-consistent replay oracle per branch (direct evidence each branch's own post-fork
/// generation is numerically correct, not corrupted by copy-on-write or by a sibling's writes).
///
/// <para>The oracle follows <see cref="HotSessionGreedyReplayTests"/>'s corrected design
/// (docs/029-prefill-batch-composition-numerics-bug.md): it replays through the session's own
/// entry points (<c>PrefillWithCache</c>/<c>BatchForwardMulti</c>) on an independent cache, not a
/// blanket cold <c>Prefill</c> call — re-Q8-quantizing a position the real session actually
/// decoded (exact F32) is exactly the mismatch that investigation found, and it would apply here
/// too since the parent's own history includes a decoded position.</para>
/// </summary>
public sealed class HotSessionForkRealModelTests : HeavyTestBase
{
    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public async Task Fork_TwoBranches_GenuinelyShareParentPagesAndDivergeCorrectlyOnCopyOnWrite()
    {
        var path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this replay test.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Same seed HotSessionGreedyReplayTests/CrossSessionPrefixSharingRealModelTests already
        // verified tokenises to exactly PagedKvCache.PageSize - 1 tokens for this model/tokenizer,
        // plus MaxNewTokens = 1: the parent's turn ends at EXACTLY one full, page-aligned block
        // (16 positions), so forking loses nothing to alignment flooring and the test can focus on
        // fork/CoW correctness rather than boundary arithmetic.
        const string seed =
            "The capital of France is Paris and the capital of Spain is Madrid and the";
        int seedTokens = tokenizer.Encode(seed).Count;
        Assert.True(seedTokens == PagedKvCache.PageSize - 1,
            $"Seed tokenises to {seedTokens} tokens; expected {PagedKvCache.PageSize - 1}.");

        ImmutableArray<ExecutionSegment> parentLog;
        ImmutableArray<int> parentHistory;
        ImmutableArray<int> branchAGenerated, branchBGenerated;
        using (var backend = new CpuBackend())
        {
            var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
            using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "fork-test", maxBatchSize: 1);
            var runtime = new HotSessionRuntime(engine, tokenizer);

            using var parent = runtime.Create();
            var t1 = await parent.RunTurnAsync(seed,
                new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
                SessionRevision.Initial, SessionOperationId.New(),
                SessionRequestDigest.FromCanonicalValue("seed"));
            Assert.Equal(SessionOperationState.Completed, t1.Operation.State);
            Assert.Equal(PagedKvCache.PageSize, t1.Cursor.MaterializedPositionCount);
            parentLog = t1.Cursor.ExecutionLog;
            parentHistory = [.. parentLog.SelectMany(s => ((TokenSegment)s).TokenIds)];

            var branches = runtime.Fork(parent, 2);
            var branchA = branches[0];
            var branchB = branches[1];
            try
            {
                // Both branches are exact, zero-copy snapshots of the parent at fork time.
                Assert.Equal(PagedKvCache.PageSize, branchA.Cursor.MaterializedPositionCount);
                Assert.Equal(PagedKvCache.PageSize, branchB.Cursor.MaterializedPositionCount);
                Assert.Equal(parentHistory, branchA.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
                Assert.Equal(parentHistory, branchB.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
                // Each of the 2 branches independently forked the same 16-position prefix -- direct
                // evidence the fork path ran for both, not a silent cold-start fallback.
                Assert.Equal(2 * PagedKvCache.PageSize, engine.CrossSessionPrefixTokensShared);

                // Branches diverge onto genuinely different continuations -- exactly the CoW
                // trigger: each branch's own generation writes to what was, until this point, a
                // page shared with its sibling and the parent.
                var tA = await branchA.RunTurnAsync(" and the capital of Italy is",
                    new SamplingParams { Temperature = 0f, MaxNewTokens = 4 },
                    SessionRevision.Initial, SessionOperationId.New(), SessionRequestDigest.FromCanonicalValue("a"));
                var tB = await branchB.RunTurnAsync(" and the capital of Germany is",
                    new SamplingParams { Temperature = 0f, MaxNewTokens = 4 },
                    SessionRevision.Initial, SessionOperationId.New(), SessionRequestDigest.FromCanonicalValue("b"));
                Assert.Equal(SessionOperationState.Completed, tA.Operation.State);
                Assert.Equal(SessionOperationState.Completed, tB.Operation.State);
                branchAGenerated = ((TokenSegment)tA.Cursor.ExecutionLog[^1]).TokenIds;
                branchBGenerated = ((TokenSegment)tB.Cursor.ExecutionLog[^1]).TokenIds;

                // The parent itself never wrote after the fork -- CoW on the branches must not
                // touch what the parent still points at.
                Assert.Equal(PagedKvCache.PageSize, parent.Cursor.MaterializedPositionCount);
                Assert.Equal(parentHistory, parent.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds));
            }
            finally
            {
                foreach (var b in branches) b.Dispose();
            }
        }

        // Oracle: for EACH branch independently, an oracle ForwardPass/cache that REPLAYS the
        // parent's own history segment by segment -- PrefillWithCache (Q8) for the prompt segment,
        // BatchForwardMulti (F32) with the parent's OWN recorded token for the generated segment --
        // rather than one blanket cold PrefillWithCache over all 16 positions. The parent's turn 1
        // decoded its last position; re-deriving it via a fresh Q8 prefill (as an earlier version
        // of this test did) reintroduces exactly the docs/029 mismatch on the shared prefix itself,
        // not a fork/CoW defect. Then continues with that branch's own suffix via the same
        // PrefillWithCache/BatchForwardMulti pair the branch itself used. If the fork shared
        // stale/wrong pages, or CoW let one branch's write leak into the other, this is where it
        // would show up as a token mismatch.
        AssertMatchesPrecisionConsistentReplay(model, hp, tokenizer, parentLog,
            " and the capital of Italy is", branchAGenerated);
        AssertMatchesPrecisionConsistentReplay(model, hp, tokenizer, parentLog,
            " and the capital of Germany is", branchBGenerated);
    }

    private static void AssertMatchesPrecisionConsistentReplay(
        GgufModel model, ModelHyperparams hp, ITokenizer tokenizer,
        ImmutableArray<ExecutionSegment> parentLog, string continuationText, ImmutableArray<int> actualGenerated)
    {
        using var backend = new CpuBackend();
        using var oracleFwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        using var oracleCache = oracleFwd.CreateCache();

        // Replay the parent's own execution log exactly as it was produced: alternating prompt
        // (prefill) and generated (decode) segments, using the parent's own recorded token ids for
        // the generated ones rather than re-sampling them.
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        for (int seg = 0; seg < parentLog.Length; seg++)
        {
            var tokens = ((TokenSegment)parentLog[seg]).TokenIds;
            bool isGenerated = seg % 2 == 1;
            if (!isGenerated)
            {
                logits = oracleFwd.PrefillWithCache(tokens, oracleCache, startPos: pos);
                pos += tokens.Length;
            }
            else
            {
                foreach (var token in tokens)
                {
                    logits = oracleFwd.BatchForwardMulti([token], [pos], [oracleCache])[0];
                    pos++;
                }
            }
        }

        var suffix = tokenizer.Encode(continuationText);
        logits = oracleFwd.PrefillWithCache(suffix, oracleCache, startPos: pos);
        pos += suffix.Count;

        var expected = new int[actualGenerated.Length];
        for (int i = 0; i < expected.Length; i++)
        {
            expected[i] = Sampler.Greedy(logits);
            if (i + 1 < expected.Length)
                logits = oracleFwd.BatchForwardMulti([expected[i]], [pos], [oracleCache])[0];
            pos++;
        }

        Assert.Equal(expected.Length, actualGenerated.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actualGenerated[i],
                $"generated token {i}: branch produced {actualGenerated[i]} but a precision-consistent "
                + $"replay of the identical history + \"{continuationText}\" produced {expected[i]} — "
                + "the forked/copy-on-written pages computed different numbers than a correct replay would.");
    }
}
