using System.Collections.Immutable;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// docs/028 Phase 2: the real-model proof <see cref="HotSessionGreedyReplayTests"/> models but
/// doesn't cover — that <see cref="HotSessionRuntime.CreateWithSharedPrefixHint"/> makes a NEW
/// session's KV cache genuinely share physical pages with an idle sibling's, not merely produce
/// the same output tokens a fresh cold prefill would. Correct output alone doesn't prove sharing
/// happened: a silently-falling-back-to-cold implementation would pass a correctness-only check
/// too. This file checks both, on a real CPU dense model: the engine's own
/// <see cref="ContinuousBatchingEngine.CrossSessionPrefixTokensShared"/> counter (direct evidence
/// the fork path actually ran, not a fallback) and a full greedy-replay oracle (direct evidence the
/// shared pages compute the same numbers a from-scratch prefill would) — mirroring
/// <see cref="HotSessionGreedyReplayTests"/>'s own "replay tokens, not text" reasoning: reusing a
/// second session's own recorded execution log sidesteps BPE seam re-tokenisation risk entirely.
/// </summary>
public sealed class CrossSessionPrefixSharingRealModelTests : HeavyTestBase
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
    public async Task CreateWithSharedPrefixHint_SecondSession_GenuinelySharesPagesAndMatchesGreedyReplay()
    {
        var path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this replay test.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Same seed HotSessionGreedyReplayTests already verified tokenises to exactly
        // PagedKvCache.PageSize - 1 tokens for this model/tokenizer, plus MaxNewTokens = 1: the
        // turn ends at EXACTLY one full, page-aligned block (16 positions), so the whole of A's
        // history is shareable with nothing lost to alignment flooring.
        const string seed =
            "The capital of France is Paris and the capital of Spain is Madrid and the";
        int seedTokens = tokenizer.Encode(seed).Count;
        Assert.True(seedTokens == PagedKvCache.PageSize - 1,
            $"Seed tokenises to {seedTokens} tokens; expected {PagedKvCache.PageSize - 1}.");

        ImmutableArray<int> history;
        ImmutableArray<int> bGenerated;
        int seeded;
        using (var backend = new CpuBackend())
        {
            var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
            using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "cross-session-prefix", maxBatchSize: 1);
            var runtime = new HotSessionRuntime(engine, tokenizer);

            using var sessionA = runtime.Create();
            var t1 = await sessionA.RunTurnAsync(seed,
                new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
                SessionRevision.Initial, SessionOperationId.New(),
                SessionRequestDigest.FromCanonicalValue("seed"));
            Assert.Equal(SessionOperationState.Completed, t1.Operation.State);
            Assert.Equal(PagedKvCache.PageSize, t1.Cursor.MaterializedPositionCount);

            history = [.. t1.Cursor.ExecutionLog.SelectMany(s => ((TokenSegment)s).TokenIds)];
            Assert.Equal(PagedKvCache.PageSize, history.Length);

            var (sessionB, seededLength) = runtime.CreateWithSharedPrefixHint(history);
            using var _ = sessionB;
            seeded = seededLength;

            // The whole 16-token history is one clean page -- nothing should be lost to
            // alignment flooring, and the fork path (not a cold-start fallback) must be the one
            // that actually ran.
            Assert.Equal(PagedKvCache.PageSize, seeded);
            Assert.Equal(PagedKvCache.PageSize, sessionB.Cursor.MaterializedPositionCount);
            Assert.Equal(PagedKvCache.PageSize, engine.CrossSessionPrefixTokensShared);

            // B diverges onto genuinely new content the shared prefix never covered -- the real
            // "many sessions share a system prompt, then diverge" use case, and it sidesteps any
            // BPE seam risk from trying to re-derive "the rest of A's own prompt" as text.
            var t2 = await sessionB.RunTurnAsync(" and the capital of Italy is",
                new SamplingParams { Temperature = 0f, MaxNewTokens = 4 },
                SessionRevision.Initial, SessionOperationId.New(),
                SessionRequestDigest.FromCanonicalValue("diverge"));
            Assert.Equal(SessionOperationState.Completed, t2.Operation.State);

            var bLog = t2.Cursor.ExecutionLog;
            // Seeded segment, then B's own append segment, then B's own generated segment.
            Assert.Equal(3, bLog.Length);
            bGenerated = ((TokenSegment)bLog[^1]).TokenIds;
        }

        // Oracle: a completely independent forward pass/backend, no shared state whatsoever with
        // the engine above, prefilling the exact same token ids A's cache actually held for its
        // shared prefix plus B's own appended suffix. If the fork shared stale, wrong, or
        // partially-copy-on-written pages, this is where it would show up as a token mismatch.
        using (var backend = new CpuBackend())
        using (var oracleFwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            var prefix = new List<int>(history);
            prefix.AddRange(tokenizer.Encode(" and the capital of Italy is"));
            var logits = oracleFwd.Prefill(prefix);
            var actual = new int[bGenerated.Length];
            int pos = prefix.Count;
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = Sampler.Greedy(logits);
                if (i + 1 < actual.Length) logits = oracleFwd.Forward(actual[i], pos++);
            }

            Assert.Equal(bGenerated.Length, actual.Length);
            for (int i = 0; i < actual.Length; i++)
                Assert.True(bGenerated[i] == actual[i],
                    $"generated token {i}: cross-session-seeded B produced {bGenerated[i]} but a full "
                    + $"from-scratch replay of the identical {prefix.Count}-token prefix produced {actual[i]} "
                    + "— the forked shared prefix computed different numbers than fresh pages would.");
        }
    }
}
