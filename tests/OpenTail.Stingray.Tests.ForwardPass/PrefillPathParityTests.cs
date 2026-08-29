
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Regression guard for the HotSession replay-divergence root cause (fixed 2026-08-13):
/// <see cref="Engine.ForwardPass.PrefillWithCache"/> used to dispatch a single-token continuation
/// (<c>N == 1</c>) to the non-quantized <c>PrefillWithCacheSequential</c>/<c>ForwardCore</c> path,
/// while any multi-token call -- including the full-replay oracle, which always recomputes the
/// whole prefix in one batched call -- took the Q8-quantized <c>PrefillCore</c> path. A retained
/// session's short next-turn prompt (e.g. one word, " capital") is exactly the shape that used to
/// hit <c>N == 1</c>, so the SAME logical sequence was computed with two different, non-bit-exact
/// precisions depending on how the caller happened to chunk it across turns -- measured at
/// maxAbsDiff ~0.85 across every one of 49152 vocab logits for the same position before the fix,
/// enough to flip a close greedy choice a few tokens later (exactly what
/// <c>HotSessionGreedyReplayTests</c> caught). The fix (<c>ForwardPass.cs</c>, around
/// <c>PrefillWithCache</c>'s dispatch) removed the <c>N == 1</c> short-circuit from that method
/// specifically, so a retained session's continuations always take the same Q8 path its own longer
/// turns and its oracle do, regardless of length.
///
/// <para>This test pins that fix down: build the identical prefix into two caches, then compute
/// logits for the SAME next token two ways -- a 1-token continuation call, and a fresh multi-token
/// batched call -- and assert they are now bit-identical. No session/engine/chunking machinery
/// involved, so a regression here can only come from <c>PrefillWithCache</c>'s dispatch itself.</para>
/// </summary>
public sealed class PrefillPathParityTests : HeavyTestBase
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
    public void SingleTokenContinuation_MatchesFreshBatchedPrefill_ForTheSamePosition()
    {
        var path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this parity check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        // Same seed as HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay: 15 tokens,
        // each word-with-leading-space a single BPE token in SmolLM2.
        const string seed = "The capital of France is Paris and the capital of Spain is Madrid and the";
        var tokens = tokenizer.Encode(seed).ToArray();
        Assert.True(tokens.Length >= 15, $"seed tokenised to {tokens.Length} tokens; expected >= 15.");

        int prefixLen = tokens.Length - 1;   // everything except the last token
        var prefix = tokens[..prefixLen];
        int boundaryToken = tokens[prefixLen];

        // Arm 1 -- retained continuation: prefix built via a batched call, THEN the final token
        // appended via a single-token PrefillWithCache call (N == 1) -- exactly the shape a
        // retained session's short next-turn prompt takes.
        var cacheContinuation = fwd.CreateCache();
        fwd.PrefillWithCache(prefix, cacheContinuation, startPos: 0);
        var logitsContinuation = fwd.PrefillWithCache([boundaryToken], cacheContinuation, startPos: prefixLen).ToArray();

        // Arm 2 -- fresh batched: the full prefix INCLUDING the boundary token computed in one
        // call (N > 1) -- mirrors the oracle, which always recomputes the whole prefix as one batch.
        var cacheBatched = fwd.CreateCache();
        var logitsBatched = fwd.PrefillWithCache(tokens[..(prefixLen + 1)], cacheBatched, startPos: 0).ToArray();

        Assert.Equal(logitsContinuation.Length, logitsBatched.Length);

        int argmaxContinuation = ArgMax(logitsContinuation);
        int argmaxBatched = ArgMax(logitsBatched);

        double maxAbsDiff = 0;
        int diffCount = 0;
        for (int i = 0; i < logitsContinuation.Length; i++)
        {
            double diff = Math.Abs(logitsContinuation[i] - logitsBatched[i]);
            if (diff > 0) diffCount++;
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }

        // Always logged (not just on failure) -- these numbers are the whole point of the test.
        Console.WriteLine(
            $"[PrefillPathParity] maxAbsDiff={maxAbsDiff}, {diffCount}/{logitsContinuation.Length} lanes differ, " +
            $"argmaxContinuation={argmaxContinuation}, argmaxBatched={argmaxBatched}");

        // Measured bit-identical (maxAbsDiff=0, 0/49152 lanes differ) after the fix. Before the
        // fix this was 0.8479442596435547 across all 49152 lanes. If this regresses, someone
        // reintroduced a precision split between a retained session's short continuations and its
        // own oracle -- see the class doc above and PrefillWithCache's dispatch comment in
        // ForwardPass.cs for the full mechanism.
        Assert.Equal(0, diffCount);
        Assert.Equal(0.0, maxAbsDiff);
        Assert.Equal(argmaxBatched, argmaxContinuation);
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}
