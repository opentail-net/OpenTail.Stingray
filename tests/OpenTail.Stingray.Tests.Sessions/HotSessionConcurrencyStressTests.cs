using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Stress test for the HotSession/ContinuousBatchingEngine continuous-batching path under
/// genuinely concurrent, independent request load — 5, then 10, then 40 simultaneous sessions on
/// one shared engine/model instance, one real CPU dense model. Deliberately NOT fake-based: the
/// whole point is exercising the real batched-decode code path
/// (<c>ContinuousBatchingEngine</c>'s decode loop → <c>IBatchedForwardPass.BatchForwardMulti</c> →
/// <c>SimdKernels.MatMulBatched</c>) under real compute, not just orchestration logic a fake can't
/// diverge on.
///
/// <para><b>Correctness signal.</b> Every concurrent session runs the IDENTICAL prompt with greedy
/// (temperature 0) sampling — deterministic, so absent any batch-composition-dependent defect,
/// every session must produce byte-identical output regardless of how many others are running
/// alongside it.</para>
///
/// <para><b>This found a real, standing defect — see docs/031-concurrent-decode-batch-tier-divergence-bug.md
/// for the full investigation.</b> <c>Stress_5ConcurrentSessions</c> and
/// <c>Stress_10ConcurrentSessions</c> are LEFT RED ON PURPOSE, the same standing pattern this
/// codebase already uses for <c>ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull</c>
/// (see <c>docs/00-current-work.md</c>, "Known defect — one deliberate red test"): a session
/// decoding concurrently with 4 or more others (but fewer than 16 — see below) can silently continue
/// past the point where it would otherwise have stopped, producing DIFFERENT output than running
/// alone or alongside a smaller/larger group would have. <c>Stress_40ConcurrentSessions</c> is
/// expected to PASS — 40 crosses <c>SimdKernels.MinBatchForBlas</c> (default 16), which routes
/// decode through a different (OpenBLAS GEMM) code path this defect does not reach.</para>
///
/// <para>Isolated with <see cref="Boundary_4Concurrent_AllMatch"/> (passes) and
/// <see cref="Boundary_8Concurrent_StillDiverges"/> (fails, same as 5/10): the boundary is exactly
/// batch composition inside <c>SimdKernels.MatMulBatched</c>'s small-batch tiered dispatch
/// (<c>MatVec4In</c> groups of 4, then <c>MatVec2In</c>, then plain <c>MatVec</c> for the odd
/// remainder) — safe only when a single tier call handles the WHOLE batch (N ≤ 4, or N ≥ 16 via
/// BLAS instead); unsafe the moment more than one call/tier is needed within one
/// <c>BatchForwardMulti</c> step (5 ≤ N ≤ 15, confirmed at N = 5, 6, 7, 8, 10 — including N = 8,
/// which is two back-to-back <c>MatVec4In</c> calls with no OTHER tier involved, ruling out
/// "tier-mixing" specifically and pointing at repeated/multiple invocation of the small-batch path
/// in general).</para>
/// </summary>
public sealed class HotSessionConcurrencyStressTests : HeavyTestBase
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

    /// <summary>Boundary evidence: exactly one <c>MatVec4In</c> call handles all 4 rows, nothing
    /// else runs in the same batched step. Expected to pass.</summary>
    [Fact]
    public Task Boundary_4Concurrent_AllMatch() => RunConcurrencyLevelAsync(4);

    /// <summary>Boundary evidence: two back-to-back <c>MatVec4In</c> calls (rows 0-3, then 4-7),
    /// no OTHER tier involved — rules out "mixing different tiers" as the necessary ingredient.
    /// Expected to fail, identically to 5/10 below.</summary>
    [Fact]
    public Task Boundary_8Concurrent_StillDiverges() => RunConcurrencyLevelAsync(8);

    /// <summary>Left red on purpose — see this file's header comment and
    /// docs/031-concurrent-decode-batch-tier-divergence-bug.md.</summary>
    [Fact]
    public Task Stress_5ConcurrentSessions() => RunConcurrencyLevelAsync(5);

    /// <summary>Left red on purpose — see this file's header comment and
    /// docs/031-concurrent-decode-batch-tier-divergence-bug.md.</summary>
    [Fact]
    public Task Stress_10ConcurrentSessions() => RunConcurrencyLevelAsync(10);

    /// <summary>Expected to pass — 40 crosses <see cref="SimdKernels.MinBatchForBlas"/> and routes
    /// through OpenBLAS GEMM instead of the small-batch tiered dispatch this defect lives in.</summary>
    [Fact]
    public Task Stress_40ConcurrentSessions() => RunConcurrencyLevelAsync(40);

    private static async Task RunConcurrencyLevelAsync(int concurrency)
    {
        var path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this stress test.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        using var backend = new CpuBackend();
        var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "stress-test", maxBatchSize: concurrency);
        var runtime = new HotSessionRuntime(engine, tokenizer);

        const string prompt = "The capital of France is";
        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 3 };

        var sessions = new HotSession[concurrency];
        for (int i = 0; i < concurrency; i++) sessions[i] = runtime.Create();

        try
        {
            var sw = Stopwatch.StartNew();
            var results = await Task.WhenAll(sessions.Select((session, i) =>
                session.RunTurnAsync(prompt, sampling, SessionRevision.Initial, SessionOperationId.New(),
                    SessionRequestDigest.FromCanonicalValue($"stress-{concurrency}-{i}"))));
            sw.Stop();

            for (int i = 0; i < results.Length; i++)
                Assert.True(results[i].Operation.State == SessionOperationState.Completed,
                    $"session {i} of {concurrency} did not complete: {results[i].Operation.State}.");

            var expected = ((TokenSegment)results[0].Cursor.ExecutionLog[^1]).TokenIds;
            for (int i = 1; i < results.Length; i++)
            {
                var actual = ((TokenSegment)results[i].Cursor.ExecutionLog[^1]).TokenIds;
                Assert.True(expected.SequenceEqual(actual),
                    $"session {i} of {concurrency} produced [{string.Join(",", actual)}] but session 0 "
                    + $"produced [{string.Join(",", expected)}] for the IDENTICAL prompt under greedy "
                    + $"sampling -- {concurrency}-way concurrent load corrupted or crossed session state. "
                    + "See docs/031-concurrent-decode-batch-tier-divergence-bug.md if this is 5<=N<=15.");
            }

            Assert.True(sw.Elapsed < TimeSpan.FromMinutes(5),
                $"{concurrency}-way concurrent decode took {sw.Elapsed} — investigate before treating "
                + "this as a timing fluke; the assertion exists to catch a genuine scheduling stall, "
                + "not to enforce a throughput target.");
        }
        finally
        {
            foreach (var session in sessions) session.Dispose();
        }

        // No resident-byte leak once every session that ran under load is fully disposed.
        Assert.Equal(0, runtime.ResidentBytes);
    }
}
