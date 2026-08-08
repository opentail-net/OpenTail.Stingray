using System.Collections.Immutable;
using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// The Milestone 0 §3.3 / Milestone 1 §8.0 item the plan leaves unchecked: <b>compare a hot
/// session against full replay under greedy sampling, on a real CPU dense model.</b>
///
/// <para>Every other <c>HotSession</c> test drives a <c>FakeForwardPass</c>. Those prove the
/// transaction machinery — revisions, leases, idempotent replay, compensation — and prove nothing
/// at all about the claim the whole runtime rests on: that retaining execution state across turns
/// produces the SAME tokens as re-running the conversation from scratch. A fake forward pass
/// cannot fail that way, because it has no numerics to diverge.</para>
///
/// <para><b>The oracle replays TOKENS, not text.</b> Re-tokenising
/// <c>prompt₁ + output₁ + prompt₂</c> as one string is not guaranteed to yield
/// <c>encode(prompt₁) ++ outputTokens₁ ++ encode(prompt₂)</c> — BPE merges across the seams. A
/// text-level replay would therefore diverge for a reason that has nothing to do with state reuse,
/// and would either fail spuriously or, worse, be "fixed" by loosening the assertion until it
/// stopped meaning anything. The session's own execution log records the exact token ids of every
/// segment, so the oracle prefills precisely those and decodes greedily from a fresh
/// <see cref="Engine.ForwardPass"/> that shares no state with the session.</para>
///
/// <para>The restart acceptance test is explicitly skipped when the reference model is not on disk.
/// A failure inside a turn must FAIL.</para>
/// </summary>
public sealed class HotSessionGreedyReplayTests
{
    private const string RestartPhaseEnvironment = "OPENTAIL_TEST_RESTART_PHASE";
    private const string RestartModelEnvironment = "OPENTAIL_TEST_RESTART_MODEL";
    private const string RestartStorageEnvironment = "OPENTAIL_TEST_RESTART_STORAGE";
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

    private const int MaxNewTokens = 6;

    [Fact]
    public async Task HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel()
    {
        var path = FindModelPath();
        // Assert.Skip, not a silent return: this file is cited by the release matrix as
        // restart/replay evidence, and a bare `return` reports PASSED on a machine without
        // models/ — i.e. on CI — which would let the matrix claim coverage from a test that
        // never executed.
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this replay test.");

        string[] turns = ["The capital of France is", " and the capital of Spain is", " and of Italy is"];

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // ── Arm A: one session, state retained across all three turns ──────────────────────
        ImmutableArray<ExecutionSegment> log;
        using (var backend = new CpuBackend())
        {
            var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
            using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "replay-test", maxBatchSize: 1);
            var runtime = new HotSessionRuntime(engine, tokenizer);
            using var session = runtime.Create();

            var revision = SessionRevision.Initial;
            foreach (var turn in turns)
            {
                var result = await session.RunTurnAsync(
                    turn,
                    new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens },
                    revision,
                    SessionOperationId.New(),
                    SessionRequestDigest.FromCanonicalValue(turn));

                Assert.Equal(SessionOperationState.Completed, result.Operation.State);
                revision = result.Operation.CommittedRevision!.Value;
            }
            log = session.Cursor.ExecutionLog;
        }

        // The log alternates prompt segment / generated segment, one pair per turn.
        Assert.Equal(turns.Length * 2, log.Length);

        // ── Arm B: full replay. For each generated segment, a FRESH forward pass prefills the
        //    exact token prefix that preceded it and decodes greedily. No retained state, no
        //    shared cache, no shared engine — the only thing carried over is the token ids.
        using (var backend = new CpuBackend())
        {
            var replay = new List<int>();
            for (int seg = 0; seg < log.Length; seg += 2)
            {
                var promptSeg = Assert.IsType<TokenSegment>(log[seg]);
                var generatedSeg = Assert.IsType<TokenSegment>(log[seg + 1]);
                replay.AddRange(promptSeg.TokenIds);

                var expected = generatedSeg.TokenIds;
                var actual = GreedyContinuation(model, backend, hp, replay, expected.Length);

                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(expected[i] == actual[i],
                        $"turn {seg / 2 + 1}, generated token {i}: session produced {expected[i]} "
                        + $"({Show(tokenizer, expected[i])}) but full greedy replay of the identical "
                        + $"{replay.Count}-token prefix produced {actual[i]} ({Show(tokenizer, actual[i])})");

                replay.AddRange(expected);
            }
        }
    }

    /// <summary>
    /// Milestone 1 required invariant: an exact append at a page boundary.
    ///
    /// <para><c>PagedKvCache</c> allocates in pages of <see cref="PagedKvCache.PageSize"/>
    /// positions. The boundary case is a turn that ends with the cache exactly full — no partial
    /// page — so the next turn's first token must open a fresh page. Off-by-one page accounting
    /// hides here specifically: it is invisible mid-page and invisible when a turn straddles a
    /// boundary; only landing exactly on it exposes the seam.</para>
    ///
    /// <para>Cache-level page behaviour is already covered by <c>PagedKvCacheTests</c>. What this
    /// adds is the SESSION path: the turn is driven through <see cref="HotSession"/> so the
    /// cursor, the reservation and the retained cache all cross the boundary together, and the
    /// result is checked against full replay — the same oracle as
    /// <see cref="HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel"/>.</para>
    ///
    /// <para><b>Design note — why <c>PageSize - 1</c> tokens for the seed and
    /// <c>MaxNewTokens = 1</c>.</b> An earlier design used a shorter seed and relied on the model
    /// generating <c>PageSize - seedTokens</c> tokens before EOS. That is fragile: "The capital of
    /// France is" (5 tokens) emits just two tokens (" Paris", ".") and then EOS, so the turn ended
    /// at <c>5 + 2 = 7 ≠ PageSize</c> and the boundary was never reached. The current design
    /// controls the boundary precisely — the seed fills <c>PageSize - 1</c> positions and the
    /// budget is 1, so the loop exits on the budget after exactly one generated token regardless
    /// of what the model would have sampled next. The only remaining failure mode is an EOS on the
    /// very first sample after 15 tokens of a rich prompt, which is practically impossible for this
    /// class of model.</para>
    ///
    /// <para><b>Both arms use <c>maxContextLength: 2048</c>, matching
    /// <see cref="GreedyContinuation"/>.</b> This is hygiene — an A/B whose arms differ in an
    /// unrelated parameter is not a comparison — and NOT because context length was observed to
    /// change the model's output. It was measured directly and it does not: greedy decoding of both
    /// the 5-token and the 15-token seed produces byte-identical output at <c>-c 512</c> and
    /// <c>-c 2048</c>. An earlier version of this note attributed a "spurious EOS at position 0" to
    /// the 512-token scratch buffer; that mechanism is unsupported, and the early-EOS behaviour
    /// above accounts for the original failure on its own.</para>
    /// </summary>
    [Fact]
    public async Task HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay()
    {
        var path = FindModelPath();
        // Assert.Skip, not a silent return: this file is cited by the release matrix as
        // restart/replay evidence, and a bare `return` reports PASSED on a machine without
        // models/ — i.e. on CI — which would let the matrix claim coverage from a test that
        // never executed.
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this replay test.");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Seed designed to be exactly PageSize - 1 tokens long, so MaxNewTokens = 1
        // places the turn end exactly on the first page boundary.
        // Each word-with-leading-space is a single BPE token in SmolLM2-1.7B.
        const string seed =
            "The capital of France is Paris and the capital of Spain is Madrid and";
        int seedTokens = tokenizer.Encode(seed).Count;
        Assert.True(seedTokens == PagedKvCache.PageSize - 1,
            $"Seed tokenises to {seedTokens} tokens; expected {PagedKvCache.PageSize - 1}. "
            + $"If the tokenizer changed, pick a new seed of exactly {PagedKvCache.PageSize - 1} tokens.");

        ImmutableArray<ExecutionSegment> log;
        using (var backend = new CpuBackend())
        {
            // maxContextLength must match GreedyContinuation (2048). A smaller value changes
            // the attention-score scratch buffer size and can alter numerical output.
            var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
            using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "page-boundary", maxBatchSize: 1);
            var runtime = new HotSessionRuntime(engine, tokenizer);
            using var session = runtime.Create();

            // MaxNewTokens = 1: the decode loop exits on the budget after exactly one token,
            // so MaterializedPositionCount = seedTokens + 1 = PageSize regardless of EOS.
            // (If the model samples EOS as the very first token the test fails — that would
            // indicate a problem with the prompt or model, not the session machinery.)
            var t1 = await session.RunTurnAsync(seed,
                new SamplingParams { Temperature = 0f, MaxNewTokens = 1 },
                SessionRevision.Initial, SessionOperationId.New(),
                SessionRequestDigest.FromCanonicalValue("seed"));

            Assert.Equal(SessionOperationState.Completed, t1.Operation.State);
            Assert.True(t1.Cursor.MaterializedPositionCount == PagedKvCache.PageSize,
                $"Expected MaterializedPositionCount={PagedKvCache.PageSize} after seed ({seedTokens} tok) "
                + $"+ budget-1 generation, but got {t1.Cursor.MaterializedPositionCount}. "
                + $"Model sampled EOS as first token — this indicates a prompt or model issue.");

            // Now append across the page seam: position 16 → opens a new page.
            var t2 = await session.RunTurnAsync(" capital",
                new SamplingParams { Temperature = 0f, MaxNewTokens = 4 },
                t1.Operation.CommittedRevision!.Value, SessionOperationId.New(),
                SessionRequestDigest.FromCanonicalValue("across"));

            Assert.Equal(SessionOperationState.Completed, t2.Operation.State);
            log = t2.Cursor.ExecutionLog;
        }

        // Full replay of every generated segment, exactly as §8.1 does.
        // Turn 1's single generated token lands at the page boundary (position 16);
        // turn 2's prompt opens the next page. Both are checked against fresh replay.
        using (var backend = new CpuBackend())
        {
            var replay = new List<int>();
            for (int seg = 0; seg + 1 < log.Length; seg += 2)
            {
                replay.AddRange(Assert.IsType<TokenSegment>(log[seg]).TokenIds);
                var expected = Assert.IsType<TokenSegment>(log[seg + 1]).TokenIds;
                var actual = GreedyContinuation(model, backend, hp, replay, expected.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(expected[i] == actual[i],
                        $"turn {seg / 2 + 1}, token {i}: session {expected[i]} vs replay {actual[i]} "
                        + "— an append landing exactly on a page boundary diverged from replay.");
                replay.AddRange(expected);
            }
        }
    }

    /// <summary>
    /// The release-quality restart-continuation gate: the persisted session must survive an actual
    /// test-process exit, then a fresh process/runtime must restore it and produce the same greedy
    /// tokens as a full replay. This deliberately orchestrates two child test processes instead of
    /// calling two runtimes in one test process; static configuration and native allocations are
    /// otherwise shared, which is not a restart proof.
    /// </summary>
    [Fact]
    public void ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay()
    {
        var modelPath = FindModelPath();
        Assert.SkipUnless(
            modelPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for the CPU-dense restart-continuation acceptance test.");

        string storage = Path.Combine(Path.GetTempPath(), $"opentail_restart_proof_{Guid.NewGuid():N}");
        try
        {
            RunRestartChild("persist", modelPath!, storage);
            RunRestartChild("restore", modelPath!, storage);
        }
        finally
        {
            if (Directory.Exists(storage)) Directory.Delete(storage, recursive: true);
        }
    }

    /// <summary>
    /// Child entry point selected by <see cref="ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay"/>.
    /// It intentionally does nothing in the normal suite, where the parent test owns orchestration.
    /// </summary>
    [Fact]
    public async Task ColdSession_RealModel_CrossProcessChild()
    {
        string? phase = Environment.GetEnvironmentVariable(RestartPhaseEnvironment);
        if (string.IsNullOrEmpty(phase)) return;

        string modelPath = Environment.GetEnvironmentVariable(RestartModelEnvironment)
            ?? throw new InvalidOperationException("Restart child did not receive a model path.");
        string storage = Environment.GetEnvironmentVariable(RestartStorageEnvironment)
            ?? throw new InvalidOperationException("Restart child did not receive a storage directory.");

        if (phase == "persist")
        {
            await PersistRestartFixture(modelPath, storage);
            return;
        }
        if (phase == "restore")
        {
            await RestoreAndReplayRestartFixture(modelPath, storage);
            return;
        }

        throw new InvalidOperationException($"Unknown restart-child phase '{phase}'.");
    }

    private static void RunRestartChild(string phase, string modelPath, string storage)
    {
        string executable = Environment.ProcessPath
            ?? throw new InvalidOperationException("Could not determine the test executable path.");
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("-method");
        start.ArgumentList.Add($"{typeof(HotSessionGreedyReplayTests).FullName}.{nameof(ColdSession_RealModel_CrossProcessChild)}");
        start.Environment[RestartPhaseEnvironment] = phase;
        start.Environment[RestartModelEnvironment] = modelPath;
        start.Environment[RestartStorageEnvironment] = storage;

        using var child = Process.Start(start)
            ?? throw new InvalidOperationException($"Could not start restart-proof child '{phase}'.");
        if (!child.WaitForExit((int)TimeSpan.FromMinutes(3).TotalMilliseconds))
            throw new TimeoutException($"Restart-proof child '{phase}' did not exit within three minutes.");

        string output = child.StandardOutput.ReadToEnd();
        string error = child.StandardError.ReadToEnd();
        Assert.True(child.ExitCode == 0,
            $"Restart-proof child '{phase}' exited {child.ExitCode}.{Environment.NewLine}{output}{Environment.NewLine}{error}");
    }

    private static async Task PersistRestartFixture(string modelPath, string storage)
    {
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "restart-proof", maxBatchSize: 1);
        var hot = new HotSessionRuntime(engine, tokenizer);
        var cold = new ColdSessionRuntime(hot, engine, storage);
        var address = new SessionAddress("restart-proof", "smollm", "thread", RestartModelFingerprint(modelPath));
        using var session = cold.Create(address);

        var result = await session.RunTurnAsync("The capital of France is",
            new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens },
            SessionRevision.Initial, SessionOperationId.New(), SessionRequestDigest.FromCanonicalValue("turn-1"));
        Assert.Equal(SessionOperationState.Completed, result.Operation.State);

        // A restart receipt must retain more than the initial turn. This forces a genuine
        // append onto already-populated KV state before the process exits; the restore child
        // will append a third turn and independently replay all three segments.
        var second = await session.RunTurnAsync(" The capital of Germany is",
            new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens },
            result.Operation.CommittedRevision!.Value, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue("turn-2"));
        Assert.Equal(SessionOperationState.Completed, second.Operation.State);

        cold.EvictToDisk(session, address.ModelFingerprint);
        Assert.True(File.Exists(Path.Combine(storage, $"{address.ToSessionId().Value:N}.manifest")),
            "Persist child did not write the session manifest.");
    }

    private static async Task RestoreAndReplayRestartFixture(string modelPath, string storage)
    {
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "restart-proof", maxBatchSize: 1);
        var hot = new HotSessionRuntime(engine, tokenizer);
        var cold = new ColdSessionRuntime(hot, engine, storage);
        var address = new SessionAddress("restart-proof", "smollm", "thread", RestartModelFingerprint(modelPath));
        using var session = cold.OpenOrCreate(address);

        // The revision to submit is the STORE's turn counter, read from the snapshot — the same
        // value the HTTP layer publishes as committed_revision. This line used to pass
        // `session.CommittedRevision` (the cursor position count) and worked only because the
        // persisted revision was also a position count, so two wrong values agreed. Both sides are
        // now the turn counter (manifest v3), so a restored session accepts the value a caller can
        // actually observe.
        var result = await session.RunTurnAsync(" and the capital of Spain is",
            new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens },
            hot.GetSessionSnapshot(session.SessionId).CommittedRevision,
            SessionOperationId.New(), SessionRequestDigest.FromCanonicalValue("turn-3"));
        Assert.Equal(SessionOperationState.Completed, result.Operation.State);

        var log = session.Cursor.ExecutionLog;
        Assert.True(log.Length > 0 && log.Length % 2 == 0,
            "The restart proof expects alternating prompt/generated token segments.");
        var replay = new List<int>();
        for (int i = 0; i < log.Length; i += 2)
        {
            var prompt = Assert.IsType<TokenSegment>(log[i]);
            var generated = Assert.IsType<TokenSegment>(log[i + 1]);
            replay.AddRange(prompt.TokenIds);
            var actual = GreedyContinuation(model, backend, hp, replay, generated.TokenIds.Length);
            Assert.Equal(generated.TokenIds, actual);
            replay.AddRange(generated.TokenIds);
        }
    }

    private static string RestartModelFingerprint(string modelPath) =>
        $"{Path.GetFileName(modelPath)}:{new FileInfo(modelPath).Length}";

    /// <summary>
    /// Full replay: a brand-new forward pass, prefill the whole prefix, then greedy-decode
    /// <paramref name="count"/> tokens. This is the "re-run the conversation from scratch" arm.
    /// </summary>
    private static int[] GreedyContinuation(GgufModel model, CpuBackend backend, ModelHyperparams hp,
        List<int> prefix, int count)
    {
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        var logits = fwd.Prefill(prefix);
        var outTokens = new int[count];
        int pos = prefix.Count;
        for (int i = 0; i < count; i++)
        {
            int next = Sampler.Greedy(logits);
            outTokens[i] = next;
            if (i + 1 < count) logits = fwd.Forward(next, pos++);
        }
        return outTokens;
    }

    private static string Show(ITokenizer tokenizer, int id)
    {
        try { return "\"" + tokenizer.Decode([id]) + "\""; }
        catch { return "<undecodable>"; }
    }
}
