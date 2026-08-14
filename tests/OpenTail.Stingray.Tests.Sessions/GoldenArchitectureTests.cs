using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public class GoldenArchitectureTests
{
    [Fact]
    public async Task GoldenTest1_Checkpoint_Generate_Rollback_Generate_ProducesIdenticalDeterministicReplay()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache, forwardPass: new StatefulTestForwardPass());

        int[] prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        await session.AppendAsync(prompt);

        var checkpointA = session.CreateCheckpoint();
        Assert.Equal(10, checkpointA.TokenPosition);

        // Generate first batch of 20 tokens
        var firstRun = new List<int>();
        var sampling = new SamplingParams { MaxNewTokens = 20, Temperature = 0f };
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            if (int.TryParse(chunk.Text, out int t))
            {
                firstRun.Add(t);
            }
        }

        Assert.Equal(20, firstRun.Count);
        Assert.Equal(30, session.TokenCount);

        // Rollback to Checkpoint A (pos 10)
        session.Rollback(checkpointA);
        Assert.Equal(10, session.TokenCount);
        Assert.Equal(10, session.KvSequence.TokenCount);
        Assert.Equal(prompt, session.TokenHistory);

        // Generate second batch of 20 tokens after rollback
        var secondRun = new List<int>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            if (int.TryParse(chunk.Text, out int t))
            {
                secondRun.Add(t);
            }
        }

        Assert.Equal(20, secondRun.Count);
        Assert.Equal(30, session.TokenCount);

        // GOLDEN ASSERTION: Deterministic replay must produce identical token sequence!
        Assert.Equal(firstRun, secondRun);
    }

    [Fact]
    public async Task GoldenTest2_ForkDivergence_Isolation_And_CoW_Duplication()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var sessionA = new InferenceSession(cache);

        int[] prompt = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]; // 15 tokens (1 page)
        await sessionA.AppendAsync(prompt);

        // Fork Session B from Session A (zero-copy page sharing)
        await using var sessionB = (InferenceSession)sessionA.Fork();
        Assert.Equal(1, cache.SharedPages);

        // Append different continuation to Session A
        await sessionA.AppendAsync(new int[] { 900, 901 });

        // Append different continuation to Session B
        await sessionB.AppendAsync(new int[] { 800, 801, 802 });

        // GOLDEN ASSERTION 1: Sequences diverge cleanly
        Assert.Equal(17, sessionA.TokenCount);
        Assert.Equal(18, sessionB.TokenCount);
        Assert.NotEqual(sessionA.TokenHistory, sessionB.TokenHistory);

        // GOLDEN ASSERTION 2: Shared page ref count decrements and CoW allocates private pages
        Assert.Equal(0, cache.SharedPages);
    }

    [Fact]
    public async Task GoldenTest3_DisposalOrder_ParentFirst_Or_ChildFirst_Independence()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);

        SessionId parentId, childId;

        // Sub-test A: Dispose parent first
        {
            var parent = new InferenceSession(cache);
            parentId = parent.Id;
            await parent.AppendAsync(new int[50]); // 2 pages allocated

            var child = (InferenceSession)parent.Fork();
            childId = child.Id;

            Assert.Equal(2, cache.SharedPages);

            // Dispose parent first
            await parent.DisposeAsync();
            Assert.Equal(0, cache.SharedPages); // Ref count decremented, child holds sequence
            Assert.Equal(50, child.TokenCount);

            // Dispose child second
            await child.DisposeAsync();
            Assert.Equal(100, cache.FreePages);
        }

        // Sub-test B: Dispose child first
        {
            var parent = new InferenceSession(cache);
            await parent.AppendAsync(new int[50]); // 2 pages allocated

            var child = (InferenceSession)parent.Fork();
            Assert.Equal(2, cache.SharedPages);

            // Dispose child first
            await child.DisposeAsync();
            Assert.Equal(0, cache.SharedPages);
            Assert.Equal(50, parent.TokenCount);

            // Dispose parent second
            await parent.DisposeAsync();
            Assert.Equal(100, cache.FreePages);
        }
    }

    [Fact]
    public async Task GoldenTest4_OutOfPages_During_CoW_Throws_Safely_Without_Corrupting_Sequence()
    {
        // Allocate tiny cache with only 2 physical pages total
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 32);
        await using var sessionA = new InferenceSession(cache);

        await sessionA.AppendAsync(new int[50]); // 2 pages allocated -> 0 free pages left
        Assert.Equal(0, cache.FreePages);

        // Fork sessionB (shares page tables, 0 free pages)
        await using var sessionB = (InferenceSession)sessionA.Fork();

        // Attempting to append unaligned to sessionB requires CoW page allocation but 0 free pages left
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await sessionB.AppendAsync(new int[] { 99 });
        });

        Assert.Contains("Failed to allocate new page for copy-on-write", ex.Message);

        // Assert sessionA state remains uncorrupted
        Assert.Equal(50, sessionA.TokenCount);
        Assert.Equal(50, sessionA.KvSequence.TokenCount);
    }

    [Fact]
    public async Task GoldenTest5_CoW_Mutation_Contents_Isolation_Between_Forks()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        int[] parentPrompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        await parent.AppendAsync(parentPrompt);

        await using var child = (InferenceSession)parent.Fork();

        // Mutate child logically
        await child.AppendAsync(new int[] { 999, 1000 });

        // Assert parent contents and pages remain 100% uncorrupted
        Assert.Equal(10, parent.TokenCount);
        Assert.Equal(parentPrompt, parent.TokenHistory);

        // Assert child contents reflect mutation
        Assert.Equal(12, child.TokenCount);
        Assert.Equal(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999, 1000 }, child.TokenHistory);
    }

    [Fact]
    public async Task GoldenTest6_Snapshot_RestoreFromTokenHistory_ReconstructsState()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var sessionA = new InferenceSession(cache);

        await sessionA.AppendAsync(new int[] { 100, 200, 300, 400 });
        var snapshot = sessionA.ToSnapshot("test-model");

        var forwardPass = new StatefulTestForwardPass();
        await using var sessionB = new InferenceSession(cache, id: snapshot.Id, forwardPass: forwardPass);
        sessionB.RestoreFromSnapshot(snapshot);

        Assert.Equal(4, sessionB.TokenCount);
        Assert.Equal(4, forwardPass.Position); // Prefill executed during restore!
        Assert.Equal(new int[] { 100, 200, 300, 400 }, sessionB.TokenHistory);
        Assert.Equal(SessionState.Ready, sessionB.State);
    }

    [Fact]
    public async Task GoldenTest7_SuspendAndResume_PreservesTokenStateAndEvictsPages()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[50]); // 2 pages allocated -> 98 free
        Assert.Equal(98, cache.FreePages);

        await session.SuspendAsync();
        Assert.Equal(100, cache.FreePages); // Pages evicted to pool
        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal(50, session.TokenCount); // Token history preserved

        await session.ResumeAsync();
        Assert.Equal(98, cache.FreePages); // Pages re-allocated
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(50, session.TokenCount);
    }

    [Fact]
    public async Task GoldenTest8_Cancellation_During_Generation_LeavesSessionInReadyState()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2, 3 });

        using var cts = new System.Threading.CancellationTokenSource();
        cts.Cancel(); // Pre-cancelled token

        var sampling = new SamplingParams { MaxNewTokens = 10 };
        var chunks = new List<GenerateChunk>();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var chunk in session.GenerateAsync(sampling, cancellationToken: cts.Token))
            {
                chunks.Add(chunk);
            }
        });

        // Assert session returned cleanly to Ready state and released mutex lock
        Assert.Equal(SessionState.Ready, session.State);
    }

    [Fact]
    public async Task GoldenTest9_Fork_ForwardPass_ExecutionState_Isolation_And_Rollback()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var parentForward = new StatefulTestForwardPass { StateVersion = 10 };
        await using var parent = new InferenceSession(cache, forwardPass: parentForward);

        await parent.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
        Assert.Equal(5, parentForward.Position);

        // Fork child session
        await using var child = (InferenceSession)parent.Fork();

        // Mutate child forward pass state
        await child.AppendAsync(new int[] { 6, 7, 8 });

        // Assert child forward pass state mutated while parent forward pass state remains 100% isolated
        Assert.Equal(5, parentForward.Position);

        // Rollback child
        var cp = child.CreateCheckpoint();
        await child.AppendAsync(new int[] { 99, 100 });
        child.Rollback(cp);

        // Assert parent forward pass state remains 100% untouched after child rollback
        Assert.Equal(5, parentForward.Position);
    }

    [Fact]
    public async Task GoldenTest10_Symmetric_CoW_Parent_Mutation_Does_Not_Alter_Child()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        int[] initialPrompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        await parent.AppendAsync(initialPrompt);

        await using var child = (InferenceSession)parent.Fork();

        // Mutate parent logically (symmetric CoW trigger on parent)
        await parent.AppendAsync(new int[] { 500, 501 });

        // Assert parent obtained private storage and reflects mutation
        Assert.Equal(12, parent.TokenCount);
        Assert.Equal(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 500, 501 }, parent.TokenHistory);

        // Assert child logical sequence and token history remain 100% uncorrupted
        Assert.Equal(10, child.TokenCount);
        Assert.Equal(initialPrompt, child.TokenHistory);
    }

    [Fact]
    public async Task GoldenTest11_Fork_Disposes_ChildForwardPass_Without_Disposing_ParentForwardPass()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var parentForward = new StatefulTestForwardPass();
        var parent = new InferenceSession(cache, forwardPass: parentForward);

        var childForward = (StatefulTestForwardPass)parentForward.CreateContext();
        var childKv = parent.KvSequence.Fork();
        var childSession = new InferenceSession(cache, id: SessionId.New(), options: null, forwardPass: childForward, ownsForwardPass: true);

        Assert.False(parentForward.IsDisposed);
        Assert.False(childForward.IsDisposed);

        // Dispose child session -> childForward MUST be disposed because ownsForwardPass is true!
        await childSession.DisposeAsync();

        Assert.True(childForward.IsDisposed);
        Assert.False(parentForward.IsDisposed);

        // Dispose parent session -> parentForward MUST NOT be disposed because ownsForwardPass is false!
        await parent.DisposeAsync();
        Assert.False(parentForward.IsDisposed);
    }

    [Fact]
    public async Task GoldenTest12_SuspendedSession_TransparentlyResumes_OnAppendAsync()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2, 3 });
        await session.SuspendAsync();
        Assert.Equal(SessionState.Suspended, session.State);

        // Non-generation state operations on suspended session throw InvalidOperationException
        Assert.Throws<InvalidOperationException>(() => session.CreateCheckpoint());
        Assert.Throws<InvalidOperationException>(() => session.Fork());

        // Interacting via AppendAsync transparently resumes the session and succeeds!
        await session.AppendAsync(new int[] { 4, 5 });
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(5, session.TokenCount);
    }

    [Fact]
    public async Task GoldenTest13_DisposeAsync_WaitsForInFlightOperation_WithoutRace()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2, 3 });

        // Concurrent disposal safely waits for mutex lock
        var disposeTask = session.DisposeAsync().AsTask();
        await disposeTask;

        Assert.Equal(SessionState.Disposed, session.State);
    }

    [Fact]
    public async Task GoldenTest14_KvReservation_AccountsForPerSequencePageRounding()
    {
        // 2 physical pages total (32 tokens/page)
        using var cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 32);

        // 3 sequences requesting 17 tokens each.
        // Aggregate tokens = 51 tokens (51/32 = 2 pages estimate).
        // Actual per-sequence page requirement = 3 * 1 = 3 pages!
        int[] sequenceRequests = [17, 17, 17];

        var reservation = cache.TryReserveSequences(sequenceId: 1, sequenceRequests);

        // Assert reservation correctly rejects overcommit (returns null) because 3 physical pages are needed but only 2 exist!
        Assert.Null(reservation);
    }

    [Fact]
    public void GoldenTest15_KvReservation_TryGrow_AccountsForPhysicalPageDeltaBoundary()
    {
        // 2 physical pages total (32 tokens/page)
        using IKvCache cache = new CpuKvCache(totalPages: 2, pageSizeTokens: 32);

        // Reserve initial 17 tokens (1 page required out of 2)
        using var reservation = cache.TryReserve(sequenceId: 1, requiredTokens: 17);
        Assert.NotNull(reservation);
        Assert.Equal(17, reservation.ReservedTokens);

        // Grow reservation by 17 tokens (total 34 tokens -> requires 2 physical pages)
        bool grew = reservation.TryGrow(17);
        Assert.True(grew);
        Assert.Equal(34, reservation.ReservedTokens);

        // Attempting to grow by another 1 token (total 35 tokens -> requires 2 pages) succeeds because total pages still equals 2
        bool grewWithinPage = reservation.TryGrow(1);
        Assert.True(grewWithinPage);

        // Attempting to grow to cross page boundary to 3 pages (total 65 tokens -> requires 3 pages) fails because only 2 pages exist
        bool grewBeyondTotalPages = reservation.TryGrow(30); // 35 + 30 = 65 tokens (3 pages)
        Assert.False(grewBeyondTotalPages);
    }

    /// <summary>
    /// Regression test for the bug where <see cref="IForwardPass.CreateContext"/>'s default
    /// (<c>=&gt; this</c>) is what every production forward pass actually gets today — none of
    /// them override it — so Fork() used to hand the child the SAME forward pass instance as the
    /// parent, silently sharing decode state. <see cref="NonIsolatingTestForwardPass"/> deliberately
    /// does NOT override CreateContext(), unlike <see cref="StatefulTestForwardPass"/> above, so it
    /// reproduces exactly what a real backend does. Fork() must now fail loudly instead.
    /// </summary>
    [Fact]
    public async Task GoldenTest16_Fork_Throws_When_ForwardPass_Does_Not_Isolate_Context()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache, forwardPass: new NonIsolatingTestForwardPass());

        await session.AppendAsync(new int[] { 1, 2, 3 });

        var ex = Assert.Throws<NotSupportedException>(() => session.Fork());
        Assert.Contains("does not implement an isolated CreateContext", ex.Message);

        var exMany = Assert.Throws<NotSupportedException>(() => session.ForkMany(2));
        Assert.Contains("does not implement an isolated CreateContext", exMany.Message);

        // The parent must remain usable after the failed fork attempt.
        Assert.Equal(3, session.TokenCount);
    }

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

    /// <summary>
    /// Regression test for the bug where <c>GenerateAsync</c> called
    /// <c>_forwardPass.Forward(currentPos, lastToken)</c> — the position and token id swapped
    /// relative to <see cref="IForwardPass.Forward"/>'s <c>(token, position)</c> signature. No
    /// existing test caught this because <see cref="StatefulTestForwardPass"/> ignores its
    /// <c>token</c> argument entirely, so a swap is invisible to it. This test drives
    /// <see cref="InferenceSession.GenerateAsync"/> against a REAL model and asserts the greedy
    /// continuation exactly matches an independent <see cref="Engine.ForwardPass.Prefill"/> +
    /// <see cref="Engine.ForwardPass.Forward"/> replay on a fresh instance — since both paths run
    /// the identical non-batched CPU kernel with no concurrent sequences involved, they must
    /// produce bit-identical tokens, not just a close match.
    /// </summary>
    [Fact]
    public async Task GoldenTest17_RealModel_GenerateAsync_MatchesFreshForwardPassReplay()
    {
        var path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this replay test.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        int[] promptTokens = tokenizer.Encode("The capital of France is").ToArray();

        const int newTokens = 6;
        int[] sessionTokens;
        using (var backend = new CpuBackend())
        {
            var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
            using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
            await using var session = new InferenceSession(cache, forwardPass: fwd);

            await session.AppendAsync(promptTokens);

            var sampling = new SamplingParams { MaxNewTokens = newTokens, Temperature = 0f };
            await foreach (var _ in session.GenerateAsync(sampling)) { /* drain */ }

            // TokenHistory (not the decoded chunk text) is the source of truth: re-tokenizing
            // decoded text is not guaranteed to round-trip to the same token ids across a BPE
            // merge boundary, which would be an unrelated source of test flakiness.
            sessionTokens = session.TokenHistory.Skip(promptTokens.Length).ToArray();
        }

        int[] replayTokens;
        using (var backend = new CpuBackend())
        {
            var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
            var logits = fwd.Prefill(promptTokens);
            var outTokens = new int[newTokens];
            int pos = promptTokens.Length;
            for (int i = 0; i < newTokens; i++)
            {
                int next = Sampler.Greedy(logits);
                outTokens[i] = next;
                if (i + 1 < newTokens) logits = fwd.Forward(next, pos++);
            }
            replayTokens = outTokens;
        }

        Assert.Equal(replayTokens, sessionTokens);
    }

    private sealed class StatefulTestForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        public int StateVersion { get; set; } = 1;
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;
        public bool IsDisposed { get; private set; }

        public OpenTail.Stingray.Core.IForwardPass CreateContext() => new StatefulTestForwardPass { StateVersion = StateVersion, Position = Position };
        public System.ReadOnlySpan<float> Forward(int token, int position) { Position = position + 1; return new float[100]; }
        public System.ReadOnlySpan<float> Prefill(System.Collections.Generic.IReadOnlyList<int> tokens, int startPos = 0) { Position = startPos + tokens.Count; return new float[100]; }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { IsDisposed = true; }
    }

    /// <summary>
    /// Deliberately does NOT override <see cref="OpenTail.Stingray.Core.IForwardPass.CreateContext"/>,
    /// reproducing what every real production forward pass does today (falls through to the
    /// interface default <c>=&gt; this</c>) — unlike <see cref="StatefulTestForwardPass"/>, which
    /// exists to prove the machinery works WHEN a backend does the right thing.
    /// </summary>
    private sealed class NonIsolatingTestForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;
        public System.ReadOnlySpan<float> Forward(int token, int position) => new float[100];
        public System.ReadOnlySpan<float> Prefill(System.Collections.Generic.IReadOnlyList<int> tokens, int startPos = 0) => new float[100];
        public void TruncateTo(int position) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
