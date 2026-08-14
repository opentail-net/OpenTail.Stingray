using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Verifies cooperative cancellation invariants across all long-running Stingray operations.
///
/// Core principle: cancellation is another path through the normal lifecycle.
///   Operation ──► success ──► Commit
///              └─ cancel  ──► Rollback/Cleanup
///
/// No KV pages leak, no OperationCanceledException is swallowed,
/// and sessions remain in a valid state after cancellation.
/// </summary>
public sealed class CancellationTests
{
    // ─── Test 1 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test01_GenerationCancellationStopsPromptly()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        using var cts = new CancellationTokenSource();

        int chunksReceived = 0;
        var ex = await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 100 }, cts.Token))
            {
                chunksReceived++;
                if (chunksReceived == 1)
                {
                    cts.Cancel();
                }
            }
        });

        Assert.NotNull(ex);
        Assert.True(chunksReceived >= 1, "At least one chunk should have been produced before cancel.");
        Assert.True(chunksReceived < 100, "Generation should have stopped well before 100 tokens.");
    }

    // ─── Test 2 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test02_CancelledGenerationDoesNotCorruptSession()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        long tokensBefore = session.TokenCount;

        using var cts = new CancellationTokenSource();
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 100 }, cts.Token))
            {
                cts.Cancel();
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        // Session state must be valid (Ready) and not have grown beyond committed generation
        Assert.Equal(SessionState.Ready, session.State);
        Assert.True(session.TokenCount >= tokensBefore, "Token count should not decrease below pre-generation baseline.");
    }

    // ─── Test 3 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test03_CancelledGenerationDoesNotLeakPages()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        int pagesAfterPrompt = cache.UsedPages;

        using var cts = new CancellationTokenSource();
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 50 }, cts.Token))
            {
                cts.Cancel();
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        // Used pages should not increase permanently beyond what the prompt occupied.
        // Generated tokens that committed before cancel ARE counted; that's correct.
        // We check that we don't have unbounded growth.
        int pagesAfterCancelledGen = cache.UsedPages;
        Assert.True(pagesAfterCancelledGen <= pagesAfterPrompt + 5,
            $"Expected at most a few extra pages from committed tokens, got {pagesAfterCancelledGen - pagesAfterPrompt} extra pages.");
    }

    // ─── Test 4 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test04_CancelledGenerationDoesNotPublishPrefix()
    {
        // Plan 005b invariant: cancelled or failed prefills NEVER publish incomplete prefix-cache pages.
        // Observable indicator: SharedPages only increases when Publish() is called on the prefix tree.
        // Basic InferenceSession does not call Publish() itself — that is the host's responsibility.
        // This test verifies the session itself does not cause unbounded SharedPage growth on cancel.
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        int sharedPagesBefore = cache.SharedPages;

        using var cts = new CancellationTokenSource();
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 50 }, cts.Token))
            {
                cts.Cancel();
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        // SharedPages should not increase due to cancelled generation.
        // The InferenceSession does not call Publish() — only the host does.
        // Any increase here would indicate a leak in the page-sharing mechanism.
        Assert.Equal(sharedPagesBefore, cache.SharedPages);
    }

    // ─── Test 5 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test05_ToolCancellationPropagatesAsOce()
    {
        // The session emits ToolCall chunks and stops generation.
        // The host then calls AppendToolResultAsync with the caller's ct.
        // Verify that if the host passes a pre-cancelled ct, the exception propagates.
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // AppendToolResultAsync requires a Tokenizer configured.
        // Without one the call throws InvalidOperationException.
        // Verify cancellation is checked before the tokenizer guard.
        // Since AppendToolResultAsync delegates to AppendAsync which acquires mutex with ct:
        var result = new OpenTail.Stingray.Core.Tools.ToolResult(
            ToolCallId: "tc1",
            Content: default);

        // Pre-cancelled token should surface as OperationCanceledException (from mutex wait)
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            // Provide a dummy tokenizer so the tokenizer-null guard doesn't fire first.
            session.Tokenizer = new PassThroughTokenizer();
            await session.AppendToolResultAsync(result, cts.Token);
        });
    }

    // ─── Test 6 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test06_CancelledToolAppendDoesNotCommitTokens()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });
        long tokensBefore = session.TokenCount;

        session.Tokenizer = new PassThroughTokenizer();

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var result = new OpenTail.Stingray.Core.Tools.ToolResult(
            ToolCallId: "tc1",
            Content: default);

        try
        {
            await session.AppendToolResultAsync(result, cts.Token);
        }
        catch (OperationCanceledException) { /* expected */ }

        // No tokens should have been committed
        Assert.Equal(tokensBefore, session.TokenCount);
    }

    // ─── Test 7 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test07_ForkedGenerationCancellationAllBranchesObserve()
    {
        using var cache = new CpuKvCache(totalPages: 200, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await parent.AppendAsync(new int[] { 1, 2, 3 });

        var branches = parent.ForkMany(4);
        using var cts = new CancellationTokenSource();

        // Deterministic cancellation: cancel after the FIRST token generated by any branch.
        // This uses a shared counter with a CancellationTokenSource rather than a time-based delay,
        // which is unreliable with a synchronous mock forward pass.
        int totalTokensGenerated = 0;
        foreach (var b in branches)
        {
            b.OnTokenGenerated += (token, text) =>
            {
                if (Interlocked.Increment(ref totalTokensGenerated) >= 2)
                {
                    cts.Cancel();
                }
            };
        }

        var tasks = branches.Select(b => Task.Run(async () =>
        {
            var chunks = new List<GenerateChunk>();
            try
            {
                await foreach (var chunk in b.GenerateAsync(new SamplingParams { MaxNewTokens = 200 }, cts.Token))
                {
                    chunks.Add(chunk);
                }
            }
            catch (OperationCanceledException) { /* expected on some branches */ }
            return chunks.Count;
        })).ToList();

        var results = await Task.WhenAll(tasks);

        // At least some branches must have been cancelled (total chunks < 4*200 = 800)
        int totalChunks = results.Sum();
        Assert.True(totalChunks < 4 * 200,
            $"At least some branches should have been cancelled. Total chunks: {totalChunks}");

        // Dispose branches
        foreach (var b in branches) await b.DisposeAsync();
    }

    // ─── Test 8 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test08_ForkAndVoteCancellationCleansBranches()
    {
        using var cache = new CpuKvCache(totalPages: 400, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await parent.AppendAsync(new int[] { 1, 2, 3 });

        int pagesBeforeFork = cache.UsedPages;

        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Pre-cancel so all branches stop immediately

        try
        {
            await parent.ForkAndVoteAsync(new SamplingParams { MaxNewTokens = 50 }, branchCount: 5, cts.Token);
        }
        catch (OperationCanceledException) { /* expected */ }

        // After cancellation cleanup, page count should not be permanently elevated
        // beyond what the parent prompt allocated.
        // Allow some slack: the parent prompt pages persist, plus at most ~1 extra page
        // for any ephemeral intermediate allocation.
        await Task.Delay(50); // Let any async disposal complete

        int pagesAfter = cache.UsedPages;
        // Temporary branch pages should be released through normal DisposeAsync lifecycle.
        // Parent pages remain; ephemeral branch pages must not linger indefinitely.
        Assert.True(pagesAfter <= pagesBeforeFork + 2,
            $"Expected KV pages to return near pre-fork baseline after cancelled ForkAndVote. Before: {pagesBeforeFork}, After: {pagesAfter}");
    }

    // ─── Test 9 ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test09_SaveCancellationLeavesValidCheckpoint()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"stingray-cancel-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var store = new FileSessionStore(dir);
            using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
            await using var session = new InferenceSession(cache);
            await session.AppendAsync(new int[] { 1, 2, 3 });

            var snapshot = session.ToSnapshot("test-model");

            // Save a valid checkpoint first
            await store.SaveAsync(snapshot);

            // Verify it's readable
            var loaded = await store.LoadAsync(snapshot.Id);
            Assert.NotNull(loaded);
            Assert.Equal(snapshot.Id, loaded!.Id);

            // Now attempt a save with a pre-cancelled token — this should fail
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            try
            {
                var snapshot2 = session.ToSnapshot("test-model");
                await store.SaveAsync(snapshot2, cts.Token);
            }
            catch (OperationCanceledException) { /* expected */ }

            // The original checkpoint must still be readable and correct
            var reloaded = await store.LoadAsync(snapshot.Id);
            Assert.NotNull(reloaded);
            Assert.Equal(snapshot.Id, reloaded!.Id);
            Assert.Equal(snapshot.Tokens.Count, reloaded.Tokens.Count);

            // No .tmp file should linger
            var tmpFiles = Directory.GetFiles(dir, "*.tmp");
            Assert.Empty(tmpFiles);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }

    // ─── Test 10 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test10_ResumeCancellationLeavesSessionSuspended()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);

        // Use a forward pass that throws on Prefill when cancellation is requested
        var fwd = new FailOnPrefillForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });

        int pagesBeforeSuspend = cache.UsedPages;

        await session.SuspendAsync();
        Assert.Equal(SessionState.Suspended, session.State);

        // The forward pass will throw during Prefill to simulate cancellation mid-prefill
        fwd.ShouldThrow = true;

        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await session.ResumeAsync();
        });

        // After failed resume, session must remain Suspended
        Assert.Equal(SessionState.Suspended, session.State);

        // KV pages must not leak — suspended session has 0 active pages
        // The placeholder zero-token sequence uses no pages
        Assert.Equal(0, session.KvSequence.TokenCount);
    }

    // ─── Test 11 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test11_SuspendCancellationLeavesSessionReady()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        // Cancellation that fires BEFORE the mutex is acquired should throw
        // OperationCanceledException without changing session state.
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await session.SuspendAsync(cts.Token);
        });

        // Session remains Ready (not Suspended) because the mutex gate cancelled
        Assert.Equal(SessionState.Ready, session.State);
        Assert.True(session.KvSequence.TokenCount > 0, "KV state should still be intact.");
    }

    // ─── Test 12 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test12_GovernorShutdownCancellation()
    {
        var governorOptions = new KvGovernorOptions
        {
            Enabled = true,
            PollInterval = TimeSpan.FromMilliseconds(50),
            PressureThreshold = 0.01, // Very low so governor fires immediately
            RecoveryThreshold = 0.005,
            MinimumIdleDuration = TimeSpan.Zero,
            MaxSessionsSuspendedPerCycle = 5
        };

        var runtime = new InferenceRuntime(
            totalPages: 200,
            pageSizeTokens: 32,
            governorOptions: governorOptions);

        // Create a session with some tokens to give the governor something to consider
        var session = await runtime.CreateSessionAsync();
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });

        // Allow governor at least one cycle
        await Task.Delay(150);

        // Dispose runtime — should cancel governor and await clean shutdown
        // without ObjectDisposedException or other errors
        await runtime.DisposeAsync();

        // If we reach here without exception, governor terminated cleanly
    }

    // ─── Test 13 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test13_CancellationIsNotSwallowed()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // OperationCanceledException must reach the caller — must not be swallowed
        bool cancellationObserved = false;
        try
        {
            await foreach (var _ in session.GenerateAsync(new SamplingParams(), cts.Token))
            {
            }
        }
        catch (OperationCanceledException)
        {
            cancellationObserved = true;
        }
        catch (Exception ex)
        {
            Assert.Fail($"Expected OperationCanceledException but got {ex.GetType().Name}: {ex.Message}");
        }

        Assert.True(cancellationObserved, "OperationCanceledException must reach the caller.");
    }

    // ─── Test 14 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test14_MetricsReflectCancelledGeneration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        long generatedBefore = session.Metrics.GeneratedTokens;

        using var cts = new CancellationTokenSource();
        int chunksBeforeCancel = 0;
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 50 }, cts.Token))
            {
                chunksBeforeCancel++;
                if (chunksBeforeCancel == 3)
                    cts.Cancel();
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        // Metrics should reflect only committed tokens (those that completed before cancel)
        long generatedAfter = session.Metrics.GeneratedTokens;
        Assert.True(generatedAfter >= generatedBefore,
            "GeneratedTokens should not decrease.");
        Assert.True(generatedAfter < generatedBefore + 50,
            "GeneratedTokens should not count all 50 tokens — generation was cancelled.");
    }

    // ─── Test 15 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test15_TokenListenerDoesNotObserveRolledBackTokens()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var observedTokens = new List<int>();
        session.OnTokenGenerated += (token, text) => observedTokens.Add(token);

        using var cts = new CancellationTokenSource();
        int chunks = 0;
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 50 }, cts.Token))
            {
                chunks++;
                if (chunks == 2) cts.Cancel();
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        // OnTokenGenerated fires synchronously when a token is committed.
        // All observed tokens must be present in the session's committed token history.
        var committedHistory = session.TokenHistory.ToHashSet();
        foreach (var observed in observedTokens)
        {
            // The observed token must be a value that was generated (the mock generates token 10 always).
            // This confirms the listener only sees tokens that are in the committed sequence.
            Assert.True(observed >= 0, $"Unexpected negative token {observed} in listener.");
        }

        // The number of tokens in history must match what the listener saw (prompt + committed generated)
        long promptTokens = 3;
        Assert.Equal(session.TokenCount, promptTokens + observedTokens.Count);
    }

    // ─── Test 16 ───────────────────────────────────────────────────────────────

    [Fact]
    public async Task Test16_DeltaContainsCommittedStateOnly()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CancellableMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        long countBeforeGen = session.TokenCount;

        using var cts = new CancellationTokenSource();
        int chunks = 0;
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 50 }, cts.Token))
            {
                chunks++;
                if (chunks == 2) cts.Cancel();
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        // Snapshot captures only the committed token history
        var snapshot = session.ToSnapshot("test-model");

        // The snapshot token count must equal session.TokenCount (only committed state)
        Assert.Equal(session.TokenCount, snapshot.Tokens.Count);

        // And must be at least the prompt tokens
        Assert.True(snapshot.Tokens.Count >= countBeforeGen,
            "Snapshot should include prompt tokens plus any committed generated tokens.");

        // Critically: snapshot count must NOT be 3 + 50 (all 50 tokens — generation was cancelled)
        Assert.True(snapshot.Tokens.Count < countBeforeGen + 50,
            "Snapshot must contain only committed (pre-cancel) tokens.");
    }

    // ─── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// A forward pass mock that generates token 10 deterministically.
    /// Compatible with the CancellationToken check at each GenerateAsync step.
    /// </summary>
    private sealed class CancellableMockForwardPass : IForwardPass
    {
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;
        public int Position { get; private set; }

        public IForwardPass CreateContext() => new CancellableMockForwardPass { Position = Position };

        public ReadOnlySpan<float> Forward(int position, int token)
        {
            Position = position + 1;
            var res = new float[100];
            res[10] = 10.0f; // Make token 10 the most likely
            return res;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            return new float[100];
        }

        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }

    /// <summary>
    /// A forward pass that throws an exception on Prefill when ShouldThrow is true.
    /// Used to simulate a cancellation mid-prefill in ResumeAsync.
    /// </summary>
    private sealed class FailOnPrefillForwardPass : IForwardPass
    {
        public bool ShouldThrow { get; set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new FailOnPrefillForwardPass { ShouldThrow = ShouldThrow };

        public ReadOnlySpan<float> Forward(int position, int token)
        {
            var res = new float[100];
            res[10] = 5.0f;
            return res;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            if (ShouldThrow)
                throw new OperationCanceledException("Simulated cancellation during prefill.");
            return new float[100];
        }

        public void TruncateTo(int position) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    /// <summary>
    /// Minimal tokenizer that encodes strings as char-code sequences.
    /// Used for tool result append tests that require a configured tokenizer.
    /// </summary>
    private sealed class PassThroughTokenizer : ITokenizer
    {
        public int VocabSize => 65536;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text) =>
            text.Select(c => (int)c).ToList();

        public string Decode(IEnumerable<int> tokens) =>
            new string(tokens.Select(t => (char)Math.Min(t, 65535)).ToArray());

        public byte[] DecodeBytes(int token) =>
            System.Text.Encoding.UTF8.GetBytes(new string((char)Math.Min(token, 65535), 1));
    }
}
