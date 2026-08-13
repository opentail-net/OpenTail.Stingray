using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Unit tests for <see cref="SessionDelta"/>, <see cref="SessionDeltaWireCompressor"/>, and incremental session store persistence.
/// </summary>
public sealed class SessionDeltaTests
{
    [Fact]
    public async Task Test01_CreateBasicDelta()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 }); // 5 prompt tokens

        var baseToken = session.GetContinuationToken();

        await session.AppendAsync(new int[] { 6, 7, 8 }); // 3 more tokens

        var delta = session.CreateDelta(baseToken);

        Assert.NotNull(delta);
        Assert.Equal(session.Id, delta.SessionId);
        Assert.Equal(baseToken, delta.BaseToken);
        Assert.Equal(session.TokenCount, delta.ResultToken.TokenPosition);
        Assert.Equal(new int[] { 6, 7, 8 }, delta.AppendedTokens);
    }

    [Fact]
    public async Task Test02_DeltaTokenRange()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 10, 20 });

        var baseToken = session.GetContinuationToken();

        await session.AppendAsync(new int[] { 30, 40, 50, 60 });
        var delta = session.CreateDelta(baseToken);

        Assert.Equal(2, delta.BaseToken.TokenPosition);
        Assert.Equal(6, delta.ResultToken.TokenPosition);
        Assert.Equal(4, delta.AppendedTokens.Count);
        Assert.Equal(new int[] { 30, 40, 50, 60 }, delta.AppendedTokens);
    }

    [Fact]
    public async Task Test03_ApplyDelta()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var sessionId = SessionId.New();

        await using var sessionA = new InferenceSession(cache, id: sessionId);
        await sessionA.AppendAsync(new int[] { 1, 2, 3 });
        var baseToken = sessionA.GetContinuationToken();

        await sessionA.AppendAsync(new int[] { 4, 5, 6 });
        var delta = sessionA.CreateDelta(baseToken);

        // Recreate target session B starting at baseToken state
        await using var sessionB = new InferenceSession(cache, id: sessionId);
        await sessionB.AppendAsync(new int[] { 1, 2, 3 });

        await sessionB.ApplyDeltaAsync(delta);

        Assert.Equal(6, sessionB.TokenCount);
        Assert.Equal(sessionA.TokenHistory, sessionB.TokenHistory);
    }

    [Fact]
    public async Task Test04_WrongSessionRejected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var sessionA = new InferenceSession(cache);
        await sessionA.AppendAsync(new int[] { 1, 2 });
        var baseTokenA = sessionA.GetContinuationToken();

        await sessionA.AppendAsync(new int[] { 3, 4 });
        var deltaA = sessionA.CreateDelta(baseTokenA);

        await using var sessionB = new InferenceSession(cache); // different SessionId
        await sessionB.AppendAsync(new int[] { 1, 2 });

        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await sessionB.ApplyDeltaAsync(deltaA);
        });
    }

    [Fact]
    public async Task Test05_FutureTokenRejected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var futureToken = new ResponseContinuationToken(session.Id, 100, 1);

        Assert.Throws<ArgumentException>(() =>
        {
            session.CreateDelta(futureToken);
        });
    }

    [Fact]
    public async Task Test06_StaleBaseDetected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var sessionId = SessionId.New();
        await using var session = new InferenceSession(cache, id: sessionId);
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });

        // Delta constructed from base at pos 2
        var staleBase = new ResponseContinuationToken(sessionId, 2, 1);
        var resultToken = new ResponseContinuationToken(sessionId, 5, 1);
        var delta = new SessionDelta
        {
            SessionId = sessionId,
            BaseToken = staleBase,
            ResultToken = resultToken,
            AppendedTokens = new int[] { 3, 4, 5 }
        };

        // Applying delta (base pos 2) to session at pos 5 throws StaleContinuationException
        await Assert.ThrowsAsync<StaleContinuationException>(async () =>
        {
            await session.ApplyDeltaAsync(delta);
        });
    }

    [Fact]
    public async Task Test07_MetadataDelta()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var sessionId = SessionId.New();
        await using var sessionA = new InferenceSession(cache, id: sessionId);
        sessionA.Metadata["user"] = "alice";
        await sessionA.AppendAsync(new int[] { 1, 2 });
        var baseToken = sessionA.GetContinuationToken();

        sessionA.Metadata["workflow"] = "code_review";
        sessionA.Metadata["phase"] = "executing";
        await sessionA.AppendAsync(new int[] { 3, 4 });

        var delta = sessionA.CreateDelta(baseToken);

        await using var sessionB = new InferenceSession(cache, id: sessionId);
        sessionB.Metadata["user"] = "alice";
        await sessionB.AppendAsync(new int[] { 1, 2 });

        await sessionB.ApplyDeltaAsync(delta);

        Assert.Equal("code_review", sessionB.Metadata.Get<string>("workflow"));
        Assert.Equal("executing", sessionB.Metadata.Get<string>("phase"));
    }

    [Fact]
    public async Task Test08_ApplyDeltaIdempotent()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var sessionId = SessionId.New();
        await using var sessionA = new InferenceSession(cache, id: sessionId);
        await sessionA.AppendAsync(new int[] { 1, 2 });
        var baseToken = sessionA.GetContinuationToken();

        await sessionA.AppendAsync(new int[] { 3, 4, 5 });
        var delta = sessionA.CreateDelta(baseToken);

        await using var sessionB = new InferenceSession(cache, id: sessionId);
        await sessionB.AppendAsync(new int[] { 1, 2 });

        // First apply
        await sessionB.ApplyDeltaAsync(delta);
        Assert.Equal(5, sessionB.TokenCount);

        // Second apply (same delta, already at resultToken) -> no-op!
        await sessionB.ApplyDeltaAsync(delta);
        Assert.Equal(5, sessionB.TokenCount);
        Assert.Equal(sessionA.TokenHistory, sessionB.TokenHistory);
    }

    [Fact]
    public async Task Test09_ForkDeltaIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 200, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);
        await parent.AppendAsync(new int[] { 1, 2, 3 });

        await using var branchA = (InferenceSession)parent.Fork();
        await using var branchB = (InferenceSession)parent.Fork();

        var baseA = branchA.GetContinuationToken();
        await branchA.AppendAsync(new int[] { 10, 20 });
        var deltaA = branchA.CreateDelta(baseA);

        // Branch B has a different SessionId than branch A
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await branchB.ApplyDeltaAsync(deltaA);
        });
    }

    [Fact]
    public async Task Test10_DuplicateDeltaSafe()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 1, 2, 3 });
        var baseToken = session.GetContinuationToken();

        await session.AppendAsync(new int[] { 4, 5 });
        var delta = session.CreateDelta(baseToken);

        Assert.Equal(5, session.TokenCount);

        // Applying delta to the source session itself (whose position is already resultToken) is safe & no-op
        await session.ApplyDeltaAsync(delta);
        Assert.Equal(5, session.TokenCount);
    }

    [Fact]
    public async Task Test11_AtomicApply()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var sessionId = SessionId.New();
        await using var session = new InferenceSession(cache, id: sessionId);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var invalidBaseDelta = new SessionDelta
        {
            SessionId = sessionId,
            BaseToken = new ResponseContinuationToken(sessionId, 999, 1),
            ResultToken = new ResponseContinuationToken(sessionId, 1000, 1),
            AppendedTokens = new int[] { 100 }
        };

        // Application fails during validation before any state modification
        await Assert.ThrowsAsync<StaleContinuationException>(async () =>
        {
            await session.ApplyDeltaAsync(invalidBaseDelta);
        });

        Assert.Equal(3, session.TokenCount);
        Assert.Equal(new int[] { 1, 2, 3 }, session.TokenHistory);
    }

    [Fact]
    public async Task Test12_FullCheckpointFallback()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        await session.AppendAsync(new int[] { 10, 20, 30 });

        var checkpoint = session.CreateCheckpoint();
        Assert.Equal(3, checkpoint.TokenPosition);

        var baseToken = session.GetContinuationToken();
        await session.AppendAsync(new int[] { 40, 50 });
        var delta = session.CreateDelta(baseToken);

        // Checkpoint and delta can coexist cleanly
        Assert.Equal(3, checkpoint.CommittedTokens.Count);
        Assert.Equal(2, delta.AppendedTokens.Count);
    }

    [Fact]
    public void Test13_CompressionRoundTripBrotli()
    {
        var sessionId = SessionId.New();
        var baseToken = new ResponseContinuationToken(sessionId, 100, 2);
        var resultToken = new ResponseContinuationToken(sessionId, 105, 3);
        var dict = new Dictionary<string, string?> { ["env"] = "prod", ["tier"] = "hot" };

        var delta = new SessionDelta
        {
            SessionId = sessionId,
            BaseToken = baseToken,
            ResultToken = resultToken,
            AppendedTokens = new int[] { 10, 20, 30, 40, 50 },
            MetadataChanges = System.Collections.Immutable.ImmutableDictionary.ToImmutableDictionary(dict)
        };

        byte[] compressed = SessionDeltaWireCompressor.Compress(delta, CompressionAlgorithm.Brotli);
        Assert.NotNull(compressed);
        Assert.True(compressed.Length > 0);

        var decompressed = SessionDeltaWireCompressor.Decompress(compressed, CompressionAlgorithm.Brotli);
        Assert.NotNull(decompressed);
        Assert.Equal(delta.SessionId, decompressed.SessionId);
        Assert.Equal(delta.BaseToken, decompressed.BaseToken);
        Assert.Equal(delta.ResultToken, decompressed.ResultToken);
        Assert.Equal(delta.AppendedTokens, decompressed.AppendedTokens);
        Assert.Equal("prod", decompressed.MetadataChanges["env"]);
        Assert.Equal("hot", decompressed.MetadataChanges["tier"]);
    }

    [Fact]
    public void Test14_CompressionRoundTripGzip()
    {
        var sessionId = SessionId.New();
        var baseToken = new ResponseContinuationToken(sessionId, 10, 1);
        var resultToken = new ResponseContinuationToken(sessionId, 15, 1);

        var delta = new SessionDelta
        {
            SessionId = sessionId,
            BaseToken = baseToken,
            ResultToken = resultToken,
            AppendedTokens = new int[] { 100, 200, 300, 400, 500 }
        };

        byte[] compressed = SessionDeltaWireCompressor.Compress(delta, CompressionAlgorithm.GZip);
        var decompressed = SessionDeltaWireCompressor.Decompress(compressed, CompressionAlgorithm.GZip);

        Assert.Equal(delta.SessionId, decompressed.SessionId);
        Assert.Equal(delta.AppendedTokens, decompressed.AppendedTokens);
    }

    [Fact]
    public async Task Test15_FileSessionStore_SaveDeltaAsync()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"stingray-delta-store-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var store = new FileSessionStore(dir);
            using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
            var sessionId = SessionId.New();
            await using var session = new InferenceSession(cache, id: sessionId);

            await session.AppendAsync(new int[] { 1, 2, 3 });
            var snapshot1 = session.ToSnapshot("test-model");
            await store.SaveAsync(snapshot1);

            var baseToken = session.GetContinuationToken();
            await session.AppendAsync(new int[] { 4, 5, 6 });
            var delta = session.CreateDelta(baseToken);

            // Append delta to disk store
            await store.SaveDeltaAsync(delta);

            // Reload snapshot from disk store
            var reloadedSnapshot = await store.LoadAsync(sessionId);
            Assert.NotNull(reloadedSnapshot);
            Assert.Equal(6, reloadedSnapshot!.Position);
            Assert.Equal(new int[] { 1, 2, 3, 4, 5, 6 }, reloadedSnapshot.Tokens);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }
}
