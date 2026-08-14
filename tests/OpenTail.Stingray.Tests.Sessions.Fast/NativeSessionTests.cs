using System;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public class NativeSessionTests
{
    [Fact]
    public async Task Session_Creation_HasValidIdAndStateReady()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        Assert.NotEqual(Guid.Empty, session.Id.Value);
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(0, session.TokenCount);
        Assert.Equal(0, session.KvSequence.TokenCount);
        Assert.Equal(100, cache.FreePages);
    }

    [Fact]
    public async Task Session_Append_MaintainsTokenCountAndKvAlignment()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        int[] promptTokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        await session.AppendAsync(promptTokens);

        Assert.Equal(10, session.TokenCount);
        Assert.Equal(10, session.KvSequence.TokenCount);
        Assert.Equal(1, session.KvSequence.PageCount);
        Assert.Equal(promptTokens, session.TokenHistory);
        Assert.Equal(SessionState.Ready, session.State);
    }

    [Fact]
    public async Task Session_CheckpointAndRollback_RestoresCommittedState()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 10, 20, 30, 40, 50 }); // 5 tokens
        var checkpoint = session.CreateCheckpoint();
        Assert.Equal(5, checkpoint.TokenPosition);

        await session.AppendAsync(new int[] { 60, 70, 80 }); // append 3 tokens -> total 8
        Assert.Equal(8, session.TokenCount);
        Assert.Equal(8, session.KvSequence.TokenCount);

        // Rollback to checkpoint (pos 5)
        session.Rollback(checkpoint);

        Assert.Equal(5, session.TokenCount);
        Assert.Equal(5, session.KvSequence.TokenCount);
        Assert.Equal(new int[] { 10, 20, 30, 40, 50 }, session.TokenHistory);
        Assert.Equal(SessionState.Ready, session.State);
    }

    [Fact]
    public async Task Session_Fork_ZeroCopyPageTableAndMutationIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        await parent.AppendAsync(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
        Assert.Equal(15, parent.TokenCount);

        await using var child = (InferenceSession)parent.Fork();

        Assert.NotEqual(parent.Id, child.Id);
        Assert.Equal(15, child.TokenCount);
        Assert.Equal(15, child.KvSequence.TokenCount);
        Assert.Equal(parent.TokenHistory, child.TokenHistory);
        Assert.Equal(1, cache.SharedPages);

        // Appending to child mutates child only (Copy-On-Write)
        await child.AppendAsync(new int[] { 99, 100 });

        Assert.Equal(17, child.TokenCount);
        Assert.Equal(15, parent.TokenCount);
        Assert.Equal(0, cache.SharedPages); // Child acquired private page copy
    }

    [Fact]
    public async Task InMemorySessionManager_ManagesLifecycleAndDisposal()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var manager = new InMemorySessionManager(cache);

        var s1 = await manager.CreateSessionAsync();
        var s2 = await manager.CreateSessionAsync();

        Assert.Equal(2, manager.ActiveSessionCount);
        Assert.NotNull(manager.GetSession(s1.Id));
        Assert.NotNull(manager.GetSession(s2.Id));

        bool removed = await manager.RemoveSessionAsync(s1.Id);
        Assert.True(removed);
        Assert.Equal(1, manager.ActiveSessionCount);
        Assert.Null(manager.GetSession(s1.Id));
    }

    [Fact]
    public async Task Session_Disposal_FreesKvResources()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);

        SessionId id;
        {
            await using var session = new InferenceSession(cache);
            id = session.Id;
            await session.AppendAsync(new int[50]); // 2 pages allocated
            Assert.Equal(98, cache.FreePages);
        } // DisposeAsync invoked

        Assert.Equal(100, cache.FreePages);
    }

    [Fact]
    public async Task Session_Snapshot_ToSnapshotAndRestore_PreservesState()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var originalSession = new InferenceSession(cache);

        await originalSession.AppendAsync(new int[] { 10, 20, 30, 40, 50 });
        var snapshot = originalSession.ToSnapshot("SmolLM2-1.7B");

        Assert.Equal("SmolLM2-1.7B", snapshot.ModelId);
        Assert.Equal(5, snapshot.Position);
        Assert.Equal(new int[] { 10, 20, 30, 40, 50 }, snapshot.Tokens);

        // Restore snapshot into a new session instance
        await using var restoredSession = new InferenceSession(cache, id: snapshot.Id);
        restoredSession.RestoreFromSnapshot(snapshot);

        Assert.Equal(originalSession.Id, restoredSession.Id);
        Assert.Equal(5, restoredSession.TokenCount);
        Assert.Equal(5, restoredSession.KvSequence.TokenCount);
        Assert.Equal(originalSession.TokenHistory, restoredSession.TokenHistory);
    }

    [Fact]
    public async Task Session_SuspendAndResume_FreesAndRestoresKvPages()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[50]); // 2 pages allocated
        Assert.Equal(98, cache.FreePages);
        Assert.Equal(SessionState.Ready, session.State);

        // Suspend frees resident KV pages
        await session.SuspendAsync();
        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal(100, cache.FreePages);

        // Resume re-allocates KV pages and restores sequence
        await session.ResumeAsync();
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(98, cache.FreePages);
        Assert.Equal(50, session.TokenCount);
        Assert.Equal(50, session.KvSequence.TokenCount);
    }

    [Fact]
    public async Task FileSessionStore_SavesAndLoadsSnapshotFromDisk()
    {
        string tempDir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "stingray_session_tests_" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new FileSessionStore(tempDir);
            using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
            await using var session = new InferenceSession(cache);

            await session.AppendAsync(new int[] { 101, 102, 103, 104, 105 });
            var snapshot = session.ToSnapshot("Qwen2.5-7B");

            // Save to disk
            await store.SaveAsync(snapshot);

            // Load from disk
            var loaded = await store.LoadAsync(session.Id);
            Assert.NotNull(loaded);
            Assert.Equal(session.Id, loaded.Id);
            Assert.Equal("Qwen2.5-7B", loaded.ModelId);
            Assert.Equal(5, loaded.Position);
            Assert.Equal(new int[] { 101, 102, 103, 104, 105 }, loaded.Tokens);

            // Delete from disk
            bool deleted = await store.DeleteAsync(session.Id);
            Assert.True(deleted);

            var emptyLoad = await store.LoadAsync(session.Id);
            Assert.Null(emptyLoad);
        }
        finally
        {
            if (System.IO.Directory.Exists(tempDir))
            {
                System.IO.Directory.Delete(tempDir, recursive: true);
            }
        }
    }
}
