using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public class InferenceRuntimeTests
{
    [Fact]
    public async Task Runtime_Creation_AllocatesKvCacheAndSessionManager()
    {
        await using var runtime = new InferenceRuntime(totalPages: 200, pageSizeTokens: 32);

        Assert.NotNull(runtime.KvCache);
        Assert.NotNull(runtime.SessionManager);
        Assert.Equal(200, runtime.KvCache.TotalPages);
        Assert.Equal(200, runtime.KvCache.FreePages);
        Assert.Equal(0, runtime.SessionManager.ActiveSessionCount);
    }

    [Fact]
    public async Task Runtime_CreateSessionAsync_RegistersSession()
    {
        await using var runtime = new InferenceRuntime(totalPages: 200, pageSizeTokens: 32);

        var s1 = await runtime.CreateSessionAsync();
        var s2 = await runtime.CreateSessionAsync();

        Assert.Equal(2, runtime.SessionManager.ActiveSessionCount);
        Assert.NotNull(runtime.GetSession(s1.Id));
        Assert.NotNull(runtime.GetSession(s2.Id));

        bool removed = await runtime.RemoveSessionAsync(s1.Id);
        Assert.True(removed);
        Assert.Equal(1, runtime.SessionManager.ActiveSessionCount);
        Assert.Null(runtime.GetSession(s1.Id));
    }

    [Fact]
    public async Task Runtime_MultiSession_IsolationAndModelSharing()
    {
        await using var runtime = new InferenceRuntime(totalPages: 200, pageSizeTokens: 32);

        var s1 = await runtime.CreateSessionAsync();
        var s2 = await runtime.CreateSessionAsync();

        await s1.AppendAsync(new int[] { 1, 2, 3 });
        await s2.AppendAsync(new int[] { 10, 20, 30, 40, 50 });

        Assert.Equal(3, s1.TokenCount);
        Assert.Equal(5, s2.TokenCount);
        Assert.Equal(new int[] { 1, 2, 3 }, s1.TokenHistory);
        Assert.Equal(new int[] { 10, 20, 30, 40, 50 }, s2.TokenHistory);
    }

    [Fact]
    public async Task Runtime_Disposal_DisposesActiveSessionsAndFreesPools()
    {
        SessionId id1, id2;
        IKvCache cacheRef;
        {
            await using var runtime = new InferenceRuntime(totalPages: 200, pageSizeTokens: 32);
            cacheRef = runtime.KvCache;

            var s1 = await runtime.CreateSessionAsync();
            var s2 = await runtime.CreateSessionAsync();
            id1 = s1.Id;
            id2 = s2.Id;

            await s1.AppendAsync(new int[50]); // 2 pages allocated
            Assert.Equal(198, runtime.KvCache.FreePages);
        } // Runtime disposal disposes sessions and frees pages

        Assert.Equal(200, cacheRef.FreePages);
    }
}
