using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class CrossSessionPrefixSynthesisTests
{
    [Fact]
    public async Task Test1_IdenticalPrefixIsSynthesized()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 16);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session1 = new InferenceSession(cache);
        await using var session2 = new InferenceSession(cache);

        // 32 tokens (2 full pages of 16 tokens)
        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = 100 + i;

        await session1.AppendAsync(prompt);
        await session2.AppendAsync(prompt);

        registry.Sessions.Add(session1);
        registry.Sessions.Add(session2);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        int published = synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.True(published > 0);
        Assert.True(metrics.PublishedPrefixes > 0);

        var ns = new PrefixCacheNamespace("default", "page16");
        var match = index.MatchPrefix(ns, prompt);
        Assert.Equal(32, match.MatchedTokenCount);
        Assert.Equal(2, match.MatchedPageCount);
    }

    [Fact]
    public async Task Test2_PartialPageNeverPublished()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 16);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session = new InferenceSession(cache);
        int[] prompt = new int[] { 1, 2, 3, 4, 5, 6, 7 }; // 7 tokens < 16 (1 page)

        await session.AppendAsync(prompt);
        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        int published = synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.Equal(0, published);
        Assert.True(metrics.SkippedPartialPages > 0);
    }

    [Fact]
    public async Task Test3_FailedOrDisposedSessionNeverPublished()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 16);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        var session = new InferenceSession(cache);
        int[] prompt = new int[32];
        for (int i = 0; i < 32; i++) prompt[i] = i;
        await session.AppendAsync(prompt);
        await session.DisposeAsync();

        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        int published = synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.Equal(0, published);
        Assert.True(metrics.SkippedUnstableSessions > 0);
    }

    [Fact]
    public async Task Test4_StartStopBackgroundSynthesizer()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 16);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        var options = new SynthesizerOptions
        {
            ScanInterval = TimeSpan.FromMilliseconds(50)
        };

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry, options);
        await synthesizer.StartAsync();

        await Task.Delay(150);
        await synthesizer.StopAsync();

        var metrics = synthesizer.Metrics;
        Assert.True(metrics.ScansCompleted > 0);
    }

    [Fact]
    public async Task Test5_ExactPageAlignment_Published()
    {
        // PageSize = 32, Tokens = 64 → 2 complete pages → publishes 64 tokens, 2 pages
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session = new InferenceSession(cache);
        int[] prompt = new int[64];
        for (int i = 0; i < 64; i++) prompt[i] = 100 + i;
        await session.AppendAsync(prompt);
        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.Equal(1, metrics.PublishedPrefixes);
        Assert.Equal(2, metrics.PublishedPages);

        var ns = new PrefixCacheNamespace("default", "page32");
        var match = index.MatchPrefix(ns, prompt);
        Assert.Equal(64, match.MatchedTokenCount);
        Assert.Equal(2, match.MatchedPageCount);
    }

    [Fact]
    public async Task Test6_PartialPage_PublishesOnlyCompletedPages()
    {
        // PageSize = 32, Tokens = 65 → 1 complete + 1 partial → publishes 64 tokens, 2 pages
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session = new InferenceSession(cache);
        int[] prompt = new int[65];
        for (int i = 0; i < 65; i++) prompt[i] = 200 + i;
        await session.AppendAsync(prompt);
        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.Equal(1, metrics.PublishedPrefixes);
        Assert.Equal(2, metrics.PublishedPages);

        var ns = new PrefixCacheNamespace("default", "page32");
        var match = index.MatchPrefix(ns, prompt.AsSpan(0, 64).ToArray());
        Assert.Equal(64, match.MatchedTokenCount);
        Assert.Equal(2, match.MatchedPageCount);
    }

    [Fact]
    public async Task Test7_VerySmallSequence_NothingPublished()
    {
        // PageSize = 32, Tokens = 31 → 0 complete pages → skipped (MinSynthesizedTokens = 32 default)
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session = new InferenceSession(cache);
        int[] prompt = new int[31];
        for (int i = 0; i < 31; i++) prompt[i] = i;
        await session.AppendAsync(prompt);
        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        int published = synthesizer.SynthesizeOnce();

        Assert.Equal(0, published);
        Assert.True(synthesizer.Metrics.SkippedPartialPages > 0);
    }

    [Fact]
    public async Task Test8_RegressionPartialPageTokenCount_33Tokens()
    {
        // Regression: PageSize=32, Tokens=33, PageCount=2
        // Old bug: 33/2=16 (wrong) → pageAlignedCount=32, but published pages=32/16=2 (wrong).
        // Correct: pageSize=32, completePageCount=1, pageAlignedCount=32, pages=1.
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session = new InferenceSession(cache);
        int[] prompt = new int[33];
        for (int i = 0; i < 33; i++) prompt[i] = 300 + i;
        await session.AppendAsync(prompt);
        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.Equal(1, metrics.PublishedPrefixes);
        // Must be 1 complete page, not 2 (the old bug produced 2)
        Assert.Equal(1, metrics.PublishedPages);

        var ns = new PrefixCacheNamespace("default", "page32");
        var match = index.MatchPrefix(ns, prompt.AsSpan(0, 32).ToArray());
        Assert.Equal(32, match.MatchedTokenCount);
        Assert.Equal(1, match.MatchedPageCount);
    }

    [Fact]
    public async Task Test9_NonDefaultPageSize_64TokensPerPage()
    {
        // PageSize=64, Tokens=65, PageCount=2 → only 1 complete page → publishes 64 tokens, 1 page
        // Proves synthesizer is not secretly relying on 16 or 32.
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 64);
        using var index = new RadixPrefixTree(cache);
        var registry = new TestSessionRegistry();

        await using var session = new InferenceSession(cache);
        int[] prompt = new int[65];
        for (int i = 0; i < 65; i++) prompt[i] = 400 + i;
        await session.AppendAsync(prompt);
        registry.Sessions.Add(session);

        var synthesizer = new CrossSessionPrefixSynthesizer(index, registry);
        synthesizer.SynthesizeOnce();

        var metrics = synthesizer.Metrics;
        Assert.Equal(1, metrics.PublishedPrefixes);
        Assert.Equal(1, metrics.PublishedPages);

        var ns = new PrefixCacheNamespace("default", "page64");
        var match = index.MatchPrefix(ns, prompt.AsSpan(0, 64).ToArray());
        Assert.Equal(64, match.MatchedTokenCount);
        Assert.Equal(1, match.MatchedPageCount);
    }

    private sealed class TestSessionRegistry : IActiveSessionRegistry
    {
        public List<IInferenceSession> Sessions { get; } = new();
        public IReadOnlyList<IInferenceSession> GetActiveSessionsSnapshot() => Sessions.ToArray();
    }
}
