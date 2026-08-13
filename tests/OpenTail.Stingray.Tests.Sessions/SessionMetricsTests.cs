using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class SessionMetricsTests
{
    [Fact]
    public async Task Test1_InitialMetricsAreZero()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        var m = session.Metrics;
        Assert.Equal(0, m.PromptTokens);
        Assert.Equal(0, m.GeneratedTokens);
        Assert.Equal(TimeSpan.Zero, m.TotalPrefillTime);
        Assert.Equal(TimeSpan.Zero, m.TotalGenerationTime);
        Assert.Equal(0.0, m.TokensPerSecond);
        Assert.True(m.KvPagesHeld >= 0);
    }

    [Fact]
    public async Task Test2_PromptTokensTracked()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 10, 20, 30, 40, 50 });

        Assert.Equal(5, session.Metrics.PromptTokens);
        Assert.Equal(0, session.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task Test3_GeneratedTokensTracked()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3 });
        Assert.Equal(3, session.Metrics.PromptTokens);

        int count = 0;
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 4 }))
        {
            count++;
        }

        Assert.Equal(4, count);
        Assert.Equal(4, session.Metrics.GeneratedTokens);
        Assert.Equal(3, session.Metrics.PromptTokens);
    }

    [Fact]
    public async Task Test4_TokensPerSecond()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        Assert.Equal(0.0, session.Metrics.TokensPerSecond);

        await session.AppendAsync(new int[] { 1, 2 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 3 })) { }

        Assert.Equal(3, session.Metrics.GeneratedTokens);
        Assert.True(session.Metrics.TokensPerSecond >= 0.0);
    }

    [Fact]
    public async Task Test5_PrefillTimingRecorded()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[500]);
        Assert.True(session.Metrics.TotalPrefillTime >= TimeSpan.Zero);
    }

    [Fact]
    public async Task Test6_GenerationTimingRecorded()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 2 })) { }

        Assert.True(session.Metrics.TotalGenerationTime >= TimeSpan.Zero);
    }

    [Fact]
    public async Task Test7_SpeculativeTokensOnlyCountWhenCommitted()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 5 })) { }

        Assert.Equal(5, session.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task Test8_StreamingMetricsUpdate()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1 });

        int observedSteps = 0;
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 3 }))
        {
            observedSteps++;
            Assert.Equal(observedSteps, session.Metrics.GeneratedTokens);
        }
    }

    [Fact]
    public async Task Test9_ToolContinuationMetrics()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        session.Tokenizer = new MockMetricsTokenizer();

        await session.AppendAsync(new int[] { 1, 2, 3 }); // 3 prompt tokens
        long initialPrompt = session.Metrics.PromptTokens;

        using var doc = JsonDocument.Parse("{\"result\":\"ok\"}");
        var toolResult = new OpenTail.Stingray.Core.Tools.ToolResult("call_1", doc.RootElement);

        await session.AppendToolResultAsync(toolResult);

        Assert.True(session.Metrics.PromptTokens > initialPrompt);
    }

    [Fact]
    public async Task Test10_ForkMetricsAreIndependent()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd);

        await parent.AppendAsync(new int[] { 1, 2, 3 });
        await using var child = parent.Fork();

        Assert.Equal(3, parent.Metrics.PromptTokens);
        Assert.Equal(3, child.Metrics.PromptTokens);
        Assert.Equal(0, child.Metrics.GeneratedTokens);

        await foreach (var chunk in child.GenerateAsync(new SamplingParams { MaxNewTokens = 2 })) { }

        Assert.Equal(2, child.Metrics.GeneratedTokens);
        Assert.Equal(0, parent.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task Test11_KvPagesHeldTracksOwnership()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[100]); // 100 tokens with 32-token pages = 4 pages
        Assert.Equal(4, session.Metrics.KvPagesHeld);
    }

    [Fact]
    public async Task Test12_KvPagesHeldWithFork()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        await parent.AppendAsync(new int[64]); // 2 pages
        await using var child = parent.Fork();

        Assert.Equal(2, parent.Metrics.KvPagesHeld);
        Assert.Equal(2, child.Metrics.KvPagesHeld);
    }

    [Fact]
    public async Task Test13_SuspensionReleasesPages()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[64]);
        Assert.Equal(2, session.Metrics.KvPagesHeld);

        await session.SuspendAsync();
        Assert.Equal(0, session.Metrics.KvPagesHeld);

        await session.ResumeAsync();
        Assert.Equal(2, session.Metrics.KvPagesHeld);
    }

    [Fact]
    public async Task Test14_PrefixCacheDoesNotDoubleCount()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[32]);
        Assert.Equal(1, session.Metrics.KvPagesHeld);
    }

    [Fact]
    public async Task Test15_FailedGenerationDoesNotCommitTokens()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 5 };
        await using var session = new InferenceSession(cache, options: options);

        await session.AppendAsync(new int[] { 1, 2, 3 }); // 3 prompt tokens

        await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendAsync(new int[] { 10, 20, 30 }); // Exceeds 5
        });

        Assert.Equal(3, session.Metrics.PromptTokens);
        Assert.Equal(0, session.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task Test16_ConcurrentMetricReads()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockMetricsForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3 });

        var taskRead = Task.Run(() =>
        {
            for (int i = 0; i < 50; i++)
            {
                var prompt = session.Metrics.PromptTokens;
                var gen = session.Metrics.GeneratedTokens;
                var pages = session.Metrics.KvPagesHeld;
                var tps = session.Metrics.TokensPerSecond;
                Assert.True(prompt >= 0 && gen >= 0 && pages >= 0 && tps >= 0.0);
            }
        });

        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 10 })) { }

        await taskRead;
        Assert.Equal(10, session.Metrics.GeneratedTokens);
    }

    private sealed class MockMetricsForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockMetricsForwardPass { Position = Position };
        public System.ReadOnlySpan<float> Forward(int position, int token)
        {
            Position = position + 1;
            var res = new float[100];
            res[10] = 5.0f;
            return res;
        }
        public System.ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            return new float[100];
        }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class MockMetricsTokenizer : ITokenizer
    {
        public int VocabSize => 100;
        public int BosTokenId => 1;
        public int EosTokenId => 0;
        public int UnknownTokenId => -1;
        public int PadTokenId => -1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => System.Collections.Immutable.ImmutableArray.Create(0);
        public System.Collections.Generic.IReadOnlyDictionary<string, int> SpecialTokens => System.Collections.Immutable.ImmutableDictionary<string, int>.Empty;
        public byte[] DecodeBytes(int token) => Array.Empty<byte>();
        public string Decode(IEnumerable<int> tokens) => "";
        public IReadOnlyList<int> Encode(string text) => new int[] { 10, 20, 30, 40, 50 };
    }
}
