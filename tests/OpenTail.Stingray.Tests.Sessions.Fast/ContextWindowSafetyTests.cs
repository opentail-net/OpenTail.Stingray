using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class ContextWindowSafetyTests
{
    [Fact]
    public async Task Test1_UnlimitedSession()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache, options: new KvSequenceOptions { MaxContextTokens = null });

        Assert.Null(session.MaxContextTokens);
        Assert.False(session.IsContextLimitReached);
        Assert.Equal(int.MaxValue, session.RemainingContextTokens);

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
        Assert.Equal(5, session.TokenCount);
    }

    [Fact]
    public async Task Test2_ExactLimitAllowed()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 10 };
        await using var session = new InferenceSession(cache, options: options);

        Assert.Equal(10, session.MaxContextTokens);

        // Appending exactly 10 tokens succeeds
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        Assert.Equal(10, session.TokenCount);
        Assert.True(session.IsContextLimitReached);
        Assert.Equal(0, session.RemainingContextTokens);
    }

    [Fact]
    public async Task Test3_OverflowRejected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 10 };
        await using var session = new InferenceSession(cache, options: options);

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }); // 9 tokens

        var ex = await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendAsync(new int[] { 10, 11 }); // Request 2 tokens -> 11 tokens total > 10 limit
        });

        Assert.Equal(9, ex.CurrentTokens);
        Assert.Equal(2, ex.RequestedTokens);
        Assert.Equal(10, ex.MaxContextTokens);
    }

    [Fact]
    public async Task Test4_FailedAppendIsAtomic()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 5 };
        await using var session = new InferenceSession(cache, options: options);

        await session.AppendAsync(new int[] { 1, 2, 3 }); // 3 tokens

        await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendAsync(new int[] { 4, 5, 6 }); // Exceeds 5
        });

        // Sequence must remain exactly as it was before the failed append (3 tokens)
        Assert.Equal(3, session.TokenCount);
        Assert.Equal(new int[] { 1, 2, 3 }, session.TokenHistory);
        Assert.Equal(3, session.KvSequence.TokenCount);
    }

    [Fact]
    public async Task Test5_IsContextLimitReached()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 4 };
        await using var session = new InferenceSession(cache, options: options);

        Assert.False(session.IsContextLimitReached);
        Assert.Equal(4, session.RemainingContextTokens);

        await session.AppendAsync(new int[] { 10, 20 });
        Assert.False(session.IsContextLimitReached);
        Assert.Equal(2, session.RemainingContextTokens);

        await session.AppendAsync(new int[] { 30, 40 });
        Assert.True(session.IsContextLimitReached);
        Assert.Equal(0, session.RemainingContextTokens);
    }

    [Fact]
    public async Task Test6_GenerationStopsAtLimit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockSafetyForwardPass();
        var options = new KvSequenceOptions { MaxContextTokens = 8 };
        await using var session = new InferenceSession(cache, options: options, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 }); // 5 prompt tokens

        var generated = new List<int>();
        var sampling = new SamplingParams { MaxNewTokens = 100 }; // Ask for 100 new tokens

        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            if (int.TryParse(chunk.Text, out int tid))
            {
                generated.Add(tid);
            }
        }

        // Generation must stop cleanly at exactly 8 total tokens (3 new tokens)
        Assert.Equal(8, session.TokenCount);
        Assert.True(session.IsContextLimitReached);
        Assert.Equal(3, generated.Count);
    }

    [Fact]
    public async Task Test7_SpeculativeGenerationCannotOverflow()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockSafetyForwardPass();
        var options = new KvSequenceOptions { MaxContextTokens = 6 };
        await using var session = new InferenceSession(cache, options: options, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 }); // 5 tokens (1 remaining capacity)

        Assert.Equal(1, session.RemainingContextTokens);

        var generated = new List<int>();
        var sampling = new SamplingParams { MaxNewTokens = 10 };

        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            if (int.TryParse(chunk.Text, out int tid))
            {
                generated.Add(tid);
            }
        }

        Assert.Equal(6, session.TokenCount);
        Assert.Single(generated);
    }

    [Fact]
    public async Task Test8_ToolResultCannotOverflow()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 10 };
        await using var session = new InferenceSession(cache, options: options);

        session.Tokenizer = new MockSafetyTokenizer();

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5, 6, 7 }); // 7 tokens (3 remaining)

        using var doc = System.Text.Json.JsonDocument.Parse("\"Result string requiring 10 tokens\"");
        var toolResult = new OpenTail.Stingray.Core.Tools.ToolResult("call_1", doc.RootElement, IsError: false);

        await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendToolResultAsync(toolResult);
        });

        // Sequence remains untouched at 7 tokens
        Assert.Equal(7, session.TokenCount);
    }

    [Fact]
    public async Task Test9_ForkPreservesLimit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 16 };
        await using var parent = new InferenceSession(cache, options: options);

        await parent.AppendAsync(new int[] { 1, 2, 3, 4 });
        await using var child = parent.Fork();

        Assert.Equal(16, parent.MaxContextTokens);
        Assert.Equal(16, child.MaxContextTokens);
    }

    [Fact]
    public async Task Test10_ForkCountsIndependently()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 10 };
        await using var parent = new InferenceSession(cache, options: options);

        await parent.AppendAsync(new int[] { 1, 2, 3, 4 }); // 4 tokens
        await using var child = parent.Fork();

        await child.AppendAsync(new int[] { 5, 6, 7, 8, 9, 10 }); // Child hits limit (10 tokens)
        Assert.True(child.IsContextLimitReached);

        // Parent context count remains at 4
        Assert.Equal(4, parent.TokenCount);
        Assert.False(parent.IsContextLimitReached);
    }

    [Fact]
    public async Task Test11_PrefixCacheCountsTowardLimit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache);

        var prefixTokens = Enumerable.Range(1, 40).ToArray();
        var options = new KvSequenceOptions { MaxContextTokens = 50 };

        await using var session = await runtime.CreateSessionAsync(options: options);
        await session.AppendAsync(prefixTokens); // 40 tokens prefilled

        Assert.Equal(50, session.MaxContextTokens);
        Assert.Equal(40, session.TokenCount);
        Assert.Equal(10, session.RemainingContextTokens);
    }

    [Fact]
    public async Task Test12_PartialPageLimit()
    {
        // 32-token page size, but 45-token max context limit (unaligned with page boundary)
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 45 };
        await using var session = new InferenceSession(cache, options: options);

        await session.AppendAsync(new int[40]); // 40 tokens appended

        Assert.Equal(5, session.RemainingContextTokens);

        await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendAsync(new int[10]); // 50 > 45 limit
        });

        Assert.Equal(40, session.TokenCount);
    }

    [Fact]
    public async Task Test13_SuspendResumePreservesLimit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 20 };
        await using var session = new InferenceSession(cache, options: options);

        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
        await session.SuspendAsync();

        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal(20, session.MaxContextTokens);

        await session.ResumeAsync();
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal(20, session.MaxContextTokens);
        Assert.Equal(5, session.TokenCount);
    }

    [Fact]
    public async Task Test14_GovernorIndependence()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 10 };
        await using var session = new InferenceSession(cache, options: options);

        await session.AppendAsync(new int[8]);

        // Context limit exception is an InvalidOperationException subclass, distinct from cache capacity failures
        var ex = await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendAsync(new int[5]); // Exceeds 10 limit
        });

        Assert.NotNull(ex);
        Assert.Equal(10, ex.MaxContextTokens);
    }

    [Fact]
    public async Task Test15_ExistingNoLimitRegression()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache); // Default options

        Assert.Null(session.MaxContextTokens);
        Assert.False(session.IsContextLimitReached);

        await session.AppendAsync(new int[100]);
        Assert.Equal(100, session.TokenCount);
        Assert.False(session.IsContextLimitReached);
    }

    [Fact]
    public async Task Test16_KvAppendFailureRollbackIsAtomic()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockFailingForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        // Initial successful append
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });
        Assert.Equal(5, session.TokenCount);
        Assert.Equal(SessionState.Ready, session.State);

        // Trigger prefill failure on second append
        fwd.ShouldFailPrefill = true;

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await session.AppendAsync(new int[] { 10, 20, 30 });
        });

        // 4-Way Transactional Rollback Invariant: Token history, KV sequence, forward pass position, and session state are restored to exact pre-append state!
        Assert.Equal(5, session.TokenCount);
        Assert.Equal(new int[] { 1, 2, 3, 4, 5 }, session.TokenHistory);
        Assert.Equal(5, session.KvSequence.TokenCount);
        Assert.Equal(5, fwd.Position);
        Assert.Equal(SessionState.Ready, session.State);
    }

    private sealed class MockFailingForwardPass : IForwardPass
    {
        public bool ShouldFailPrefill { get; set; }
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockFailingForwardPass { Position = Position, ShouldFailPrefill = ShouldFailPrefill };
        public System.ReadOnlySpan<float> Forward(int position, int token)
        {
            Position = position + 1;
            return new float[100];
        }
        public System.ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            if (ShouldFailPrefill)
            {
                throw new InvalidOperationException("Simulated prefill failure");
            }
            Position = startPos + tokens.Count;
            return new float[100];
        }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class MockSafetyForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockSafetyForwardPass { Position = Position };
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

    private sealed class MockSafetyTokenizer : ITokenizer
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
        public IReadOnlyList<int> Encode(string text) => new int[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 }; // 10 tokens
    }
}
