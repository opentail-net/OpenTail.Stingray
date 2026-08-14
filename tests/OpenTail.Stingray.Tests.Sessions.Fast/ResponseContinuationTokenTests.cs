using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class ResponseContinuationTokenTests
{
    [Fact]
    public async Task Test1_TokenRepresentsCommittedPosition()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 10, 20, 30 });
        var token = session.GetContinuationToken();

        Assert.Equal(session.Id, token.SessionId);
        Assert.Equal(3, token.TokenPosition);
        Assert.True(token.Generation > 0);
    }

    [Fact]
    public async Task Test2_InterruptedGenerationProducesValidToken()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockContinuationForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 1, 2, 3 });

        using var cts = new CancellationTokenSource();
        int generatedCount = 0;

        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 10 }, cts.Token))
            {
                generatedCount++;
                if (generatedCount == 2)
                {
                    cts.Cancel(); // Cancel mid-generation
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Expected interruption
        }

        var token = session.GetContinuationToken();
        Assert.Equal(session.TokenCount, token.TokenPosition);
        Assert.Equal(5, token.TokenPosition); // 3 prompt + 2 generated
    }

    [Fact]
    public async Task Test3_ContinueFromToken()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2, 3 });
        var token = session.GetContinuationToken();

        // Valid continuation token passes validation
        await session.ContinueAsync(token);
        session.ValidateContinuationToken(token);
    }

    [Fact]
    public async Task Test4_StaleTokenRejected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2, 3 });
        var oldToken = session.GetContinuationToken();

        // Advance session with new append
        await session.AppendAsync(new int[] { 4, 5 });

        // Old token must be rejected as stale!
        var ex = Assert.Throws<StaleContinuationException>(() =>
        {
            session.ValidateContinuationToken(oldToken);
        });

        Assert.Equal(session.Id, ex.SessionId);
        Assert.Equal(3, ex.TokenPosition);
        Assert.Equal(5, ex.CurrentPosition);
    }

    [Fact]
    public async Task Test5_TokenCannotRewindSession()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 10, 20 });
        var oldToken = session.GetContinuationToken();

        await session.AppendAsync(new int[] { 30, 40 }); // Advance to 4 tokens

        // Attempting to continue with oldToken must NOT rewind session to 2 tokens
        await Assert.ThrowsAsync<StaleContinuationException>(async () =>
        {
            await session.ContinueAsync(oldToken);
        });

        Assert.Equal(4, session.TokenCount);
        Assert.Equal(new int[] { 10, 20, 30, 40 }, session.TokenHistory);
    }

    [Fact]
    public async Task Test6_WrongSessionRejected()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var sessionA = new InferenceSession(cache);
        await using var sessionB = new InferenceSession(cache);

        await sessionA.AppendAsync(new int[] { 1, 2 });
        var tokenA = sessionA.GetContinuationToken();

        var ex = Assert.Throws<ArgumentException>(() =>
        {
            sessionB.ValidateContinuationToken(tokenA);
        });

        Assert.Contains(sessionA.Id.ToString(), ex.Message);
    }

    [Fact]
    public async Task Test7_CancellationUsesLastCommittedPosition()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockContinuationForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        await session.AppendAsync(new int[] { 100, 200 });

        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Immediate cancellation before loop

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await session.AppendAsync(new int[] { 300, 400 }, cts.Token);
        });

        var token = session.GetContinuationToken();
        Assert.Equal(2, token.TokenPosition); // Zero uncommitted tokens leaked into continuation point
    }

    [Fact]
    public async Task Test8_SpeculativeTokensAreNotExposed()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2, 3, 4 });
        var token = session.GetContinuationToken();

        // Continuation token represents only committed tokens (4)
        Assert.Equal(4, token.TokenPosition);
        Assert.Equal(session.TokenCount, token.TokenPosition);
    }

    [Fact]
    public async Task Test9_ToolContinuation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        session.Tokenizer = new MockContinuationTokenizer();
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var token = session.GetContinuationToken();

        using var doc = JsonDocument.Parse("{\"status\":\"ok\"}");
        var toolResult = new OpenTail.Stingray.Core.Tools.ToolResult("call_123", doc.RootElement);

        // Continuation with valid token succeeds
        await session.AppendToolResultAsync(toolResult, token);
        Assert.True(session.TokenCount > 3);
    }

    [Fact]
    public async Task Test10_ToolResultOverflow()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var options = new KvSequenceOptions { MaxContextTokens = 5 };
        await using var session = new InferenceSession(cache, options: options);

        session.Tokenizer = new MockContinuationTokenizer();
        await session.AppendAsync(new int[] { 1, 2, 3, 4 }); // 4 tokens (1 remaining)

        var token = session.GetContinuationToken();
        using var doc = JsonDocument.Parse("{\"status\":\"ok\"}");
        var toolResult = new OpenTail.Stingray.Core.Tools.ToolResult("call_123", doc.RootElement);

        // Plan 010 ContextLimitExceededException applies during tool continuation
        await Assert.ThrowsAsync<ContextLimitExceededException>(async () =>
        {
            await session.AppendToolResultAsync(toolResult, token);
        });
    }

    [Fact]
    public async Task Test11_ForkIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        await parent.AppendAsync(new int[] { 1, 2, 3 });
        var parentToken = parent.GetContinuationToken();

        await using var child = parent.Fork();

        // Parent token is rejected on child due to SessionId mismatch
        Assert.Throws<ArgumentException>(() =>
        {
            child.ValidateContinuationToken(parentToken);
        });
    }

    [Fact]
    public async Task Test12_ForkContinuationIndependence()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        await parent.AppendAsync(new int[] { 1, 2, 3 });
        await using var child = parent.Fork();

        var parentToken = parent.GetContinuationToken();
        var childToken = child.GetContinuationToken();

        Assert.NotEqual(parentToken.SessionId, childToken.SessionId);
        parent.ValidateContinuationToken(parentToken);
        child.ValidateContinuationToken(childToken);
    }

    [Fact]
    public async Task Test13_SuspendResumeContinuation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 10, 20, 30 });
        var token = session.GetContinuationToken();

        await session.SuspendAsync();
        Assert.Equal(SessionState.Suspended, session.State);

        await session.ResumeAsync();
        Assert.Equal(SessionState.Ready, session.State);

        // Token remains valid across suspend/resume
        session.ValidateContinuationToken(token);
    }

    [Fact]
    public async Task Test14_ConcurrentContinuation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        await session.AppendAsync(new int[] { 1, 2 });
        var token = session.GetContinuationToken();

        // First continuation advances session
        await session.AppendAsync(new int[] { 3 });

        // Concurrent/second caller with old token receives StaleContinuationException
        Assert.Throws<StaleContinuationException>(() =>
        {
            session.ValidateContinuationToken(token);
        });
    }

    [Fact]
    public async Task Test15_MetadataUnaffectedAndTokenEncoding()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        session.Metadata.Set("workflow", "review");
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var token = session.GetContinuationToken();

        // Encode to Base64URL string and parse back
        string encoded = token.Encode();
        Assert.False(string.IsNullOrEmpty(encoded));

        var parsedToken = ResponseContinuationToken.Parse(encoded);
        Assert.Equal(token.SessionId, parsedToken.SessionId);
        Assert.Equal(token.TokenPosition, parsedToken.TokenPosition);
        Assert.Equal(token.Generation, parsedToken.Generation);

        // Session metadata remains unchanged
        Assert.Equal("review", session.Metadata.Get<string>("workflow"));
    }

    private sealed class MockContinuationForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockContinuationForwardPass { Position = Position };
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

    private sealed class MockContinuationTokenizer : ITokenizer
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
        public IReadOnlyList<int> Encode(string text) => new int[] { 100, 200 };
    }
}
