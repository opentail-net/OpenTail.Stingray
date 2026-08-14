using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Tools;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Unit tests for <see cref="GenerationResult"/> and <see cref="GenerationStream"/>.
/// </summary>
public sealed class GenerationResultTests
{
    [Fact]
    public async Task Test01_BasicResult_IsPopulated()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 5 });
        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in stream)
        {
            chunks.Add(chunk);
        }

        var result = stream.Result;
        Assert.NotNull(result);
        Assert.Equal(FinishReason.MaxTokens, result.FinishReason);
        Assert.Equal(5, result.GeneratedTokenCount);
        Assert.NotNull(result.Metrics);
        Assert.NotNull(result.ContinuationToken);
        Assert.Equal(session.Id, result.ContinuationToken.Value.SessionId);
        Assert.Equal(session.TokenCount, result.ContinuationToken.Value.TokenPosition);
    }

    [Fact]
    public async Task Test02_StreamingStillWorks()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 3 });
        var textChunks = new List<string>();
        await foreach (var chunk in stream)
        {
            if (!string.IsNullOrEmpty(chunk.Text))
            {
                textChunks.Add(chunk.Text);
            }
        }

        Assert.Equal(3, textChunks.Count);
        Assert.Equal(3, stream.Result.GeneratedTokenCount);
    }

    [Fact]
    public async Task Test03_NoDuplicateGeneration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        int forwardCallsBefore = fwd.ForwardCalls;

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 4 });
        await foreach (var _ in stream) { }

        int forwardCallsAfterFirstRead = fwd.ForwardCalls;
        Assert.Equal(forwardCallsBefore + 4, forwardCallsAfterFirstRead);

        // Accessing .Result should not trigger any extra forward pass calls
        var result1 = stream.Result;
        var result2 = stream.Result;

        Assert.Equal(forwardCallsAfterFirstRead, fwd.ForwardCalls);
        Assert.Same(result1, result2);
    }

    [Fact]
    public async Task Test04_TokenCount_MatchesCommittedOutput()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 10, 20 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 7 });
        await foreach (var _ in stream) { }

        Assert.Equal(7, stream.Result.GeneratedTokenCount);
    }

    [Fact]
    public async Task Test05_ToolCallResult_CapturedInResult()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        // Configure a ToolCallParser on session that triggers a tool call on token count > 3
        var expectedCall = new ToolCall("call_1", "get_weather", default);
        session.ToolCallParser = history => history.Count > 3 ? new[] { expectedCall } : null;
        session.ToolProvider = new TestToolProvider(new ToolDefinition("get_weather"));

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 10 });
        await foreach (var _ in stream) { }

        var result = stream.Result;
        Assert.Equal(FinishReason.ToolCall, result.FinishReason);
        Assert.Single(result.ToolCalls);
        Assert.Equal("get_weather", result.ToolCalls[0].Name);
    }

    [Fact]
    public async Task Test06_ContinuationToken_PointsToCommittedPosition()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3, 4 }); // 4 prompt tokens

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 3 });
        await foreach (var _ in stream) { }

        var result = stream.Result;
        Assert.NotNull(result.ContinuationToken);
        Assert.Equal(7, result.ContinuationToken.Value.TokenPosition);
        Assert.Equal(7, session.TokenCount);

        // Validation against session should succeed
        session.ValidateContinuationToken(result.ContinuationToken.Value);
    }

    [Fact]
    public async Task Test07_MetricsSnapshot_IsImmutable()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 3 });
        await foreach (var _ in stream) { }

        var result = stream.Result;
        long snapshotGenTokens = result.Metrics.GeneratedTokens;
        Assert.Equal(3, snapshotGenTokens);

        // Run more generation on session
        await session.AppendAsync(new int[] { 99 });
        var stream2 = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 2 });
        await foreach (var _ in stream2) { }

        // Initial result's metrics snapshot must remain unchanged
        Assert.Equal(snapshotGenTokens, result.Metrics.GeneratedTokens);
        Assert.Equal(5, stream2.Result.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task Test08_Cancellation_FinishReasonCancelled()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        using var cts = new CancellationTokenSource();
        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 50 }, cts.Token);

        int count = 0;
        try
        {
            await foreach (var _ in stream)
            {
                count++;
                if (count == 2)
                {
                    cts.Cancel();
                }
            }
        }
        catch (OperationCanceledException) { /* expected */ }

        var result = stream.Result;
        Assert.Equal(FinishReason.Cancelled, result.FinishReason);
        Assert.Equal(2, result.GeneratedTokenCount);
    }

    [Fact]
    public async Task Test09_MaxTokens_FinishReasonMaxTokens()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 4 });
        await foreach (var _ in stream) { }

        Assert.Equal(FinishReason.MaxTokens, stream.Result.FinishReason);
        Assert.Equal(4, stream.Result.GeneratedTokenCount);
    }

    [Fact]
    public async Task Test10_EmptyToolCalls_NotNull()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 2 });
        await foreach (var _ in stream) { }

        var result = stream.Result;
        Assert.NotNull(result.ToolCalls);
        Assert.Empty(result.ToolCalls);
    }

    [Fact]
    public async Task Test11_ResultImmutability()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 2 });
        await foreach (var _ in stream) { }

        var result = stream.Result;
        Assert.IsAssignableFrom<IReadOnlyList<ToolCall>>(result.ToolCalls);
        Assert.False(result.ToolCalls is List<ToolCall>, "ToolCalls should be read-only collection wrapper, not a raw List.");
    }

    [Fact]
    public async Task Test12_SessionConsistency()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        int promptCount = 5;
        await session.AppendAsync(new int[] { 1, 2, 3, 4, 5 });

        var stream = session.GenerateWithResultAsync(new SamplingParams { MaxNewTokens = 6 });
        await foreach (var _ in stream) { }

        var result = stream.Result;
        Assert.Equal(promptCount + result.GeneratedTokenCount, (int)session.TokenCount);
        Assert.Equal(session.TokenCount, result.ContinuationToken!.Value.TokenPosition);
    }

    [Fact]
    public async Task Test13_LinkedCtsDisposedOnEnumerationEnd()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new CountingMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2 });

        using var streamCts = new CancellationTokenSource();
        using var enumCts = new CancellationTokenSource();

        var stream = new GenerationStream(session, new SamplingParams { MaxNewTokens = 3 }, streamCts.Token);

        await using (var enumerator = stream.GetAsyncEnumerator(enumCts.Token))
        {
            Assert.True(await enumerator.MoveNextAsync());
        }

        // Enumerator disposal disposes the internal linked CTS, which unregisters cancellation callbacks.
        Assert.NotNull(stream.Result);
        Assert.Equal(FinishReason.Completed, stream.Result.FinishReason);
    }

    private sealed class CountingMockForwardPass : IForwardPass
    {
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;
        public int Position { get; private set; }
        public int ForwardCalls { get; private set; }

        public IForwardPass CreateContext() => new CountingMockForwardPass { Position = Position, ForwardCalls = ForwardCalls };

        public ReadOnlySpan<float> Forward(int position, int token)
        {
            ForwardCalls++;
            Position = position + 1;
            var res = new float[100];
            res[10] = 5.0f;
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

    private sealed class TestToolProvider : IToolProvider
    {
        private readonly IReadOnlyList<ToolDefinition> _tools;

        public TestToolProvider(params ToolDefinition[] tools)
        {
            _tools = tools;
        }

        public IReadOnlyList<ToolDefinition> GetTools(InferenceToolContext? context = null) => _tools;
    }
}
