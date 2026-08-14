using System;
using System.Collections.Generic;
using System.Text.Json;
using OpenTail.Stingray.Core.Tools;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class ToolCapabilityTests
{
    [Fact]
    public void ToolProvider_ReturnsContextAppropriateTools()
    {
        var readTool = new ToolDefinition("read_file", "Reads file content", JsonDocument.Parse("{}").RootElement);
        var writeTool = new ToolDefinition("write_file", "Writes file content", JsonDocument.Parse("{}").RootElement, Annotations: new ToolAnnotations(DestructiveHint: true));

        var provider = new MemoryToolProvider(new[] { readTool, writeTool });
        var context = new InferenceToolContext(Mode: "CodeReview");

        var tools = provider.GetTools(context);

        Assert.Single(tools);
        Assert.Equal("read_file", tools[0].Name);
    }

    [Fact]
    public void ToolProvider_ExcludesWriteToolsFromReview()
    {
        var readTool = new ToolDefinition("search_code", "Searches code");
        var pushTool = new ToolDefinition("git_push", "Pushes commits", Annotations: new ToolAnnotations(DestructiveHint: true));

        var provider = new MemoryToolProvider(new[] { readTool, pushTool });
        var tools = provider.GetTools(new InferenceToolContext(Mode: "CodeReview"));

        Assert.DoesNotContain(tools, t => t.Name == "git_push");
        Assert.Contains(tools, t => t.Name == "search_code");
    }

    [Fact]
    public void DifferentSessions_CanHaveDifferentToolSets()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);

        var readTool = new ToolDefinition("read_file");
        var writeTool = new ToolDefinition("write_file", Annotations: new ToolAnnotations(DestructiveHint: true));
        var provider = new MemoryToolProvider(new[] { readTool, writeTool });

        var sessionA = new InferenceSession(cache)
        {
            ToolProvider = provider,
            ToolContext = new InferenceToolContext(Mode: "CodeReview")
        };

        var sessionB = new InferenceSession(cache)
        {
            ToolProvider = provider,
            ToolContext = new InferenceToolContext(Mode: "Implementation")
        };

        var writeCall = new ToolCall("call-1", "write_file", JsonDocument.Parse("{}").RootElement);

        Assert.False(sessionA.ValidateToolCall(writeCall));
        Assert.True(sessionB.ValidateToolCall(writeCall));
    }

    [Fact]
    public void ToolCall_UsesCanonicalShape()
    {
        var args = JsonDocument.Parse("{\"path\": \"main.cs\"}").RootElement;
        var call = new ToolCall("call-123", "read_file", args);

        Assert.Equal("call-123", call.Id);
        Assert.Equal("read_file", call.Name);
        Assert.True(call.Arguments.TryGetProperty("path", out var pathElement));
        Assert.Equal("main.cs", pathElement.GetString());
    }

    [Fact]
    public void ToolResult_CorrelatesByToolCallId()
    {
        var content = JsonDocument.Parse("{\"output\": \"OK\"}").RootElement;
        var result = new ToolResult("call-123", content);

        Assert.Equal("call-123", result.ToolCallId);
        Assert.False(result.IsError);
    }

    [Fact]
    public void DisallowedTool_CannotBeExecuted()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        var readTool = new ToolDefinition("read_file");
        var provider = new MemoryToolProvider(new[] { readTool });

        var session = new InferenceSession(cache)
        {
            ToolProvider = provider
        };

        var disallowedCall = new ToolCall("call-99", "exec_shell", JsonDocument.Parse("{}").RootElement);

        Assert.False(session.ValidateToolCall(disallowedCall));
    }

    [Fact]
    public void EmptyToolSet_BehavesExactlyAsBefore()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        var session = new InferenceSession(cache); // ToolProvider is null

        var anyCall = new ToolCall("call-1", "any_tool", JsonDocument.Parse("{}").RootElement);

        // Fail-closed: missing ToolProvider rejects tool call authorization
        Assert.False(session.ValidateToolCall(anyCall));
    }

    [Fact]
    public async System.Threading.Tasks.Task ToolCall_Continuation_PreservesSessionAndKvState()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);

        // Append initial prompt
        await session.AppendAsync(new int[] { 10, 20, 30 });
        long initialTokens = session.TokenCount;
        var initialSeqId = session.KvSequence.SequenceId;

        session.Tokenizer = new MockTokenizer();

        // Append tool result continuation
        var result = new ToolResult("call-123", JsonDocument.Parse("{\"status\": \"success\"}").RootElement);
        await session.AppendToolResultAsync(result);

        // Invariant check: session ID, KV sequence ID remain identical; tokens & KV capacity increased seamlessly
        Assert.Equal(initialSeqId, session.KvSequence.SequenceId);
        Assert.True(session.TokenCount > initialTokens);
        Assert.Equal(SessionState.Ready, session.State);
    }

    [Fact]
    public async System.Threading.Tasks.Task GenerateAsync_NoToolProvider_StillGeneratesNormally()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        using var fwd = new TestForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd); // ToolProvider is null

        await session.AppendAsync(new int[] { 1, 2, 3 });

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 5 }))
        {
            chunks.Add(chunk);
        }

        Assert.NotEmpty(chunks);
    }

    [Fact]
    public async System.Threading.Tasks.Task AppendToolResultAsync_UsesConfiguredTokenizer()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache);
        var mockTokenizer = new MockTokenizer();
        session.Tokenizer = mockTokenizer;

        var content = JsonDocument.Parse("{\"message\":\"hello\",\"emoji\":\"😀\"}").RootElement;
        var result = new ToolResult("call-999", content);

        await session.AppendToolResultAsync(result);

        Assert.True(mockTokenizer.EncodeCalled);
        Assert.Contains(9999, session.TokenHistory); // MockTokenizer encodes text into [9999]
    }

    [Fact]
    public async System.Threading.Tasks.Task GenerateAsync_EmitsToolCall_WhenToolCallsDetectedAndAuthorized()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        using var fwd = new TestForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        var readTool = new ToolDefinition("read_file");
        session.ToolProvider = new MemoryToolProvider(new[] { readTool });

        // Wire ToolCallParser delegate
        session.ToolCallParser = history => new[]
        {
            new ToolCall("call-abc", "read_file", JsonDocument.Parse("{\"path\":\"test.txt\"}").RootElement)
        };

        await session.AppendAsync(new int[] { 1, 2 });

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 5 }))
        {
            chunks.Add(chunk);
        }

        var toolChunk = Assert.Single(chunks, c => c.Kind == GenerateChunkKind.ToolCall);
        Assert.NotNull(toolChunk.ToolCalls);
        Assert.Single(toolChunk.ToolCalls);
        Assert.Equal("read_file", toolChunk.ToolCalls[0].Name);
        Assert.True(toolChunk.ToolCalls[0].IsAuthorized);
    }

    [Fact]
    public async System.Threading.Tasks.Task AppendToolResultAsync_WithoutTokenizer_ThrowsInvalidOperationException()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        await using var session = new InferenceSession(cache); // Tokenizer is null

        var content = JsonDocument.Parse("{\"output\":\"OK\"}").RootElement;
        var result = new ToolResult("call-1", content);

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await session.AppendToolResultAsync(result);
        });
    }

    [Fact]
    public async System.Threading.Tasks.Task UnauthorisedToolCall_IsNotSilentlyDiscarded()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        using var fwd = new TestForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        // Session only permits "read_file"
        var readTool = new ToolDefinition("read_file");
        session.ToolProvider = new MemoryToolProvider(new[] { readTool });

        // Model attempts to call unauthorized "delete_file"
        session.ToolCallParser = history => new[]
        {
            new ToolCall("call-xyz", "delete_file", JsonDocument.Parse("{}").RootElement)
        };

        await session.AppendAsync(new int[] { 1, 2 });

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 5 }))
        {
            chunks.Add(chunk);
        }

        // Must NOT silently discard: chunk kind is ToolCall, HasUnauthorizedToolCall is true, call has IsAuthorized = false
        var toolChunk = Assert.Single(chunks, c => c.Kind == GenerateChunkKind.ToolCall);
        Assert.True(toolChunk.HasUnauthorizedToolCall);
        Assert.NotNull(toolChunk.ToolCalls);
        Assert.Single(toolChunk.ToolCalls);
        Assert.False(toolChunk.ToolCalls[0].IsAuthorized);
    }

    private sealed class MockTokenizer : OpenTail.Stingray.Core.ITokenizer
    {
        public bool EncodeCalled { get; private set; }
        public IReadOnlyList<int> Encode(string text)
        {
            EncodeCalled = true;
            return new[] { 9999 };
        }
        public string Decode(IEnumerable<int> tokens) => "decoded";
        public byte[] DecodeBytes(int token) => new byte[] { (byte)'a' };
        public int VocabSize => 10000;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
    }

    private sealed class TestForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        private readonly float[] _logits = new float[100];
        public int VocabSize => 100;
        public int MaxSeqLen => 512;
        public ReadOnlySpan<float> Forward(int token, int position) => _logits;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => _logits;
        public void ResetCache() { }
        public void TruncateTo(int newPosition) { }
        public void Dispose() { }
    }
}
