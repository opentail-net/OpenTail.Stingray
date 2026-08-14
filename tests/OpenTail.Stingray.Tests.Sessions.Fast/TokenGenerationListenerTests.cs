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

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class TokenGenerationListenerTests
{
    [Fact]
    public async Task Test1_ListenerReceivesCommittedTokens()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockListenerTokenizer();

        var receivedTokens = new List<int>();
        var receivedTexts = new List<string>();

        session.OnTokenGenerated += (token, text) =>
        {
            receivedTokens.Add(token);
            receivedTexts.Add(text);
        };

        await session.AppendAsync(new int[] { 1 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { Temperature = 0.0f, MaxNewTokens = 3 })) { }

        Assert.Equal(3, receivedTokens.Count);
        Assert.Equal(new int[] { 10, 10, 10 }, receivedTokens);
        Assert.Equal(new string[] { "T10", "T10", "T10" }, receivedTexts);
    }

    [Fact]
    public async Task Test2_NoListenerPreservesNormalGeneration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockListenerTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { Temperature = 0.0f, MaxNewTokens = 2 }))
        {
            chunks.Add(chunk.Text);
        }

        Assert.Equal(2, chunks.Count);
    }

    [Fact]
    public async Task Test3_ListenerReceivesDecodedText()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockListenerTokenizer();

        string? lastText = null;
        session.OnTokenGenerated += (token, text) => lastText = text;

        await session.AppendAsync(new int[] { 1 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { Temperature = 0.0f, MaxNewTokens = 1 })) { }

        Assert.Equal("T10", lastText);
    }

    [Fact]
    public async Task Test4_ListenerFiresOnlyAfterCommit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockListenerTokenizer();

        long historyCountAtEvent = 0;
        session.OnTokenGenerated += (token, text) =>
        {
            // At the moment of event invocation, session history must already contain the committed token!
            historyCountAtEvent = session.TokenCount;
        };

        await session.AppendAsync(new int[] { 1, 2 }); // Prompt = 2
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 1 })) { }

        Assert.Equal(3, historyCountAtEvent); // 2 prompt + 1 generated committed token
    }

    [Fact]
    public async Task Test5_SpeculativeRejectedTokensNotObserved()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockListenerTokenizer();

        var events = new List<int>();
        session.OnTokenGenerated += (token, text) => events.Add(token);

        await session.AppendAsync(new int[] { 1 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 2 })) { }

        // Only the 2 committed tokens were received (no speculative drafts)
        Assert.Equal(2, events.Count);
    }

    [Fact]
    public async Task Test6_SpeculativeOrdering()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        var events = new List<int>();
        session.OnTokenGenerated += (token, text) => events.Add(token);

        await session.AppendAsync(new int[] { 1 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 3 })) { }

        Assert.Equal(3, events.Count);
    }

    [Fact]
    public async Task Test7_MultipleListeners()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        int count1 = 0;
        int count2 = 0;

        session.OnTokenGenerated += (t, s) => count1++;
        session.OnTokenGenerated += (t, s) => count2++;

        await session.AppendAsync(new int[] { 1 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 2 })) { }

        Assert.Equal(2, count1);
        Assert.Equal(2, count2);
    }

    [Fact]
    public async Task Test8_ListenerExceptionDoesNotBreakInference()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        // Host subscriber throws an exception
        session.OnTokenGenerated += (t, s) => throw new InvalidOperationException("Host TUI render bug!");

        await session.AppendAsync(new int[] { 1 });

        var chunks = new List<string>();
        // Inference completes cleanly despite subscriber exception!
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 2 }))
        {
            chunks.Add(chunk.Text);
        }

        Assert.Equal(2, chunks.Count);
        Assert.Equal(SessionState.Ready, session.State);
    }

    [Fact]
    public async Task Test9_ForkListenerIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd);

        int parentEvents = 0;
        parent.OnTokenGenerated += (t, s) => parentEvents++;

        await parent.AppendAsync(new int[] { 1 });
        await using var child = parent.Fork();

        // Generate on child branch
        await foreach (var chunk in child.GenerateAsync(new SamplingParams { MaxNewTokens = 3 })) { }

        // Child generation does NOT invoke parent's event listener!
        Assert.Equal(0, parentEvents);
    }

    [Fact]
    public async Task Test10_SuspensionResumeListener()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        int events = 0;
        session.OnTokenGenerated += (t, s) => events++;

        await session.AppendAsync(new int[] { 1 });
        await session.SuspendAsync();
        await session.ResumeAsync();

        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 2 })) { }

        Assert.Equal(2, events);
    }

    [Fact]
    public async Task Test11_ToolResultNotObserved()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);
        session.Tokenizer = new MockListenerTokenizer();

        int events = 0;
        session.OnTokenGenerated += (t, s) => events++;

        await session.AppendAsync(new int[] { 1 });

        using var doc = JsonDocument.Parse("{\"status\":\"ok\"}");
        var toolResult = new OpenTail.Stingray.Core.Tools.ToolResult("call_1", doc.RootElement);

        await session.AppendToolResultAsync(toolResult);

        // Tool result append is prompt prefill, so OnTokenGenerated (for model output) does NOT fire!
        Assert.Equal(0, events);
    }

    [Fact]
    public async Task Test12_ConstrainedChoiceIntegration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockListenerTokenizer();

        var events = new List<string>();
        session.OnTokenGenerated += (t, s) => events.Add(s);

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "T10" }
        };

        await foreach (var chunk in session.GenerateAsync(sampling)) { }

        Assert.Single(events);
        Assert.Equal("T10", events[0]);
    }

    [Fact]
    public async Task Test13_Cancellation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        var events = new List<int>();
        session.OnTokenGenerated += (t, s) => events.Add(t);

        await session.AppendAsync(new int[] { 1 });

        using var cts = new CancellationTokenSource();
        int count = 0;
        try
        {
            await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 10 }, cts.Token))
            {
                count++;
                if (count == 2)
                {
                    cts.Cancel(); // Cancel mid-generation
                }
            }
        }
        catch (OperationCanceledException)
        {
        }

        // Exactly 2 committed tokens were observed before cancellation
        Assert.Equal(2, events.Count);
    }

    [Fact]
    public async Task Test14_PLDSafety()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockListenerForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);

        var events = new List<int>();
        session.OnTokenGenerated += (t, s) => events.Add(t);

        await session.AppendAsync(new int[] { 1, 2, 3 });
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 2 })) { }

        Assert.Equal(2, events.Count);
    }

    private sealed class MockListenerForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockListenerForwardPass { Position = Position };
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

    private sealed class MockListenerTokenizer : ITokenizer
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
        public string Decode(IEnumerable<int> tokens) => string.Join("", tokens.Select(t => $"T{t}"));
        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "T10" => new int[] { 10 },
            _ => new int[] { 99 }
        };
    }
}
