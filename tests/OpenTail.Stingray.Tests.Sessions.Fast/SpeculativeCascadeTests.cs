
namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class SpeculativeCascadeTests
{
    [Fact]
    public void Test1_CascadePldHit_DoesNotInvokeDraftModel()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);

        var decoder = new SpeculativeDecoder(target, pld, draft, lookahead: 4);

        // Seed repeating history so PLD matches: prompt = [10, 20, 30, 40, 10, 20, 30, 40]
        var prompt = new int[] { 10, 20, 30, 40, 10, 20, 30, 40 };
        var initLogits = target.Prefill(prompt);
        decoder.Initialize(prompt, initLogits);

        var emitted = new List<int>();
        decoder.Decode(maxTokens: 2, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token));

        var metrics = decoder.Metrics;

        Assert.True(metrics.PromptLookupHits > 0);
        Assert.Equal(0, draft.ForwardCallCount);
    }

    [Fact]
    public void Test2_CascadePldMiss_FallsBackToDraftModel()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);

        var decoder = new SpeculativeDecoder(target, pld, draft, lookahead: 4);

        // Seed unique tokens so PLD misses: prompt = [1, 2, 3, 4, 5]
        var prompt = new int[] { 1, 2, 3, 4, 5 };
        decoder.Initialize(prompt, new float[100]);

        var emitted = new List<int>();
        decoder.Decode(maxTokens: 3, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token));

        var metrics = decoder.Metrics;

        Assert.True(metrics.DraftFallbackAttempts > 0);
        Assert.True(draft.ForwardCallCount > 0);
    }

    [Fact]
    public void Test3_CascadeMetricsTracking()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);

        var decoder = new SpeculativeDecoder(target, pld, draft, lookahead: 4);
        var prompt = new int[] { 10, 20, 30, 10, 20 };
        decoder.Initialize(prompt, new float[100]);

        var emitted = new List<int>();
        decoder.Decode(maxTokens: 4, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token));

        var metrics = decoder.Metrics;

        Assert.True(metrics.PromptLookupAttempts > 0);
        Assert.True(metrics.TotalEmitted > 0);
    }

    [Fact]
    public void Test4_CascadeRejectionAndRollback()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);

        var decoder = new SpeculativeDecoder(target, pld, draft, lookahead: 4);
        decoder.Initialize(new int[] { 1, 2, 3, 4 }, new float[100]);

        var emitted = new List<int>();
        decoder.Decode(maxTokens: 2, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token));

        Assert.True(target.TruncateCallCount > 0);
    }

    [Fact]
    public void Test5_DisabledCascadeSingleSourceCompatibility()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();

        // Single-source model draft mode
        var decoder = new SpeculativeDecoder(target, draft, lookahead: 4);
        decoder.Initialize(4, new float[100]);

        var emitted = new List<int>();
        decoder.Decode(maxTokens: 2, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token));

        var metrics = decoder.Metrics;
        Assert.Equal(0, metrics.PromptLookupAttempts);
        Assert.True(draft.ForwardCallCount > 0);
    }

    private sealed class TrackingForwardPass : IForwardPass
    {
        public bool SupportsPartialRewind => true;
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;
        public int ForwardCallCount { get; private set; }
        public int TruncateCallCount { get; private set; }

        public IForwardPass CreateContext() => new TrackingForwardPass();
        public ReadOnlySpan<float> Forward(int token, int position)
        {
            ForwardCallCount++;
            Position = position + 1;
            var logits = new float[100];
            int nextToken = token switch { 10 => 20, 20 => 30, 30 => 40, 40 => 10, _ => 10 };
            logits[nextToken] = 10f;
            return logits;
        }
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            var logits = new float[100];
            int lastToken = tokens.Count > 0 ? tokens[^1] : 40;
            int nextToken = lastToken switch { 10 => 20, 20 => 30, 30 => 40, 40 => 10, _ => 10 };
            logits[nextToken] = 10f;
            return logits;
        }
        public void TruncateTo(int position)
        {
            TruncateCallCount++;
            Position = position;
        }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
