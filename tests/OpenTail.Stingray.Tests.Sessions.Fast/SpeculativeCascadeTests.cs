
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

    /// <summary>
    /// Correctness test for docs/1-correctness/bugstofix.md:
    /// "SpeculativeDecoder.cs StepSampled/PLD bugs".
    /// When PLD proposes fewer tokens than lookahead - 1 (e.g. 4 proposals for lookahead=7) and all proposals
    /// are accepted, StepSampled must not access beyond the batch size and must decode successfully.
    /// PLD may return fewer proposals than the configured lookahead.
    /// </summary>
    [Fact]
    public void Test6_CascadeSampled_PldHit_ProposalsShorterThanLookahead_DecodesSuccessfully()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        var sampling = new SamplingParams { Temperature = 0.5f };
        var rng = new Random(42);

        var decoder = new SpeculativeDecoder(target, pld, draft, sampling, rng, lookahead: 7);

        var prompt = new int[] { 10, 20, 30, 40, 10 };
        var initLogits = target.Prefill(prompt);
        decoder.Initialize(prompt, initLogits);

        var emitted = new List<int>();
        var ex = Record.Exception(() =>
            decoder.Decode(maxTokens: 10, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token)));
        Assert.Null(ex);
        Assert.NotEmpty(emitted);
    }

    /// <summary>
    /// Correctness test for docs/1-correctness/bugstofix.md:
    /// "SpeculativeDecoder.cs StepSampled/PLD bugs".
    /// When PLD is used in sampled mode, StepSampled must synchronize the draft forward pass cache
    /// with the accepted PLD proposals, and increment PromptLookupAcceptedTokens in metrics.
    /// The sampled PLD path must account for accepted tokens and keep the draft cache aligned.
    /// </summary>
    [Fact]
    public void Test7_CascadeSampled_PldHit_TracksAcceptedTokens()
    {
        var target = new TrackingForwardPass();
        var draft = new TrackingForwardPass();
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        var sampling = new SamplingParams { Temperature = 0.5f };
        var rng = new Random(42);

        var decoder = new SpeculativeDecoder(target, pld, draft, sampling, rng, lookahead: 4);

        // Repeating history lets the PLD proposals follow the target's deterministic cycle.
        var prompt = new int[] { 10, 20, 30, 40, 10, 20, 30, 40 };
        var initLogits = target.Prefill(prompt);
        decoder.Initialize(prompt, initLogits);

        var emitted = new List<int>();
        decoder.Decode(maxTokens: 10, stopTokenIds: Array.Empty<int>(), token => emitted.Add(token));

        var metrics = decoder.Metrics;
        Assert.True(metrics.PromptLookupHits > 0);
        Assert.True(metrics.PromptLookupAcceptedTokens > 0);
        Assert.Equal(target.Position, draft.Position);
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
