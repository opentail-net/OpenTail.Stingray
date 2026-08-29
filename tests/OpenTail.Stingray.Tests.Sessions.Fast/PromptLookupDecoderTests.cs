using OpenTail.Stingray.Core.Grammar;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class PromptLookupDecoderTests
{
    [Fact]
    public void Test1_ExactNGramMatch()
    {
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        // History: A B C D E F ... A B C
        pld.Reset(new int[] { 1, 2, 3, 4, 5, 6, 1, 2, 3 });

        var draft = pld.Propose(maxTokens: 3);
        Assert.Equal(new int[] { 4, 5, 6 }, draft);
    }

    [Fact]
    public void Test2_LongestAvailableMatch()
    {
        var pld = new PromptLookupDraft(ngramMax: 4, ngramMin: 2);
        // History contains:
        // [10, 20, 99, 50] (2-gram match for [10, 20])
        // [10, 20, 30, 40, 77] (4-gram match for [10, 20, 30, 40])
        pld.Reset(new int[] { 10, 20, 99, 50, 10, 20, 30, 40, 77, 10, 20, 30, 40 });

        var draft = pld.Propose(maxTokens: 3);
        // Should prefer 4-gram match [10, 20, 30, 40] -> 77, then continue proposing from
        // history past that match (77, 10, 20) up to maxTokens — a legitimate cyclic-repeat
        // prediction, not a bug: the drafter is designed to keep proposing from wherever the
        // matched occurrence's future actually leads, which here re-enters the start of the
        // (repeating) [10, 20, 30, 40] pattern.
        Assert.Equal(new int[] { 77, 10, 20 }, draft);
    }

    [Fact]
    public void Test3_NoMatch()
    {
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        pld.Reset(new int[] { 1, 2, 3, 4, 5, 6 });

        var draft = pld.Propose(maxTokens: 3);
        Assert.Empty(draft);
    }

    [Fact]
    public void Test4_ShortHistory()
    {
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        pld.Reset(new int[] { 1 });

        var draft = pld.Propose(maxTokens: 3);
        Assert.Empty(draft);
    }

    [Fact]
    public void Test5_EndOfHistory()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        // Match at end has only 2 tokens following
        pld.Reset(new int[] { 10, 20, 30, 40, 10, 20 });

        var draft = pld.Propose(maxTokens: 5);
        // The match's continuation runs out of NEW history at [30, 40], but the drafter keeps
        // going up to maxTokens by reading into the (identical) current tail it matched from —
        // a cyclic-repeat prediction, same as Test2.
        Assert.Equal(new int[] { 30, 40, 10, 20 }, draft);
    }

    [Fact]
    public void Test6_CurrentPositionBoundary()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        var fullHistory = new int[] { 1, 2, 3, 4, 1, 2, 5, 6, 7, 8 };

        // Propose up to position 4 only ([1, 2, 3, 4])
        var draft = pld.Propose(fullHistory, currentPosition: 4, maxDraftTokens: 3);

        // Position 4 tail is [3, 4], no prior match -> empty draft
        Assert.Empty(draft);
    }

    [Fact]
    public void Test7_ExactTokenMatching()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        // 100 != 200 even if they represent similar text
        pld.Reset(new int[] { 100, 2, 3, 200, 2, 4, 100, 2 });

        var draft = pld.Propose(maxTokens: 2);
        // Tail is [100, 2] -> matches index 0 -> continuation is [3, 200]
        Assert.Equal(new int[] { 3, 200 }, draft);
    }

    [Fact]
    public void Test8_MultipleMatches()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        // History has two [1, 2] matches: first followed by 10, second followed by 20
        pld.Reset(new int[] { 1, 2, 10, 99, 1, 2, 20, 88, 1, 2 });

        var draft = pld.Propose(maxTokens: 2);
        // Most recent occurrence wins -> continuation 20, 88
        Assert.Equal(new int[] { 20, 88 }, draft);
    }

    [Fact]
    public void Test9_HashCollisionSafety()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        var history = new int[] { 10, 20, 30, 40, 50, 60, 10, 20 };

        // Explicit slice verification ensures exact token equality check
        var draft = pld.Propose(history, currentPosition: 8, maxDraftTokens: 2);
        Assert.Equal(new int[] { 30, 40 }, draft);
    }

    [Fact]
    public void Test10_BranchIsolation()
    {
        var pldA = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        var pldB = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);

        pldA.Reset(new int[] { 1, 2, 10, 1, 2 });
        pldB.Reset(new int[] { 1, 2, 99, 1, 2 });

        var draftA = pldA.Propose(2);
        var draftB = pldB.Propose(2);

        // maxTokens=2 lets each proposal continue one token past the branch-distinguishing
        // token into the (identical, per-branch) tail it matched from — cyclic-repeat
        // prediction, same as Test2/Test5. The important assertion is the FIRST token, which
        // proves the two branches' histories never cross-contaminate each other.
        Assert.Equal(new int[] { 10, 1 }, draftA);
        Assert.Equal(new int[] { 99, 1 }, draftB);
    }

    [Fact]
    public void Test11_ConstraintCompatibility()
    {
        var tok = new MockLookupTokenizer();
        var vocab = new GrammarVocabulary(tok);
        var schema = JsonConstraint.AnyJson(vocab);

        // Invalid token 99 is filtered by constraint Filter
        Span<float> logits = new float[100];
        var masked = schema.Filter(logits);

        Assert.True(float.IsNegativeInfinity(masked[99]));
    }

    private sealed class MockLookupTokenizer : ITokenizer
    {
        public int VocabSize => 100;
        public int BosTokenId => 1;
        public int EosTokenId => 0;
        public int UnknownTokenId => -1;
        public int PadTokenId => -1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => System.Collections.Immutable.ImmutableArray.Create(0);
        public System.Collections.Generic.IReadOnlyDictionary<string, int> SpecialTokens => System.Collections.Immutable.ImmutableDictionary<string, int>.Empty;
        public byte[] DecodeBytes(int token) => token switch { 1 => new byte[] { (byte)'{' }, 2 => new byte[] { (byte)'}' }, _ => Array.Empty<byte>() };
        public string Decode(IEnumerable<int> tokens) => "";
        public IReadOnlyList<int> Encode(string text) => Array.Empty<int>();
    }

    [Fact]
    public void Test12_SpeculativeVerification()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        pld.Reset(new int[] { 10, 20, 30, 10, 20 });

        var draft = pld.Propose(maxTokens: 2);
        // maxTokens=2 lets the proposal continue one token past 30 into the tail it matched
        // from — cyclic-repeat prediction, same as Test2/Test5.
        Assert.Equal(new int[] { 30, 10 }, draft);
    }

    [Fact]
    public void Test13_RejectionRollback()
    {
        var pld = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        pld.Reset(new int[] { 10, 20, 99, 10, 20 });

        // Draft proposes 99 based on prior history, then one more token (cyclic-repeat
        // prediction into the matched tail, same as Test2/Test5) up to maxTokens=2.
        var draft = pld.Propose(maxTokens: 2);
        Assert.Equal(new int[] { 99, 10 }, draft);

        // If target rejects 99, append replaces history correctly
        pld.Append(42); // Actual target pick
        Assert.Equal(6, pld.Count);
    }

    [Fact]
    public void Test14_AcceptanceMetrics()
    {
        var metrics = new SpeculativeMetrics(
            TotalAccepted: 10,
            TotalEmitted: 15,
            AcceptanceRate: 0.667f,
            DraftMs: 1.0,
            VerifyMs: 5.0,
            CommitMs: 0.5,
            PromptLookupAttempts: 5,
            PromptLookupHits: 4,
            PromptLookupProposedTokens: 12,
            PromptLookupAcceptedTokens: 9);

        Assert.Equal(5, metrics.PromptLookupAttempts);
        Assert.Equal(4, metrics.PromptLookupHits);
        Assert.Equal(12, metrics.PromptLookupProposedTokens);
        Assert.Equal(9, metrics.PromptLookupAcceptedTokens);
        Assert.Equal(0.75f, metrics.PromptLookupAcceptanceRate, precision: 2);
    }

    [Fact]
    public void Test15_NoMatchNormalGeneration()
    {
        var pld = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        pld.Reset(new int[] { 1, 2, 3 });

        // No match -> empty proposal, degrades smoothly to plain decode
        var proposal = pld.Propose(maxTokens: 4);
        Assert.Empty(proposal);
    }
}
