
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Model-independent tests for <see cref="ForcedToolCallConstraint"/> and the per-family
/// <see cref="IToolCallAdapter.BuildForcedCallConstraint"/> that supplies it — the mechanism behind
/// OpenAI <c>tool_choice:"required"</c>.
///
/// <para>This is the inverse of every other constraint in <c>Grammar/</c>: those arm on the open
/// marker and shape what follows, so none of them can make a call happen. The two properties worth
/// pinning are therefore (a) before the marker, nothing else is samplable, and (b) after it, the
/// constraint gets out of the way completely — a forced constraint that stayed armed would mask the
/// entire call body down to one token and no arguments could ever be produced.</para>
/// </summary>
public sealed class ForcedToolCallConstraintTests
{
    /// <summary>Tokenizer whose special-token table is supplied per test, so the "marker present"
    /// and "marker absent" cases differ only in the vocabulary.</summary>
    private sealed class MarkerTokenizer(params string[] specials) : ITokenizer
    {
        // Ids start at 1: token 0 is reserved so the tests always have a non-marker token to prove
        // is masked, and ForcedToolCallConstraint rejects id 0 as a marker.
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } =
            specials.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i + 1, StringComparer.Ordinal);

        public int VocabSize => 32;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public byte[] DecodeBytes(int token) => [];
        public IReadOnlyList<int> Encode(string text) => [];
        public string Decode(IEnumerable<int> tokens) => "";
    }

    private static bool[] AllowedMask(ITokenConstraint c, int vocabSize)
    {
        Span<float> logits = new float[vocabSize];
        var masked = c.Filter(logits);
        var allowed = new bool[masked.Length];
        for (int i = 0; i < masked.Length; i++) allowed[i] = !float.IsNegativeInfinity(masked[i]);
        return allowed;
    }

    [Fact]
    public void BeforeTheMarker_OnlyTheMarkerIsSamplable()
    {
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<tool_call>"));
        var c = new ForcedToolCallConstraint(vocab, 1);

        Assert.True(c.IsConstraining);
        var allowed = AllowedMask(c, vocab.VocabSize);
        Assert.True(allowed[1]);
        Assert.Equal(1, allowed.Count(a => a));
    }

    [Fact]
    public void AfterTheMarker_TheConstraintIsInertAndReturnsLogitsUntouched()
    {
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<tool_call>"));
        var c = new ForcedToolCallConstraint(vocab, 1);
        c.Accept(1);

        Assert.False(c.IsConstraining);
        // Not merely "everything allowed" — the same span must come back, because an inert
        // constraint that still copied into scratch would allocate on every remaining token.
        Span<float> logits = new float[vocab.VocabSize];
        logits[7] = 42f;
        var masked = c.Filter(logits);
        Assert.Equal(vocab.VocabSize, masked.Length);
        Assert.Equal(42f, masked[7]);
        Assert.True(AllowedMask(c, vocab.VocabSize).All(a => a));
    }

    [Fact]
    public void AcceptingSomethingElse_LeavesItArmed()
    {
        // Can't happen while the constraint is honoured, but if a sampler ever ignored the mask the
        // constraint must not disarm on a token that isn't the marker — that would silently drop the
        // guarantee mid-turn.
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<tool_call>"));
        var c = new ForcedToolCallConstraint(vocab, 1);
        c.Accept(5);
        Assert.True(c.IsConstraining);
    }

    [Fact]
    public void Reset_ReArmsForTheNextTurn()
    {
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<tool_call>"));
        var c = new ForcedToolCallConstraint(vocab, 1);
        c.Accept(1);
        c.Reset();
        Assert.True(c.IsConstraining);
        Assert.True(AllowedMask(c, vocab.VocabSize)[1]);
    }

    [Fact]
    public void ShorterLogitsThanVocab_MasksWhatItWasGiven()
    {
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<tool_call>"));
        var c = new ForcedToolCallConstraint(vocab, 1);

        Span<float> logits = new float[8];
        var masked = c.Filter(logits);
        Assert.Equal(8, masked.Length);
        Assert.False(float.IsNegativeInfinity(masked[1]));
        Assert.True(float.IsNegativeInfinity(masked[0]));
    }

    [Fact]
    public void MarkerOutsideTheVocabulary_IsRejectedAtConstruction()
    {
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<tool_call>"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ForcedToolCallConstraint(vocab, vocab.VocabSize));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ForcedToolCallConstraint(vocab, 0));
    }

    /// <summary>
    /// Every dispatched family can be forced, each on the marker it actually emits first. Qwen3-Coder
    /// is the interesting one: its <c>&lt;function=&gt;</c> open tag is ordinary multi-token text, so
    /// it must force the <c>&lt;tool_call&gt;</c> envelope instead. DeepSeek likewise forces its
    /// OUTER block marker, not the inner one the argument grammar arms on.
    /// </summary>
    [Theory]
    [InlineData("qwen3", "<tool_call>")]
    [InlineData("qwen3coder", "<tool_call>")]
    [InlineData("llama", "<|python_tag|>")]
    [InlineData("deepseek2", "<|tool_calls_begin|>")]
    [InlineData("gemma4", "<|tool_call>")]
    public void EachFamilyForcesItsOwnOpenMarker(string architecture, string marker)
    {
        var tokenizer = new MarkerTokenizer("<unrelated>", marker);
        var vocab = new GrammarVocabulary(tokenizer);
        int markerId = tokenizer.SpecialTokens[marker];

        var c = ToolCallAdapterRegistry.Get(architecture).BuildForcedCallConstraint(vocab);

        Assert.NotNull(c);
        var allowed = AllowedMask(c!, vocab.VocabSize);
        Assert.True(allowed[markerId]);
        Assert.Equal(1, allowed.Count(a => a));
    }

    /// <summary>
    /// The refusal signal. A vocabulary without the family's marker must yield null so the caller
    /// rejects the request — returning an unforced constraint here would hand prose back to a client
    /// that asked for a guaranteed call, undetectably.
    /// </summary>
    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3coder")]
    [InlineData("llama")]
    [InlineData("deepseek2")]
    [InlineData("gemma4")]
    public void AFamilyWhoseMarkerIsAbsentFromTheVocabulary_CannotBeForced(string architecture)
    {
        var vocab = new GrammarVocabulary(new MarkerTokenizer("<unrelated>"));
        Assert.Null(ToolCallAdapterRegistry.Get(architecture).BuildForcedCallConstraint(vocab));
    }
}
