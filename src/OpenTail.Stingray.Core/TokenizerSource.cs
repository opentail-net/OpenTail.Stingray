namespace OpenTail.Stingray.Core;

/// <summary>
/// Tokenizer metadata in a form independent of the package format that supplied it.
/// </summary>
/// <remarks>
/// <para>GGUF embeds its vocabulary in model metadata; a Hugging Face package keeps it in
/// <c>tokenizer.json</c> with special tokens and the chat template spread across
/// <c>tokenizer_config.json</c> and <c>special_tokens_map.json</c>. Both normalise into this record,
/// and <see cref="GgufTokenizer.FromSource"/> is the single construction path, so the two formats
/// cannot drift into behaving differently for the same vocabulary.</para>
///
/// <para>Token-type codes follow the GGUF convention because that is the vocabulary this codebase
/// already speaks; an adapter that has no type information leaves <see cref="TokenTypes"/> null and
/// supplies <see cref="AdditionalSpecialTokens"/> instead.</para>
/// </remarks>
public sealed record TokenizerSource
{
    /// <summary>GGUF token-type code for a control token (never produced by BPE merges).</summary>
    public const int ControlTokenType = 3;

    /// <summary>GGUF token-type code for a user-defined token such as a chat marker.</summary>
    public const int UserDefinedTokenType = 4;

    /// <summary>Vocabulary indexed by token id.</summary>
    public required string[] Tokens { get; init; }

    /// <summary>BPE merge rules in priority order, each "left right". Empty for non-BPE vocabularies.</summary>
    public string[] Merges { get; init; } = [];

    /// <summary>
    /// Per-token SentencePiece unigram scores (<c>tokenizer.ggml.scores</c>), aligned with
    /// <see cref="Tokens"/> by index. Null when the source has none. This is the REAL priority
    /// signal for classic SentencePiece BPE (<c>tokenizer.ggml.model=llama</c>) tokenization --
    /// <see cref="Merges"/> is a GGUF export convenience some converters also emit, not something
    /// llama.cpp's own <c>llm_tokenizer_spm</c> reads at all. A model can (and some real
    /// checkpoints do) ship neither array; llama.cpp's fallback there is to treat every score as
    /// 0.0f and still tokenize correctly, since its merge algorithm is gated on vocabulary
    /// membership, not on the scores/merges array being present.
    /// </summary>
    public float[]? Scores { get; init; }

    /// <summary>Per-token type codes, or null when the source does not classify tokens.</summary>
    public int[]? TokenTypes { get; init; }

    /// <summary>
    /// Special tokens the source names directly, merged over anything derived from
    /// <see cref="TokenTypes"/>. This is how a Hugging Face package contributes its added tokens.
    /// </summary>
    public IReadOnlyDictionary<string, int> AdditionalSpecialTokens { get; init; } =
        new Dictionary<string, int>(StringComparer.Ordinal);

    /// <summary>
    /// Whether a leading space is prepended before tokenizing (real llama.cpp's
    /// <c>add_space_prefix</c>, default <c>true</c> for classic SentencePiece
    /// <c>tokenizer.ggml.model=llama</c>, overridable via <c>tokenizer.ggml.add_space_prefix</c>).
    /// Only consulted for genuine SPM tokenization; ignored otherwise.
    /// </summary>
    public bool AddSpacePrefix { get; init; }

    /// <summary>
    /// True when <see cref="Merges"/> is a real, authoritative rank-priority merge list (a genuine
    /// HF "fast tokenizer" BPE export of a SentencePiece-family vocabulary, e.g. Gemma/T5Gemma's
    /// <c>tokenizer.json</c>) rather than GGUF's <c>tokenizer.ggml.merges</c> convenience array
    /// (which real llama.cpp SPM tokenization, <c>tokenizer.ggml.model=llama</c>, does NOT use at
    /// all -- that real algorithm is score-based, see <see cref="GgufTokenizer.SpmMergePiecesByScore"/>'s
    /// doc comment). Defaults to <c>false</c> so every existing GGUF-sourced SPM tokenizer keeps
    /// using the score-based algorithm unchanged; only set <c>true</c> by a source that is
    /// genuinely rank-priority BPE with no unigram scores at all.
    /// </summary>
    public bool MergesAreRankPriority { get; init; }

    public int BosTokenId { get; init; } = 1;
    public int EosTokenId { get; init; } = 2;
    public int UnknownTokenId { get; init; }
    public int PadTokenId { get; init; } = 2;

    /// <summary>Whether the model expects BOS to be prepended by the caller.</summary>
    public bool AddBosToken { get; init; }

    /// <summary>
    /// Tokenizer family name (GGUF's <c>tokenizer.ggml.model</c>, or the equivalent for a package).
    /// Selects the SentencePiece-style construction path for <c>llama</c>/<c>gemma*</c>.
    /// </summary>
    public string ModelFamily { get; init; } = string.Empty;

    /// <summary>
    /// The GGUF <c>tokenizer.ggml.pre</c> value, naming the pre-tokenizer variant (e.g.
    /// <c>"tekken"</c> for the Mistral-Nemo family). Byte-level BPE vocabs that share a merges
    /// format still differ in how text is split before merging, and that split is not derivable
    /// from the vocab, so it has to be carried explicitly. Empty when the model declares none.
    /// </summary>
    public string TokenizerPre { get; init; } = string.Empty;

    /// <summary>
    /// A pre-tokenizer split regex read VERBATIM from a <c>tokenizer.json</c>'s own
    /// <c>pre_tokenizer</c> field (the "Split" stage of a "Sequence" pre-tokenizer, e.g. Qwen2's
    /// real `(?i:'s|'t|...)|[^\r\n\p{L}\p{N}]?\p{L}+|...` pattern), when present. Takes priority
    /// over <see cref="TokenizerPre"/>'s named-pattern lookup: a HF export's own declared regex is
    /// ground truth and needs no name-matching against <see cref="PreTokenizerPatterns"/>'s known
    /// GGUF <c>tokenizer.ggml.pre</c> table, which a <c>tokenizer.json</c> load never populates in
    /// the first place (GGUF-only metadata). Empty when the source has no such field, or the
    /// package is GGUF-derived (that path uses <see cref="TokenizerPre"/> instead).
    /// </summary>
    public string TokenizerPreRawRegex { get; init; } = string.Empty;

    /// <summary>Jinja chat template, or null when the package ships none.</summary>
    public string? ChatTemplate { get; init; }
}
