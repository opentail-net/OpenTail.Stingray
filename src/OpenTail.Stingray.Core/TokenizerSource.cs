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

    /// <summary>Per-token type codes, or null when the source does not classify tokens.</summary>
    public int[]? TokenTypes { get; init; }

    /// <summary>
    /// Special tokens the source names directly, merged over anything derived from
    /// <see cref="TokenTypes"/>. This is how a Hugging Face package contributes its added tokens.
    /// </summary>
    public IReadOnlyDictionary<string, int> AdditionalSpecialTokens { get; init; } =
        new Dictionary<string, int>(StringComparer.Ordinal);

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

    /// <summary>Jinja chat template, or null when the package ships none.</summary>
    public string? ChatTemplate { get; init; }
}
