using System.Diagnostics.CodeAnalysis;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Byte-level BPE pre-tokenizer split patterns, keyed on the GGUF <c>tokenizer.ggml.pre</c> field.
///
/// <para>A byte-level BPE vocabulary does not fully determine its tokenization: the merge table says
/// how pieces combine, and the pre-tokenizer regex says where the text is cut into pieces in the
/// first place. Two models can share a vocabulary and still tokenize the same string differently.
/// GGUF records which pre-tokenizer a model was trained with in <c>tokenizer.ggml.pre</c>, and
/// getting it wrong is silent: the pieces still reassemble, so <c>Decode(Encode(s)) == s</c> holds
/// and only the token IDs — the thing the model actually sees — are wrong.</para>
///
/// <para><b>Patterns are ported from llama.cpp</b> (MIT), <c>src/llama-vocab.cpp</c>, the
/// <c>llm_tokenizer_bpe</c> constructor; local reference copy at
/// <c>examples/llama.cpp/llama.cpp/src/llama-vocab.cpp</c>, binary build <c>b8585-cpu</c>. Each
/// pre-type maps to an ORDERED LIST, not one pattern: llama.cpp's <c>unicode_regex_split</c> applies
/// them as a cascade, each one further splitting the pieces produced by the previous. Several
/// pre-types rely on that (<c>smollm</c> splits digits out first, then applies the GPT-2 pattern to
/// what is left).</para>
///
/// <para><b>Known deviation:</b> llama.cpp splits over Unicode codepoints; .NET <see cref="Regex"/>
/// works over UTF-16 code units, so a character outside the BMP is two chars to a class like
/// <c>\p{L}</c>. This affects astral-plane text (emoji, rare CJK extensions) only, and matches the
/// caveat already carried by the Tekken pattern.</para>
/// </summary>
public static partial class PreTokenizerPatterns
{
    // --- Pattern sources -------------------------------------------------------------------
    // Each [GeneratedRegex] is compiled at build time, which is what keeps this NativeAOT-safe;
    // RegexOptions.Compiled would need runtime codegen and must not be used here.

    /// <summary>
    /// GPT-2's original split. Also the correct pattern for <c>mpt</c>, <c>olmo</c>, <c>jais</c>,
    /// <c>trillion</c> and <c>granite-docling</c>, which llama.cpp maps onto the same case.
    /// </summary>
    [GeneratedRegex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)""")]
    private static partial Regex Gpt2();

    /// <summary>Single digit. Used as the first stage of the StarCoder/SmolLM cascade.</summary>
    [GeneratedRegex("""\p{N}""")]
    private static partial Regex SingleDigit();

    /// <summary>
    /// Llama-3 family. Differs from GPT-2 in three ways that change real text: case-insensitive
    /// contractions, any single non-alphanumeric may attach to a following word (not just a space),
    /// and digit runs are capped at three.
    /// </summary>
    [GeneratedRegex("""(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")]
    private static partial Regex Llama3();

    /// <summary>
    /// JAIS-2. Identical to Llama-3 except the trailing whitespace-run alternative cascades
    /// through descending fixed lengths (512, 256, ..., 1) before falling to a single space —
    /// an optimization for text with very long whitespace runs (heavy code indentation).
    /// </summary>
    [GeneratedRegex("""(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}""")]
    private static partial Regex Jais2();

    /// <summary>
    /// Qwen-2 family (which is what Qwen3 GGUFs declare). As Llama-3 but with single digits.
    /// </summary>
    [GeneratedRegex("""(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")]
    private static partial Regex Qwen2();

    /// <summary>Qwen-3.5 family. As Qwen-2 but combining marks join their base letter.</summary>
    [GeneratedRegex("""(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")]
    private static partial Regex Qwen35();

    /// <summary>
    /// Mistral Tekken (Mistral-Nemo / Ministral / Pixtral). Taken from the model's own
    /// <c>tokenizer.json</c> <c>pre_tokenizer</c> pattern rather than llama.cpp's ASCII-approximated
    /// rewrite — llama.cpp rewrote it because <c>std::regex</c> lacks Unicode categories, and .NET
    /// does not have that limitation.
    /// </summary>
    [GeneratedRegex("""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""")]
    private static partial Regex Tekken();

    /// <summary>1-3 digit numbers. First stage of the Hunyuan-Dense/DeepSeek3/JoyAI cascade.</summary>
    [GeneratedRegex("""\p{N}{1,3}""")]
    private static partial Regex DigitRun3();

    /// <summary>CJK block (Han + Hiragana + Katakana). Second stage of the same cascade.</summary>
    [GeneratedRegex("""[一-龥぀-ゟ゠-ヿ]+""")]
    private static partial Regex Cjk();

    /// <summary>
    /// Hunyuan-Dense (distinct from the plain <c>hunyuan</c>/Qwen-2 pre-type above), DeepSeek3-LLM,
    /// and JoyAI-LLM share this third stage. Punctuation-then-Latin, then letters/marks, then
    /// punctuation/symbol runs, then whitespace.
    /// </summary>
    [GeneratedRegex("""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")]
    private static partial Regex HunyuanDenseTail();

    // --- Registry --------------------------------------------------------------------------

    /// <summary>
    /// Resolves the pre-tokenizer cascade for a <c>tokenizer.ggml.pre</c> value.
    /// </summary>
    /// <param name="pre">The raw metadata value. Empty or absent resolves to GPT-2, matching
    /// llama.cpp's default for a BPE vocab that declares no pre-tokenizer.</param>
    /// <param name="patterns">The ordered cascade to apply.</param>
    /// <returns>
    /// <c>true</c> when <paramref name="pre"/> is a value this table recognises. <c>false</c> means
    /// the value is unknown, and <paramref name="patterns"/> is then the GPT-2 fallback — usable,
    /// but the caller should surface that the model's declared pre-tokenizer is not implemented,
    /// because the failure it produces is silent.
    /// </returns>
    public static bool TryResolve(string? pre, out Regex[] patterns)
    {
        switch (pre)
        {
            // llama.cpp: LLAMA_VOCAB_PRE_TYPE_GPT2 and the cases folded onto it. exaone4 is a
            // deliberate llama.cpp choice, not an oversight — confirmed against llama-vocab.cpp:
            // tokenizer_pre=="exaone4" maps to the plain GPT2 pre_type, distinct from "exaone"
            // (already covered below under the SmolLM/digit-split group) and "exaone-moe" (its
            // own dedicated pre_type, not yet ported).
            case null or "":
            case "gpt-2":
            case "mpt":
            case "olmo":
            case "jais":
            case "trillion":
            case "granite-docling":
            case "exaone4":
                patterns = [Gpt2()];
                return true;

            // llama.cpp: LLAMA_VOCAB_PRE_TYPE_SMOLLM and the cases folded onto it. Two stages —
            // digits are split out individually BEFORE the GPT-2 pattern sees the text.
            case "smollm":
            case "starcoder":
            case "refact":
            case "command-r":
            case "codeshell":
            case "exaone":
            case "minerva-7b":
            case "mellum2":
                patterns = [SingleDigit(), Gpt2()];
                return true;

            case "llama3":
            case "llama-bpe":
            case "dbrx":
            case "smaug-bpe":
                patterns = [Llama3()];
                return true;

            case "jais-2":
                patterns = [Jais2()];
                return true;

            case "qwen2":
            case "stablelm2":
            case "hunyuan":
            case "solar-open":
                patterns = [Qwen2()];
                return true;

            case "qwen35":
                patterns = [Qwen35()];
                return true;

            case "tekken":
                patterns = [Tekken()];
                return true;

            // llama.cpp: LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE and the cases folded onto it
            // (DEEPSEEK3_LLM, JOYAI_LLM) — distinct from plain "hunyuan" above, which uses the
            // Qwen-2 cascade instead.
            case "hunyuan-dense":
            case "deepseek3-llm":
            case "joyai-llm":
                patterns = [DigitRun3(), Cjk(), HunyuanDenseTail()];
                return true;

            default:
                patterns = [Gpt2()];
                return false;
        }
    }

    /// <summary>
    /// Applies a cascade to <paramref name="text"/>, returning the pieces in order. Mirrors
    /// llama.cpp's <c>unicode_regex_split</c>: every regex splits the pieces the previous one
    /// produced, and an unmatched gap is itself a piece — dropping gaps would silently discard input.
    /// </summary>
    public static List<string> Split(string text, Regex[] patterns)
    {
        ArgumentNullException.ThrowIfNull(patterns);

        var pieces = new List<string>(1) { text };
        var next = new List<string>();

        foreach (var pattern in patterns)
        {
            next.Clear();
            foreach (var piece in pieces)
                SplitOne(piece, pattern, next);
            (pieces, next) = (next, pieces);
        }

        return pieces;
    }

    private static void SplitOne(string piece, Regex pattern, List<string> into)
    {
        if (piece.Length == 0) return;

        int pos = 0;
        foreach (Match m in pattern.Matches(piece))
        {
            if (m.Index > pos) into.Add(piece[pos..m.Index]);
            if (m.Length > 0) into.Add(m.Value);
            pos = m.Index + m.Length;
        }
        if (pos < piece.Length) into.Add(piece[pos..]);
    }

    /// <summary>
    /// Every <c>tokenizer.ggml.pre</c> value this table implements. Exposed so diagnostics and tests
    /// can enumerate coverage rather than restating the list.
    /// </summary>
    [SuppressMessage("Performance", "CA1861:Avoid constant arrays as arguments",
        Justification = "Read once by diagnostics and tests, not on a hot path.")]
    public static IReadOnlyList<string> KnownPreTypes { get; } =
    [
        "gpt-2", "mpt", "olmo", "jais", "trillion", "granite-docling", "exaone4",
        "smollm", "starcoder", "refact", "command-r", "codeshell", "exaone", "minerva-7b", "mellum2",
        "llama3", "llama-bpe", "dbrx", "smaug-bpe",
        "jais-2",
        "qwen2", "stablelm2", "hunyuan", "solar-open",
        "qwen35",
        "tekken",
        "hunyuan-dense", "deepseek3-llm", "joyai-llm",
    ];
}
