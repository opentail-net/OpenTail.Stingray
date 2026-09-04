using System.Text.RegularExpressions;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 prompt assembly + tokenization (`MiniMaxMusic3TokenizeStep`,
/// `_clean_caption`, `_normalize_lyrics`), transcribed directly from
/// `diffusers/modular_pipelines/minimax_music3/encoders.py`. See
/// docs/066-minimax-music3-future-plan.md, "Real autoregressive generation loop, fully specified
/// from source" for the archaeology this was captured from.
///
/// <para>Real tokenizer is a stock `Qwen2Tokenizer` (BPE) -- this checkpoint's own
/// `tokenizer/tokenizer.json` (real vocab=200000, confirmed real special-token ids
/// `&lt;|audio_cfg|&gt;`=151654, `&lt;|audio_end|&gt;`=151670 match this project's earlier
/// archaeology) is loaded via the engine's existing generic <see cref="HuggingFaceTokenizerSource"/>
/// + <see cref="GgufTokenizer.FromSource"/> BPE path -- no new tokenizer engine needed.</para>
/// </summary>
public sealed partial class MiniMaxMusic3PromptEncoder
{
    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";
    private const string CaptionStart = "<|caption_start|>";
    private const string CaptionEnd = "<|caption_end|>";
    private const string LyricsStart = "<|lyrics_start|>";
    private const string LyricsEnd = "<|lyrics_end|>";
    private const string AudioStart = "<|audio_start|>";

    private readonly GgufTokenizer _tokenizer;

    private MiniMaxMusic3PromptEncoder(GgufTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    /// <summary>Loads the real checkpoint tokenizer from a package directory containing
    /// `tokenizer.json` (e.g. `models/minimax-music3/tokenizer/`).</summary>
    public static MiniMaxMusic3PromptEncoder Load(string tokenizerPackageDir)
    {
        var result = HuggingFaceTokenizerSource.Load(tokenizerPackageDir);
        if (!result.IsUsable)
        {
            string reasons = string.Join("; ", result.Rejections.Select(r => r.ToString()));
            throw new InvalidOperationException($"Could not load MiniMax-Music3 tokenizer from '{tokenizerPackageDir}': {reasons}");
        }
        return new MiniMaxMusic3PromptEncoder(GgufTokenizer.FromSource(result.Source!));
    }

    /// <summary>Real prompt assembly + tokenization. Returns the real conditional token sequence
    /// (ending in `&lt;|audio_start|&gt;`) -- the real CFG-null counterpart is derived from this by
    /// <see cref="MiniMaxMusic3AutoregressiveGenerator"/>'s own `BuildUnconditionalPrompt`, matching
    /// the real `text_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID` slice exactly.</summary>
    public int[] BuildConditionalPrompt(string musicDescription, string lyrics)
    {
        string text = ImStart + CaptionStart + CleanCaption(musicDescription) + CaptionEnd
            + LyricsStart + NormalizeLyrics(lyrics) + LyricsEnd + ImEnd + AudioStart;

        var tokens = _tokenizer.Encode(text);
        if (tokens.Count > MiniMaxMusic3Config.MaxPromptTokens)
            throw new InvalidOperationException($"The assembled prompt has {tokens.Count} tokens; the maximum is {MiniMaxMusic3Config.MaxPromptTokens}");

        return [.. tokens];
    }

    /// <summary>Real `_clean_caption`: rewrites `&lt;|tag arg|&gt;`-style special tags to
    /// "tag is arg", strips markdown headings/bullets/bold/italic/hr-rules line by line, drops
    /// U+FFFD-space and 4-space runs, then collapses blank-line runs.</summary>
    internal static string CleanCaption(string caption)
    {
        string text = SpecialTagRegex().Replace(caption, m =>
        {
            string inner = m.Groups[1].Value.Trim();
            var parts = inner.Split((char[]?)null, 2, StringSplitOptions.RemoveEmptyEntries);
            return parts.Length == 2 ? $"{parts[0]} is {parts[1]}" : inner;
        });

        var linesOut = new List<string>();
        foreach (var rawLine in text.Split('\n'))
        {
            string line = HeadingRegex().Replace(rawLine, "");
            line = BulletDashRegex().Replace(line, "");
            line = BulletStarRegex().Replace(line, "");
            while (line.Contains("**"))
            {
                string updated = BoldRegex().Replace(line, "$1");
                if (updated == line) break;
                line = updated;
            }
            line = ItalicRegex().Replace(line, "$1");
            linesOut.Add(line.TrimEnd());
        }
        text = string.Join("\n", linesOut);
        text = HorizontalRuleRegex().Replace(text, "");
        text = text.Replace("� ", "").Replace("    ", "");
        return MultiNewlineRegex().Replace(text, "\n");
    }

    /// <summary>Real `_normalize_lyrics`: keeps only consecutive leading `[tag]` structure markers
    /// per line (text sharing a tag's line is dropped), forces each tag onto its own line, lowercases
    /// tag contents, and prepends a real `[start]` marker line.</summary>
    internal static string NormalizeLyrics(string lyrics)
    {
        var output = new List<string>();
        foreach (var line in lyrics.Split('\n'))
        {
            var match = LeadingTagsRegex().Match(line);
            output.Add(match.Success ? match.Groups[1].Value.Trim() : line);
        }
        string text = string.Join("\n", output);
        text = text.Replace("] ", "]\n").Replace(" [", "\n[");
        text = text.Replace(" ^ ", "\n");
        text = TagContentRegex().Replace(text, m => $"[{m.Groups[1].Value.ToLowerInvariant()}]");
        return $"[start]\n{text}";
    }

    [GeneratedRegex(@"<\|([^|]*)\|>")]
    private static partial Regex SpecialTagRegex();

    [GeneratedRegex(@"^\s{0,3}#{1,6}\s+")]
    private static partial Regex HeadingRegex();

    [GeneratedRegex(@"^\s*[*+-]\s+")]
    private static partial Regex BulletDashRegex();

    [GeneratedRegex(@"^\s*\*\s+")]
    private static partial Regex BulletStarRegex();

    [GeneratedRegex(@"\*\*([^*]+)\*\*")]
    private static partial Regex BoldRegex();

    [GeneratedRegex(@"(?<!\*)\*([^*\n]+)\*(?!\*)")]
    private static partial Regex ItalicRegex();

    [GeneratedRegex(@"^\s*[-*_]{3,}\s*$", RegexOptions.Multiline)]
    private static partial Regex HorizontalRuleRegex();

    [GeneratedRegex(@"\n{2,}")]
    private static partial Regex MultiNewlineRegex();

    [GeneratedRegex(@"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")]
    private static partial Regex LeadingTagsRegex();

    [GeneratedRegex(@"\[([^\]]+)\]")]
    private static partial Regex TagContentRegex();
}
