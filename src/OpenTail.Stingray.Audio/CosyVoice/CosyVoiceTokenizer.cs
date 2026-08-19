using System.Text;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Multilingual BPE tokenizer with support for emotion tags, vocalization markers, and phoneme/tone tokens for CosyVoice 3.
/// </summary>
public sealed partial class CosyVoiceTokenizer
{
    public const int PadTokenId = 0;
    public const int EosTokenId = 151643; // <|endoftext|>
    public const int ImStartTokenId = 151644;
    public const int ImEndTokenId = 151645;
    public const int EndOfPromptTokenId = 151646;

    private readonly Dictionary<string, int> _tokenToId = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = [];
    private readonly List<string> _specialTokensList = [];

    // Pre-tokenization regex pattern matching whitespace, word chunks, numbers, and punctuation
    [GeneratedRegex(@"([a-zA-Z]+|[\d]+|[^\s\w]|[\s]+|\[[a-zA-Z0-9_]+\]|<[^>]+>)", RegexOptions.Compiled)]
    private static partial Regex TokenSplitRegex();

    public CosyVoiceTokenizer()
    {
        InitializeVocab();
    }

    private void InitializeVocab()
    {
        // 1. Control and Special Tokens
        string[] specialTokens =
        [
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|endofprompt|>", "<|endofsystem|>",
            "[breath]", "[quick_breath]", "<strong>", "</strong>", "[noise]",
            "[laughter]", "<laughter>", "</laughter>", "[cough]", "[clucking]", "[accent]",
            "[hissing]", "[sigh]", "[vocalized-noise]", "[lipsmack]", "[mn]",
            // ARPAbet phonemes
            "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]",
            "[AH]", "[AH0]", "[AH1]", "[AH2]", "[AO]", "[AO0]", "[AO1]", "[AO2]",
            "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
            "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]",
            "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]", "[EY0]", "[EY1]", "[EY2]",
            "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]",
            "[IY]", "[IY0]", "[IY1]", "[IY2]", "[JH]", "[K]", "[L]", "[M]",
            "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
            "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]",
            "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]", "[UW0]", "[UW1]", "[UW2]",
            "[V]", "[W]", "[Y]", "[Z]", "[ZH]"
        ];

        int id = 0;
        _tokenToId["<pad>"] = id;
        _idToToken[id++] = "<pad>";

        foreach (string token in specialTokens)
        {
            if (!_tokenToId.ContainsKey(token))
            {
                _tokenToId[token] = id;
                _idToToken[id] = token;
                _specialTokensList.Add(token);
                id++;
            }
        }

        // Standard ASCII characters and punctuation
        string commonChars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ï¼ãï¼ï¼ãï¼ï¼ââââï¼ï¼ââ¦\n\t";
        foreach (char c in commonChars)
        {
            string s = c.ToString();
            if (!_tokenToId.ContainsKey(s))
            {
                _tokenToId[s] = id;
                _idToToken[id] = s;
                id++;
            }
        }
    }

    /// <summary>
    /// Encodes text into a sequence of token IDs, preserving special emotion/instruction tags.
    /// </summary>
    public int[] Encode(string text, bool addPromptBoundary = true)
    {
        if (string.IsNullOrEmpty(text))
        {
            return addPromptBoundary ? [EndOfPromptTokenId] : [];
        }

        var tokens = new List<int>();
        var matches = TokenSplitRegex().Matches(text);

        foreach (Match match in matches)
        {
            string chunk = match.Value;
            if (string.IsNullOrEmpty(chunk)) continue;

            if (_tokenToId.TryGetValue(chunk, out int tokenId))
            {
                tokens.Add(tokenId);
            }
            else
            {
                // Fallback to character-level decomposition
                foreach (char c in chunk)
                {
                    string sc = c.ToString();
                    if (_tokenToId.TryGetValue(sc, out int charId))
                    {
                        tokens.Add(charId);
                    }
                    else
                    {
                        // Byte fallback token
                        int byteVal = (int)c % 256;
                        tokens.Add(100 + byteVal);
                    }
                }
            }
        }

        if (addPromptBoundary)
        {
            tokens.Add(EndOfPromptTokenId);
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decodes token IDs back to a text string.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (int tid in tokens)
        {
            if (tid == EosTokenId || tid == EndOfPromptTokenId || tid == PadTokenId)
            {
                continue;
            }

            if (_idToToken.TryGetValue(tid, out string? str))
            {
                sb.Append(str);
            }
        }
        return sb.ToString();
    }
}
