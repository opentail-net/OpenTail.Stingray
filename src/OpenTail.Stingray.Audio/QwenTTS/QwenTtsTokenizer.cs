
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Tokenizer and prompt formatter for Qwen3-TTS 12Hz supporting ChatML, language tags, named speakers, and voice design.
/// </summary>
public sealed partial class QwenTtsTokenizer
{
    public const int ImStartTokenId = 151644;
    public const int ImEndTokenId = 151645;
    public const int TtsBosTokenId = 151646;
    public const int TtsEosTokenId = 151647;
    public const int TtsPadTokenId = 151648;
    public const int CodecBosTokenId = 151649;
    public const int CodecEosTokenId = 151650;
    public const int CodecPadTokenId = 151651;

    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = [];
    private readonly Dictionary<string, int> _speakerToId = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _speakerDialects = new(StringComparer.OrdinalIgnoreCase);

    [GeneratedRegex(@"([a-zA-Z]+|[\d]+|[^\s\w]|[\s]+|<[^>]+>|\[[a-zA-Z0-9_]+\])", RegexOptions.Compiled)]
    private static partial Regex TokenRegex();

    public QwenTtsTokenizer()
    {
        InitializeVocab();
        InitializeSpeakers();
    }

    private void InitializeVocab()
    {
        _vocab["<|im_start|>"] = ImStartTokenId;
        _vocab["<|im_end|>"] = ImEndTokenId;
        _vocab["<|tts_bos|>"] = TtsBosTokenId;
        _vocab["<|tts_eos|>"] = TtsEosTokenId;
        _vocab["<|tts_pad|>"] = TtsPadTokenId;
        _vocab["<|codec_bos|>"] = CodecBosTokenId;
        _vocab["<|codec_eos|>"] = CodecEosTokenId;
        _vocab["<|codec_pad|>"] = CodecPadTokenId;

        // Languages and dialect tags
        string[] tags =
        [
            "[en]", "[zh]", "[ja]", "[ko]", "[yue]", "[sichuan]", "[beijing]",
            "[fr]", "[de]", "[es]", "[ru]", "[it]", "[pt]"
        ];

        int id = 1000;
        foreach (string tag in tags)
        {
            _vocab[tag] = id;
            _idToToken[id] = tag;
            id++;
        }

        // ASCII chars and standard punctuation
        string chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n\t";
        foreach (char c in chars)
        {
            string s = c.ToString();
            if (!_vocab.ContainsKey(s))
            {
                _vocab[s] = id;
                _idToToken[id] = s;
                id++;
            }
        }
    }

    private void InitializeSpeakers()
    {
        _speakerToId["eric"] = 1;
        _speakerDialects["eric"] = "sichuan";

        _speakerToId["dylan"] = 2;
        _speakerDialects["dylan"] = "beijing";

        _speakerToId["serena"] = 3;
        _speakerToId["alex"] = 4;
        _speakerToId["chelsea"] = 5;
    }

    /// <summary>
    /// Formats a complete ChatML prompt for Qwen3-TTS with speaker/language conditioning.
    /// </summary>
    public string FormatPrompt(string text, string voice, string? language = null, string? voiceDesignPrompt = null)
    {
        var sb = new StringBuilder();

        // Resolve speaker dialect override if applicable
        string effectiveLang = language ?? "en";
        if (_speakerDialects.TryGetValue(voice, out string? dialect))
        {
            effectiveLang = dialect;
        }

        sb.Append("<|im_start|>system\n");
        if (!string.IsNullOrEmpty(voiceDesignPrompt))
        {
            sb.Append($"Voice Design: {voiceDesignPrompt}\n");
        }
        else
        {
            sb.Append($"Speaker: {voice}\nLanguage: {effectiveLang}\n");
        }
        sb.Append("<|im_end|>\n");

        sb.Append("<|im_start|>user\n");
        sb.Append(text);
        sb.Append("<|im_end|>\n");

        sb.Append("<|im_start|>assistant\n<|tts_bos|>");
        return sb.ToString();
    }

    /// <summary>
    /// Encodes formatted prompt text into token IDs.
    /// </summary>
    public int[] Encode(string promptText)
    {
        if (string.IsNullOrEmpty(promptText)) return [TtsBosTokenId];

        var tokens = new List<int>();
        var matches = TokenRegex().Matches(promptText);

        foreach (Match match in matches)
        {
            string chunk = match.Value;
            if (string.IsNullOrEmpty(chunk)) continue;

            if (_vocab.TryGetValue(chunk, out int id))
            {
                tokens.Add(id);
            }
            else
            {
                foreach (char c in chunk)
                {
                    string sc = c.ToString();
                    if (_vocab.TryGetValue(sc, out int charId))
                    {
                        tokens.Add(charId);
                    }
                    else
                    {
                        tokens.Add(200 + ((int)c % 256));
                    }
                }
            }
        }

        return tokens.ToArray();
    }
}
