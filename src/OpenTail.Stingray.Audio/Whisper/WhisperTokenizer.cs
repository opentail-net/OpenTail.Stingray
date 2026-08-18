using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Multilingual BPE tokenizer and timestamp decoder for OpenAI Whisper.
/// Supports 99+ languages, special control tokens, and time-aligned subtitle tokens (&lt;|0.00|&gt; .. &lt;|30.00|&gt;).
/// </summary>
public sealed class WhisperTokenizer
{
    // Standard Whisper Special Tokens (Multilingual)
    public const int EndOfText = 50257;
    public const int StartOfTranscript = 50258;
    public const int EnglishLanguageToken = 50259;
    public const int TranslateToken = 50358;
    public const int TranscribeToken = 50359;
    public const int StartOfLmToken = 50360;
    public const int StartOfPrevToken = 50361;
    public const int NoSpeechToken = 50362;
    public const int NoTimestampsToken = 50363;
    public const int TimestampBegin = 50364; // <|0.00|>
    public const int TimestampEnd = 51864;   // <|30.00|>

    // Language codes mapped to token IDs in standard multilingual Whisper
    private static readonly string[] LanguageCodes =
    [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
    ];

    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = [];
    private readonly Dictionary<string, int> _langTokenMap = new(StringComparer.OrdinalIgnoreCase);

    public WhisperTokenizer(IReadOnlyDictionary<string, int>? customVocab = null)
    {
        // Populate standard language mapping
        for (int i = 0; i < LanguageCodes.Length; i++)
        {
            int tokenId = StartOfTranscript + 1 + i;
            _langTokenMap[LanguageCodes[i]] = tokenId;
        }

        if (customVocab != null)
        {
            foreach (var kvp in customVocab)
            {
                _vocab[kvp.Key] = kvp.Value;
                _idToToken[kvp.Value] = kvp.Key;
            }
        }
        else
        {
            // Build fallback standard token mappings for basic transcription
            InitializeDefaultVocab();
        }
    }

    private void InitializeDefaultVocab()
    {
        _vocab["<|endoftext|>"] = EndOfText;
        _vocab["<|startoftranscript|>"] = StartOfTranscript;
        _vocab["<|translate|>"] = TranslateToken;
        _vocab["<|transcribe|>"] = TranscribeToken;
        _vocab["<|startoflm|>"] = StartOfLmToken;
        _vocab["<|startofprev|>"] = StartOfPrevToken;
        _vocab["<|nospeech|>"] = NoSpeechToken;
        _vocab["<|notimestamps|>"] = NoTimestampsToken;

        foreach (var (lang, id) in _langTokenMap)
        {
            string tag = $"<|{lang}|>";
            _vocab[tag] = id;
        }

        for (int i = 0; i <= 1500; i++)
        {
            float seconds = i * 0.02f;
            string tag = $"<|{seconds:F2}|>";
            int id = TimestampBegin + i;
            _vocab[tag] = id;
        }

        foreach (var kvp in _vocab)
        {
            _idToToken[kvp.Value] = kvp.Key;
        }
    }

    /// <summary>
    /// Gets the token ID for a language code (e.g. "en" -> 50259).
    /// </summary>
    public int GetLanguageToken(string language)
    {
        if (_langTokenMap.TryGetValue(language, out int tokenId))
        {
            return tokenId;
        }
        return EnglishLanguageToken;
    }

    /// <summary>
    /// Constructs initial decoder prompt tokens based on language, task, and timestamp options.
    /// Format: [&lt;|startoftranscript|&gt;, &lt;|lang|&gt;, &lt;|transcribe|translate|&gt;, (&lt;|notimestamps|&gt;)]
    /// </summary>
    public int[] BuildInitialPrompt(string? language, SpeechTask task, bool enableTimestamps)
    {
        var tokens = new List<int>
        {
            StartOfTranscript,
            string.IsNullOrEmpty(language) ? EnglishLanguageToken : GetLanguageToken(language),
            task == SpeechTask.Translate ? TranslateToken : TranscribeToken
        };

        if (!enableTimestamps)
        {
            tokens.Add(NoTimestampsToken);
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decodes a sequence of Whisper tokens into text and time-stamped segments.
    /// </summary>
    public (string FullText, List<SpeechSegment> Segments) DecodeSegments(ReadOnlySpan<int> tokens, TimeSpan chunkOffset)
    {
        var segments = new List<SpeechSegment>();
        var fullTextBuilder = new StringBuilder();
        var currentSegmentText = new StringBuilder();
        var currentSegmentTokens = new List<int>();

        TimeSpan segStart = chunkOffset;
        TimeSpan segEnd = chunkOffset;
        bool inSegment = false;
        int segmentId = 0;

        for (int i = 0; i < tokens.Length; i++)
        {
            int token = tokens[i];

            if (token == EndOfText || token == StartOfTranscript)
            {
                continue;
            }

            if (IsTimestamp(token))
            {
                float timeSeconds = (token - TimestampBegin) * 0.02f;
                TimeSpan timestamp = chunkOffset + TimeSpan.FromSeconds(timeSeconds);

                if (!inSegment)
                {
                    segStart = timestamp;
                    inSegment = true;
                }
                else
                {
                    segEnd = timestamp;
                    string segText = currentSegmentText.ToString().Trim();
                    if (segText.Length > 0)
                    {
                        segments.Add(new SpeechSegment
                        {
                            Id = segmentId++,
                            Start = segStart,
                            End = segEnd,
                            Text = segText,
                            Tokens = currentSegmentTokens.ToArray()
                        });

                        if (fullTextBuilder.Length > 0) fullTextBuilder.Append(' ');
                        fullTextBuilder.Append(segText);
                    }

                    currentSegmentText.Clear();
                    currentSegmentTokens.Clear();
                    inSegment = false;
                }
                continue;
            }

            if (IsSpecialToken(token))
            {
                continue;
            }

            // Normal text token
            currentSegmentTokens.Add(token);
            string tokenStr = DecodeToken(token);
            currentSegmentText.Append(tokenStr);
        }

        // Catch trailing text if no closing timestamp was emitted
        if (currentSegmentText.Length > 0)
        {
            string segText = currentSegmentText.ToString().Trim();
            if (segText.Length > 0)
            {
                segments.Add(new SpeechSegment
                {
                    Id = segmentId++,
                    Start = segStart,
                    End = segEnd > segStart ? segEnd : segStart + TimeSpan.FromSeconds(1),
                    Text = segText,
                    Tokens = currentSegmentTokens.ToArray()
                });

                if (fullTextBuilder.Length > 0) fullTextBuilder.Append(' ');
                fullTextBuilder.Append(segText);
            }
        }

        return (fullTextBuilder.ToString(), segments);
    }

    public static bool IsTimestamp(int token) => token >= TimestampBegin && token <= TimestampEnd;

    public static bool IsSpecialToken(int token) => token >= EndOfText && token < TimestampBegin;

    public string DecodeToken(int token)
    {
        if (_idToToken.TryGetValue(token, out var str))
        {
            // Replace GPT2 BPE leading space character '\u0120' with ' '
            return str.Replace('\u0120', ' ');
        }

        // Fallback for byte-level tokens
        if (token >= 0 && token < 256)
        {
            return ((char)token).ToString();
        }

        return string.Empty;
    }

    /// <summary>
    /// Decodes a sequence of tokens directly into a text string.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < tokens.Length; i++)
        {
            int t = tokens[i];
            if (!IsSpecialToken(t) && !IsTimestamp(t))
            {
                sb.Append(DecodeToken(t));
            }
        }
        return sb.ToString().Trim();
    }
}
