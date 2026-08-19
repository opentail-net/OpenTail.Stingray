using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Multilingual BPE tokenizer and timestamp decoder for OpenAI Whisper (supporting v1/v2 and Large-v3/Large-v3-Turbo with 100 languages).
/// </summary>
public sealed class WhisperTokenizer
{
    // Standard Whisper Special Tokens (v1 / v2)
    public const int EndOfText = 50257;
    public const int StartOfTranscript = 50258;
    public const int EnglishLanguageToken = 50259;

    public bool IsV3 { get; }
    public int TranslateToken => IsV3 ? 50359 : 50358;
    public int TranscribeToken => IsV3 ? 50360 : 50359;
    public int StartOfLmToken => IsV3 ? 50361 : 50360;
    public int StartOfPrevToken => IsV3 ? 50362 : 50361;
    public int NoSpeechToken => IsV3 ? 50363 : 50362;
    public int NoTimestampsToken => IsV3 ? 50364 : 50363;
    public int TimestampBegin => IsV3 ? 50365 : 50364; // <|0.00|>
    public int TimestampEnd => IsV3 ? 51865 : 51864;   // <|30.00|>

    // Language codes mapped to token IDs in standard multilingual Whisper
    private static readonly string[] LanguageCodesV1V2 =
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

    // Language codes for Large-v3 and Large-v3-Turbo (includes Cantonese "yue" as 100th language)
    private static readonly string[] LanguageCodesV3 =
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
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"
    ];

    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = [];
    private readonly Dictionary<string, int> _langTokenMap = new(StringComparer.OrdinalIgnoreCase);

    public WhisperTokenizer() : this(null, false)
    {
    }

    public WhisperTokenizer(IReadOnlyDictionary<string, int>? customVocab) : this(customVocab, false)
    {
    }

    public WhisperTokenizer(IReadOnlyDictionary<string, int>? customVocab, bool isV3)
    {
        IsV3 = isV3;
        string[] languages = isV3 ? LanguageCodesV3 : LanguageCodesV1V2;

        // Populate language mapping
        for (int i = 0; i < languages.Length; i++)
        {
            int tokenId = StartOfTranscript + 1 + i;
            _langTokenMap[languages[i]] = tokenId;
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
            InitializeDefaultVocab();
        }
    }

    public static WhisperTokenizer CreateV3() => new(null, isV3: true);
    public static WhisperTokenizer CreateV2() => new(null, isV3: false);

    public static WhisperTokenizer FromGgml(WhisperGgmlModel ggml)
    {
        var dict = new Dictionary<string, int>(ggml.VocabSize, StringComparer.Ordinal);
        for (int i = 0; i < ggml.TokenById.Length; i++)
        {
            dict[ggml.TokenById[i]] = i;
        }
        return new WhisperTokenizer(dict, isV3: ggml.VocabSize == 51866);
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
    /// Gets the token ID for a language code (e.g. "en" -> 50259, "yue" -> 50358 in v3).
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
            GetLanguageToken(language ?? "en"),
            (task == SpeechTask.Translate) ? TranslateToken : TranscribeToken
        };

        if (!enableTimestamps)
        {
            tokens.Add(NoTimestampsToken);
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Determines if a given token ID represents a time-aligned timestamp token (&lt;|0.00|&gt; .. &lt;|30.00|&gt;).
    /// </summary>
    public bool IsTimestampToken(int tokenId)
    {
        return tokenId >= TimestampBegin && tokenId <= TimestampEnd;
    }

    /// <summary>
    /// Converts a timestamp token ID to its duration in seconds.
    /// </summary>
    public float TokenToSeconds(int tokenId)
    {
        if (!IsTimestampToken(tokenId)) return 0.0f;
        return (tokenId - TimestampBegin) * 0.02f;
    }

    /// <summary>
    /// Decodes a sequence of tokens into text segments with timestamps.
    /// </summary>
    public (string FullText, List<SpeechSegment> Segments) DecodeSegments(
        ReadOnlySpan<int> tokens,
        TimeSpan timeOffset) => DecodeWithTimestamps(tokens, timeOffset);

    /// <summary>
    /// Decodes a sequence of tokens into text segments with timestamps.
    /// </summary>
    public (string FullText, List<SpeechSegment> Segments) DecodeWithTimestamps(
        ReadOnlySpan<int> tokens,
        TimeSpan timeOffset)
    {
        var sbFull = new StringBuilder();
        var segments = new List<SpeechSegment>();

        var currentText = new StringBuilder();
        var currentTokens = new List<int>();
        float? segmentStartSec = null;
        int segmentId = 0;

        foreach (int tokenId in tokens)
        {
            if (tokenId == EndOfText || tokenId == StartOfTranscript)
            {
                continue;
            }

            if (IsTimestampToken(tokenId))
            {
                float sec = TokenToSeconds(tokenId);

                if (segmentStartSec == null)
                {
                    segmentStartSec = sec;
                }
                else
                {
                    float startSec = segmentStartSec.Value;
                    float endSec = sec;
                    if (endSec < startSec) endSec = startSec + 0.1f;

                    string segStr = currentText.ToString().Trim();
                    if (segStr.Length > 0)
                    {
                        segments.Add(new SpeechSegment
                        {
                            Id = segmentId++,
                            Start = timeOffset + TimeSpan.FromSeconds(startSec),
                            End = timeOffset + TimeSpan.FromSeconds(endSec),
                            Text = segStr,
                            Tokens = currentTokens.ToArray(),
                            Probability = 1.0f
                        });

                        if (sbFull.Length > 0) sbFull.Append(' ');
                        sbFull.Append(segStr);
                    }

                    currentText.Clear();
                    currentTokens.Clear();
                    segmentStartSec = null;
                }
            }
            else
            {
                currentTokens.Add(tokenId);
                string piece = DecodeSingleToken(tokenId);
                currentText.Append(piece);
            }
        }

        // Emit any trailing segment
        if (currentText.Length > 0)
        {
            string segStr = currentText.ToString().Trim();
            if (segStr.Length > 0)
            {
                float startSec = segmentStartSec ?? 0.0f;
                float endSec = startSec + 1.0f;

                segments.Add(new SpeechSegment
                {
                    Id = segmentId++,
                    Start = timeOffset + TimeSpan.FromSeconds(startSec),
                    End = timeOffset + TimeSpan.FromSeconds(endSec),
                    Text = segStr,
                    Tokens = currentTokens.ToArray(),
                    Probability = 1.0f
                });

                if (sbFull.Length > 0) sbFull.Append(' ');
                sbFull.Append(segStr);
            }
        }

        return (sbFull.ToString(), segments);
    }

    private string DecodeSingleToken(int tokenId)
    {
        if (_idToToken.TryGetValue(tokenId, out string? text))
        {
            if (text.StartsWith("<|") && text.EndsWith("|>"))
            {
                return string.Empty; // Skip control tags in text stream
            }
            // Byte-pair replacement for leading whitespace 'Ġ'
            return text.Replace("Ġ", " ");
        }

        // Fallback for character / byte fallback tokens
        if (tokenId >= 0 && tokenId < 256)
        {
            return ((char)tokenId).ToString();
        }

        return " ";
    }
}
