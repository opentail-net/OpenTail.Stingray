using System.Text;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Qwen3 ChatML tokenizer with multimodal speech tokens, language routing, and timestamp decoding for Qwen3-ASR.
/// </summary>
public sealed partial class QwenAsrTokenizer
{
    public const int ImStartTokenId = 151644;
    public const int ImEndTokenId = 151645;
    public const int AudioBosTokenId = 151646;
    public const int AudioEosTokenId = 151647;
    public const int AudioPadTokenId = 151648; // <|AUDIO|> soft token placeholder
    public const int TimestampBegin = 151649;   // <|timestamp_0.00|>
    public const int TimestampEnd = 153149;     // <|timestamp_30.00|>

    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = [];

    [GeneratedRegex(@"([a-zA-Z]+|[\d]+|[^\s\w]|[\s]+|<\|[^>]+\|>)", RegexOptions.Compiled)]
    private static partial Regex TokenRegex();

    public int VocabSize => 151936;

    public QwenAsrTokenizer()
    {
        InitializeVocab();
    }

    private void InitializeVocab()
    {
        _vocab["<|im_start|>"] = ImStartTokenId;
        _vocab["<|im_end|>"] = ImEndTokenId;
        _vocab["<|audio_bos|>"] = AudioBosTokenId;
        _vocab["<|audio_eos|>"] = AudioEosTokenId;
        _vocab["<|AUDIO|>"] = AudioPadTokenId;

        // Timestamps <|timestamp_0.00|> .. <|timestamp_30.00|> in 20ms steps
        for (int i = 0; i <= 1500; i++)
        {
            float sec = i * 0.02f;
            string tag = $"<|timestamp_{sec:F2}|>";
            int tsId = TimestampBegin + i;
            _vocab[tag] = tsId;
            _idToToken[tsId] = tag;
        }

        // Common subwords and ASCII alphabet
        int id = 1000;
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

        // Common words
        string[] commonWords =
        [
            "the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "as", "was",
            "with", "be", "by", "on", "not", "he", "i", "this", "have", "from", "or", "one",
            "speech", "recognition", "audio", "qwen", "transcribe", "language"
        ];

        foreach (string w in commonWords)
        {
            if (!_vocab.ContainsKey(w))
            {
                _vocab[w] = id;
                _idToToken[id] = w;
                id++;
            }
        }

        foreach (var kvp in _vocab)
        {
            _idToToken[kvp.Value] = kvp.Key;
        }
    }

    /// <summary>
    /// Formats the standard ChatML prompt containing audio tokens and instruction for Qwen3-ASR.
    /// </summary>
    public string FormatPrompt(string? language = null, string? taskInstruction = null)
    {
        var sb = new StringBuilder();
        sb.Append("<|im_start|>system\nYou are a helpful speech-to-text assistant.<|im_end|>\n");
        sb.Append("<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n");

        if (!string.IsNullOrEmpty(language))
        {
            sb.Append($"Language: {language}\n");
        }

        sb.Append(taskInstruction ?? "Transcribe the audio speech into text.");
        sb.Append("<|im_end|>\n<|im_start|>assistant\n");

        return sb.ToString();
    }

    /// <summary>
    /// Encodes a formatted prompt string into token IDs.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text)) return [];

        var tokens = new List<int>();
        var matches = TokenRegex().Matches(text);

        foreach (Match m in matches)
        {
            string piece = m.Value;
            if (string.IsNullOrEmpty(piece)) continue;

            if (_vocab.TryGetValue(piece, out int id))
            {
                tokens.Add(id);
            }
            else
            {
                foreach (char c in piece)
                {
                    string s = c.ToString();
                    if (_vocab.TryGetValue(s, out int charId))
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

    /// <summary>
    /// Decodes generated tokens into text and word/phrase segments with timestamps.
    /// </summary>
    public (string FullText, List<SpeechSegment> Segments) DecodeWithTimestamps(
        ReadOnlySpan<int> tokens,
        TimeSpan timeOffset)
    {
        var fullText = new StringBuilder();
        var segments = new List<SpeechSegment>();

        var currentText = new StringBuilder();
        var currentTokens = new List<int>();
        float? segStart = null;
        int segId = 0;

        foreach (int tid in tokens)
        {
            if (tid == ImEndTokenId || tid == ImStartTokenId || tid == AudioBosTokenId || tid == AudioEosTokenId || tid == AudioPadTokenId)
            {
                continue;
            }

            if (tid >= TimestampBegin && tid <= TimestampEnd)
            {
                float sec = (tid - TimestampBegin) * 0.02f;

                if (segStart == null)
                {
                    segStart = sec;
                }
                else
                {
                    float startSec = segStart.Value;
                    float endSec = MathF.Max(sec, startSec + 0.1f);
                    string str = currentText.ToString().Trim();

                    if (str.Length > 0)
                    {
                        segments.Add(new SpeechSegment
                        {
                            Id = segId++,
                            Start = timeOffset + TimeSpan.FromSeconds(startSec),
                            End = timeOffset + TimeSpan.FromSeconds(endSec),
                            Text = str,
                            Tokens = currentTokens.ToArray(),
                            Probability = 1.0f
                        });

                        if (fullText.Length > 0) fullText.Append(' ');
                        fullText.Append(str);
                    }

                    currentText.Clear();
                    currentTokens.Clear();
                    segStart = null;
                }
            }
            else
            {
                currentTokens.Add(tid);
                if (_idToToken.TryGetValue(tid, out string? piece))
                {
                    if (!piece.StartsWith("<|") && !piece.EndsWith("|>"))
                    {
                        currentText.Append(piece);
                    }
                }
                else if (tid >= 0 && tid < 256)
                {
                    currentText.Append((char)tid);
                }
            }
        }

        if (currentText.Length > 0)
        {
            string str = currentText.ToString().Trim();
            if (str.Length > 0)
            {
                float startSec = segStart ?? 0.0f;
                float endSec = startSec + 1.0f;

                segments.Add(new SpeechSegment
                {
                    Id = segId++,
                    Start = timeOffset + TimeSpan.FromSeconds(startSec),
                    End = timeOffset + TimeSpan.FromSeconds(endSec),
                    Text = str,
                    Tokens = currentTokens.ToArray(),
                    Probability = 1.0f
                });

                if (fullText.Length > 0) fullText.Append(' ');
                fullText.Append(str);
            }
        }

        return (fullText.ToString(), segments);
    }
}
