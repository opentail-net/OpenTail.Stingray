using System.Text.Json;

namespace OpenTail.Stingray.Audio.NeuTts;

/// <summary>One built NeuTTS prompt: real token ids ready for prefill, plus the special token ids
/// the AR generation loop needs to interpret its own output (real speech-token offset range and
/// the real stop token).</summary>
public readonly struct NeuTtsPrompt(int[] tokenIds, int speechTokenStart, int speechTokenEnd, int speechGenerationEnd)
{
    public int[] TokenIds { get; } = tokenIds;
    public int SpeechTokenStart { get; } = speechTokenStart;
    public int SpeechTokenEnd { get; } = speechTokenEnd;
    public int SpeechGenerationEnd { get; } = speechGenerationEnd;
}

/// <summary>
/// Real prompt builder for NeuTTS, ported from `examples/audio.cpp/src/models/neutts/prompt.cpp`'s
/// `NeuTTSPromptBuilder::build` (not guessed). Real, notably simple multimodal mechanism: unlike
/// every other TTS model ported this session (VoxCPM2/MOSS-TTS-Nano/VibeVoice/PersonaPlex/Higgs),
/// NeuTTS's discrete speech-codec tokens are NOT spliced in via a separate embedding table or
/// synthetic vocab extension -- they are genuinely native vocabulary ids in the checkpoint's own
/// tokenizer (`&lt;|speech_0|&gt;` .. `&lt;|speech_65535|&gt;`, a real contiguous 65536-token
/// range), so a codec code simply becomes token id `speechTokenStart + code`. No bridging trick
/// needed at all -- plain `ForwardPass` prefill/decode over the combined text+speech token
/// sequence handles everything.
///
/// Real prompt layout: `&lt;|TEXT_PROMPT_START|&gt; [reference_text] [EMOTION_TOKEN if not
/// neutral] [input_text] &lt;|TEXT_PROMPT_END|&gt; &lt;|SPEECH_GENERATION_START|&gt;
/// [reference_speaker's real pre-baked speech_codes, offset into speechTokenStart..]` -- generation
/// then continues autoregressively from there until `&lt;|SPEECH_GENERATION_END|&gt;`.
/// </summary>
public static class NeuTtsPromptBuilder
{
    private const string TextPromptStart = "<|TEXT_PROMPT_START|>";
    private const string TextPromptEnd = "<|TEXT_PROMPT_END|>";
    private const string SpeechGenerationStart = "<|SPEECH_GENERATION_START|>";
    private const string SpeechGenerationEnd = "<|SPEECH_GENERATION_END|>";
    private const string SpeechTokenStartPiece = "<|speech_0|>";
    private const string SpeechTokenEndPiece = "<|speech_65535|>";

    /// <summary>
    /// Real, direct `tokenizer.json`-`added_tokens` reader, used ONLY for control/speech tokens
    /// this class needs. Real, verified limitation of this project's general
    /// `HuggingFaceTokenizerSource`/`GgufTokenizer` loading pipeline: it does not surface every
    /// `added_tokens` entry into `TokenizerSource.AdditionalSpecialTokens`/`.Tokens` for a
    /// checkpoint this size -- confirmed directly (this session) that NeuTTS's real 65536-entry
    /// `&lt;|speech_N|&gt;` range genuinely exists in the checkpoint's own `tokenizer.json`
    /// `added_tokens` array (`grep -c speech_` = 65536), with real, confirmed ids `speech_0=151684`
    /// .. `speech_65535=217219` -- but neither ends up reachable through the standard loader. Rather
    /// than special-case the shared tokenizer infra for one checkpoint, this reads the same real
    /// `added_tokens` array directly, which is small work (one JSON scan, done once) and needs no
    /// changes to already-verified shared code.
    /// </summary>
    public static Dictionary<string, int> LoadAddedTokenIds(byte[] tokenizerJsonBytes)
    {
        using var doc = JsonDocument.Parse(tokenizerJsonBytes);
        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var t in doc.RootElement.GetProperty("added_tokens").EnumerateArray())
            result[t.GetProperty("content").GetString()!] = t.GetProperty("id").GetInt32();
        return result;
    }

    private static int RequireTokenId(IReadOnlyDictionary<string, int> addedTokens, TokenizerSource source, string token)
    {
        if (addedTokens.TryGetValue(token, out int id)) return id;
        if (source.AdditionalSpecialTokens.TryGetValue(token, out id)) return id;
        throw new InvalidDataException($"NeuTTS tokenizer missing token: {token}");
    }

    /// <summary>Real text normalization, ported verbatim from `prompt.cpp`'s
    /// `normalize_neutts_text`: curly quotes -&gt; straight, ideographic space -&gt; ASCII space.</summary>
    public static string NormalizeText(string text) => text
        .Replace('‘', '\'').Replace('’', '\'')
        .Replace('“', '"').Replace('”', '"')
        .Replace('　', ' ');

    public static NeuTtsPrompt Build(
        GgufTokenizer tokenizer, TokenizerSource source, IReadOnlyDictionary<string, int> addedTokens,
        string referenceText, string inputText, int[] speakerSpeechCodes,
        string emotion = "neutral", string[]? supportedEmotions = null)
    {
        int textPromptStart = RequireTokenId(addedTokens, source, TextPromptStart);
        int textPromptEnd = RequireTokenId(addedTokens, source, TextPromptEnd);
        int speechGenerationStart = RequireTokenId(addedTokens, source, SpeechGenerationStart);
        int speechGenerationEnd = RequireTokenId(addedTokens, source, SpeechGenerationEnd);
        int speechTokenStart = RequireTokenId(addedTokens, source, SpeechTokenStartPiece);
        int speechTokenEnd = RequireTokenId(addedTokens, source, SpeechTokenEndPiece);

        string resolvedEmotion = string.IsNullOrEmpty(emotion) ? "neutral" : emotion;
        if (supportedEmotions is not null && Array.IndexOf(supportedEmotions, resolvedEmotion) < 0)
            throw new ArgumentException($"unsupported NeuTTS emotion: {resolvedEmotion}", nameof(emotion));
        bool hasEmotionToken = resolvedEmotion != "neutral";

        string normalizedReference = NormalizeText(referenceText);
        string normalizedInput = NormalizeText(inputText);

        var textIds = new List<int>();
        if (hasEmotionToken)
        {
            textIds.AddRange(tokenizer.Encode(normalizedReference));
            string emotionToken = $"<|{resolvedEmotion.ToUpperInvariant()}|>";
            textIds.Add(RequireTokenId(addedTokens, source, emotionToken));
            textIds.AddRange(tokenizer.Encode(normalizedInput));
        }
        else
        {
            textIds.AddRange(tokenizer.Encode($"{normalizedReference} {normalizedInput}"));
        }

        var tokenIds = new List<int>(textIds.Count + speakerSpeechCodes.Length + 3)
        {
            textPromptStart,
        };
        tokenIds.AddRange(textIds);
        tokenIds.Add(textPromptEnd);
        tokenIds.Add(speechGenerationStart);
        foreach (int code in speakerSpeechCodes)
        {
            if (code < 0 || speechTokenStart + code > speechTokenEnd)
                throw new ArgumentException($"speaker prompt code {code} out of range.", nameof(speakerSpeechCodes));
            tokenIds.Add(speechTokenStart + code);
        }

        return new NeuTtsPrompt([.. tokenIds], speechTokenStart, speechTokenEnd, speechGenerationEnd);
    }
}
