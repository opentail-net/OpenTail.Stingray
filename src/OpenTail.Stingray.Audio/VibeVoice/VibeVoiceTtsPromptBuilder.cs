namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>One parsed "Speaker N: text" line from a real VibeVoice TTS script.</summary>
public readonly struct VibeVoicePromptLine(int speakerId, string text)
{
    public int SpeakerId { get; } = speakerId;
    public string Text { get; } = text;
}

/// <summary>
/// Real text prompt-template builder for VibeVoice TTS, ported from `tokenizer_text.cpp`'s
/// `parse_script`/`VibeVoiceTextTokenizer::encode_prompt` (not guessed) -- the real gap flagged
/// since this session's earlier `VibeVoiceGeneratorRealWeightsTests` entry ("Real prompt-template
/// construction... is separate, unstarted work this test deliberately bypasses").
///
/// <para>Real template (reference-audio-free case -- no `Voice input:` speaker-audio section;
/// see this class's doc comment on <see cref="BuildPrompt"/> for the real voice-cloning branch
/// this deliberately does not implement yet): real fixed system prompt -&gt; `" Text input:\n"`
/// -&gt; one `" Speaker {id}:{text}\n"` line per parsed script line (0-indexed speaker ids, real
/// `min_speaker` normalization: if the lowest speaker id in the script is &gt;0, EVERY id is
/// shifted down by 1) -&gt; `" Speech output:\n"` -&gt; the real `speech_start` token.</para>
///
/// <para>Real script parsing (`parse_script`): each line matched against
/// `^Speaker\s+(\d+)\s*:\s*(.*)$` (case-insensitive), trimmed; non-matching/blank lines are
/// silently dropped (not an error) -- only the empty-result case throws.</para>
/// </summary>
public static class VibeVoiceTtsPromptBuilder
{
    private const string SystemPrompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

    /// <summary>Real `parse_script`: splits on newlines, matches `Speaker N: text`, trims, and
    /// re-indexes speaker ids to start at 0 if the script's own lowest id is not already 0.</summary>
    public static List<VibeVoicePromptLine> ParseScript(string script)
    {
        var regex = new System.Text.RegularExpressions.Regex(@"^Speaker\s+(\d+)\s*:\s*(.*)$",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        var raw = new List<(int SpeakerId, string Text)>();
        foreach (var rawLine in script.Split('\n'))
        {
            string line = rawLine.Trim();
            if (line.Length == 0) continue;
            var match = regex.Match(line);
            if (!match.Success) continue;
            int speakerId = int.Parse(match.Groups[1].Value);
            string text = " " + match.Groups[2].Value.Trim();
            raw.Add((speakerId, text));
        }
        if (raw.Count == 0) throw new ArgumentException("VibeVoice prompt has no valid 'Speaker N:' lines.", nameof(script));

        int minSpeaker = raw.Min(l => l.SpeakerId);
        int shift = minSpeaker > 0 ? 1 : 0;
        return [.. raw.Select(l => new VibeVoicePromptLine(l.SpeakerId - shift, l.Text))];
    }

    /// <summary>
    /// Builds the real reference-audio-free prompt token sequence. `tokenize` is the caller's
    /// real BPE encoder (e.g. `GgufTokenizer.Encode`); `speechStartId` is the real
    /// `speech_start_id` (reused Qwen2 `&lt;|vision_start|&gt;`, confirmed real, not a new token).
    ///
    /// <para>Real, deliberate scope limit: this does NOT implement the reference's real
    /// `Voice input:` speaker-audio section (per-speaker real audio-prompt speech tokens spliced
    /// in before the text, needed for VOICE CLONING specifically) -- matches this session's
    /// established "reference-audio-free" scoping convention (VoxCPM2's zero-shot path, Qwen3
    /// Forced Aligner's audio-free... etc.) applied consistently across this pass; voice cloning
    /// remains real, separate, unstarted work.</para>
    /// </summary>
    public static int[] BuildPrompt(Func<string, int[]> tokenize, string script, int speechStartId)
    {
        var lines = ParseScript(script);
        var ids = new List<int>();
        ids.AddRange(tokenize(SystemPrompt));
        ids.AddRange(tokenize(" Text input:\n"));
        foreach (var line in lines)
            ids.AddRange(tokenize($" Speaker {line.SpeakerId}:{line.Text}\n"));
        ids.AddRange(tokenize(" Speech output:\n"));
        ids.Add(speechStartId);
        return [.. ids];
    }

    /// <summary>Real `speech_token_count`: `ceil(samples / compressionRatio)`.</summary>
    public static int SpeechTokenCount(int sampleCount24k, int compressionRatio)
    {
        if (compressionRatio <= 0) throw new ArgumentOutOfRangeException(nameof(compressionRatio));
        return (sampleCount24k + compressionRatio - 1) / compressionRatio;
    }

    /// <summary>
    /// Real voice-cloning prompt build, ported from `tokenizer_text.cpp`'s
    /// `VibeVoiceTextTokenizer::encode_prompt`'s `Voice input:` branch (not guessed) -- the gap
    /// <see cref="BuildPrompt"/>'s doc comment explicitly deferred. Real template: system prompt
    /// -&gt; `" Voice input:\n"` -&gt; per speaker (in the SAME order as `speakerSampleCounts24k`,
    /// which the caller must align with `ParseScript`'s own 0-indexed speaker ids): `" Speaker
    /// {index}:"` -&gt; `speechStartId` (mask=0) -&gt; `speechTokenCount` real placeholder
    /// `speechDiffusionId` tokens (mask=1, one per real compressed-audio frame) -&gt;
    /// `speechEndId` (mask=0) -&gt; `"\n"` -&gt; then the same real `" Text input:\n"`/per-line/
    /// `" Speech output:\n"`/`speechStartId` tail as the reference-audio-free path.
    /// </summary>
    public static (int[] Ids, bool[] SpeechMask, int[] SpeechTokenCounts) BuildPromptWithVoiceCloning(
        Func<string, int[]> tokenize, string script, int speechStartId, int speechEndId, int speechDiffusionId,
        int[] speakerSampleCounts24k, int compressionRatio)
    {
        var lines = ParseScript(script);
        var ids = new List<int>();
        var mask = new List<bool>();
        var tokenCounts = new int[speakerSampleCounts24k.Length];

        void Add(IEnumerable<int> tokens, bool speech)
        {
            foreach (int t in tokens) { ids.Add(t); mask.Add(speech); }
        }
        void AddOne(int token, bool speech) { ids.Add(token); mask.Add(speech); }

        Add(tokenize(SystemPrompt), false);

        if (speakerSampleCounts24k.Length > 0)
        {
            Add(tokenize(" Voice input:\n"), false);
            for (int index = 0; index < speakerSampleCounts24k.Length; index++)
            {
                Add(tokenize($" Speaker {index}:"), false);
                AddOne(speechStartId, false);
                int speechTokens = SpeechTokenCount(speakerSampleCounts24k[index], compressionRatio);
                tokenCounts[index] = speechTokens;
                for (int i = 0; i < speechTokens; i++) AddOne(speechDiffusionId, true);
                AddOne(speechEndId, false);
                Add(tokenize("\n"), false);
            }
        }

        Add(tokenize(" Text input:\n"), false);
        foreach (var line in lines)
            Add(tokenize($" Speaker {line.SpeakerId}:{line.Text}\n"), false);
        Add(tokenize(" Speech output:\n"), false);
        AddOne(speechStartId, false);

        return ([.. ids], [.. mask], tokenCounts);
    }
}
