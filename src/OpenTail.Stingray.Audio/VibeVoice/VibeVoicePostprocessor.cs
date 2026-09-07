namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>One transcribed speech segment (real fields: Start time/End time/Speaker ID/Content,
/// matching the real prompt's own instruction text this session scoped earlier).</summary>
public readonly struct VibeVoiceAsrSegment(double startTime, double endTime, string speakerId, string text)
{
    public double StartTime { get; } = startTime;
    public double EndTime { get; } = endTime;
    public string SpeakerId { get; } = speakerId;
    public string Text { get; } = text;
}

public readonly struct VibeVoiceAsrDecoded(string rawText, string text, IReadOnlyList<VibeVoiceAsrSegment> segments)
{
    public string RawText { get; } = rawText;
    public string Text { get; } = text;
    public IReadOnlyList<VibeVoiceAsrSegment> Segments { get; } = segments;
}

/// <summary>
/// Real output postprocessing for VibeVoice ASR, ported from `postprocess.cpp`'s
/// `VibeVoiceASRPostprocessor::decode` (not guessed): extracts the first balanced `[...]`/`{...}`
/// JSON payload out of the model's raw decoded text (the model is prompted to answer in JSON, but
/// real generations often wrap it in extra prose/markdown fences -- this scan finds the real
/// payload regardless), parses each object's `Start time`/`End time`/`Speaker ID`/`Content` keys
/// (real fallback key aliases `Start`/`End`/`Speaker`/`text` also accepted, matching the
/// reference's `optional_string_any_key` exactly), and falls back to the raw trimmed text when no
/// valid JSON payload parses (a real, deliberate degrade-gracefully behavior, not an error).
/// </summary>
public static class VibeVoicePostprocessor
{
    public static VibeVoiceAsrDecoded Decode(string rawText)
    {
        string payload = ExtractJsonPayload(rawText);
        var segments = new List<VibeVoiceAsrSegment>();

        if (payload.Length > 0)
        {
            try
            {
                using var doc = JsonDocument.Parse(payload);
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in doc.RootElement.EnumerateArray()) TryParseSegment(item, segments);
                }
                else if (doc.RootElement.ValueKind == JsonValueKind.Object)
                {
                    TryParseSegment(doc.RootElement, segments);
                }
            }
            catch (JsonException)
            {
                // Real behavior: an unparseable payload falls through to the raw-text fallback below.
            }
        }

        string text = string.Join(" ", segments.Select(s => s.Text));
        if (text.Length == 0) text = rawText.Trim();

        return new VibeVoiceAsrDecoded(rawText, text, segments);
    }

    private static void TryParseSegment(JsonElement item, List<VibeVoiceAsrSegment> segments)
    {
        if (item.ValueKind != JsonValueKind.Object) return;
        double start = OptionalF64AnyKey(item, "Start time", "Start", "start_time");
        double end = OptionalF64AnyKey(item, "End time", "End", "end_time");
        string speaker = OptionalStringAnyKey(item, "Speaker ID", "Speaker", "speaker_id");
        string text = OptionalStringAnyKey(item, "Content", "text").Trim();
        if (text.Length > 0) segments.Add(new VibeVoiceAsrSegment(start, end, speaker, text));
    }

    private static string OptionalStringAnyKey(JsonElement obj, params string[] keys)
    {
        foreach (string key in keys)
        {
            if (!obj.TryGetProperty(key, out var value)) continue;
            if (value.ValueKind == JsonValueKind.String) return value.GetString() ?? "";
            if (value.ValueKind == JsonValueKind.Number)
            {
                double number = value.GetDouble();
                return Math.Floor(number) == number ? ((long)number).ToString() : number.ToString();
            }
        }
        return "";
    }

    private static double OptionalF64AnyKey(JsonElement obj, params string[] keys)
    {
        string value = OptionalStringAnyKey(obj, keys);
        return value.Length == 0 ? 0.0 : double.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>Real balanced-bracket scan: finds the first `[`/`{` and returns the substring up
    /// to its matching close, or empty if unbalanced/absent.</summary>
    private static string ExtractJsonPayload(string text)
    {
        // Real reference behavior: prefers '[' if present ANYWHERE, even after '{' -- not the
        // earliest of the two (matches `find('[') != npos ? find('[') : find('{')` exactly).
        int bracketIdx = text.IndexOf('[');
        int start = bracketIdx >= 0 ? bracketIdx : text.IndexOf('{');
        if (start < 0) return "";

        int depth = 0;
        for (int i = start; i < text.Length; i++)
        {
            char c = text[i];
            if (c is '[' or '{') depth++;
            else if (c is ']' or '}')
            {
                depth--;
                if (depth == 0) return text[start..(i + 1)];
            }
        }
        return "";
    }
}
