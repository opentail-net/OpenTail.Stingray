namespace OpenTail.Stingray.Vision;

/// <summary>One part of a video prompt. <see cref="Text"/> null means a frame at <see cref="FrameIndex"/>.</summary>
public readonly record struct VideoPromptPart(int FrameIndex, string? Text)
{
    public bool IsFrame => Text is null;
}

/// <summary>
/// The prompt layout of a video as the reference builds it (tools/mtmd/mtmd-helper.cpp <c>mtmd_helper_video_read_next</c>,
/// tools/mtmd/mtmd.cpp <c>expand_lazy_bitmaps</c> and <c>mtmd_group_mergeable_bitmaps</c>, pinned bed0a8566):
/// "Video:", then every frame, with a "[%dm%.2fs]" text after the frame that reaches the next timestamp threshold.
/// Two frames merge into one temporal chunk only when they are adjacent parts, so a timestamp between two frames
/// leaves each of them alone.
/// </summary>
public static class VideoPromptLayout
{
    public const string StartText = "Video:";

    public static List<VideoPromptPart> Parts(int frameCount, float fps, int timestampIntervalMs = 5000)
    {
        var parts = new List<VideoPromptPart> { new(-1, StartText) };
        float next = 0f;
        for (int k = 0; k < frameCount; k++)
        {
            parts.Add(new(k, null));
            if (timestampIntervalMs <= 0) continue;
            float elapsedMs = (float)k / fps * 1000f;
            if (elapsedMs < next) continue;
            parts.Add(new(-1, Timestamp(elapsedMs)));
            next += timestampIntervalMs;
        }
        return parts;
    }

    /// <summary>Text the reference writes for a frame time: minutes, then seconds with two decimals.</summary>
    public static string Timestamp(float elapsedMs)
    {
        float elapsedS = elapsedMs / 1000f;
        int minutes = (int)(elapsedS / 60f);
        float seconds = elapsedS - minutes * 60f;
        return FormattableString.Invariant($"[{minutes}m{(double)seconds:F2}s]");
    }

    /// <summary>Temporal chunks in reference order: each chunk is one or two frame indices.</summary>
    public static List<int[]> Chunks(IReadOnlyList<VideoPromptPart> parts) =>
        Segments(parts).Where(s => s.Frames is not null).Select(s => s.Frames!).ToList();

    /// <summary>The media in prompt order: text parts as they are, and each temporal chunk.</summary>
    public static List<VideoPromptSegment> Segments(IReadOnlyList<VideoPromptPart> parts)
    {
        var segments = new List<VideoPromptSegment>();
        for (int i = 0; i < parts.Count; i++)
        {
            if (!parts[i].IsFrame)
            {
                segments.Add(new(parts[i].Text, null));
            }
            else if (i + 1 < parts.Count && parts[i + 1].IsFrame)
            {
                segments.Add(new(null, [parts[i].FrameIndex, parts[i + 1].FrameIndex]));
                i++;
            }
            else
            {
                segments.Add(new(null, [parts[i].FrameIndex]));
            }
        }
        return segments;
    }
}

/// <summary>A text part (<see cref="Text"/>) or a temporal chunk of one or two frames (<see cref="Frames"/>).</summary>
public readonly record struct VideoPromptSegment(string? Text, int[]? Frames);
