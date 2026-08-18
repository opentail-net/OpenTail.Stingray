namespace OpenTail.Stingray.Audio;

/// <summary>
/// Common interface for Speech-to-Text (STT / ASR) transcription and translation engines.
/// </summary>
public interface ISpeechToTextPipeline : IDisposable
{
    string Architecture { get; }
    int SampleRate { get; }

    /// <summary>
    /// Transcribes or translates audio samples into text with timestamps and segments.
    /// </summary>
    SpeechToTextResult Transcribe(SpeechToTextRequest request);
}

public enum SpeechTask
{
    Transcribe = 0,
    Translate = 1
}

public sealed record SpeechToTextRequest
{
    public required float[] AudioSamples { get; init; }
    public int SampleRate { get; init; } = 16000;
    public string? Language { get; init; }
    public SpeechTask Task { get; init; } = SpeechTask.Transcribe;
    public bool EnableTimestamps { get; init; } = true;
    public float Temperature { get; init; } = 0.0f;
    public int BeamSize { get; init; } = 1;
    public string? InitialPrompt { get; init; }
    public Action<int, int>? Progress { get; init; }
}

public sealed record SpeechSegment
{
    public int Id { get; init; }
    public TimeSpan Start { get; init; }
    public TimeSpan End { get; init; }
    public required string Text { get; init; }
    public int[] Tokens { get; init; } = [];
    public float Probability { get; init; } = 1.0f;
}

public sealed class SpeechToTextResult
{
    public string Text { get; }
    public string Language { get; }
    public TimeSpan Duration { get; }
    public IReadOnlyList<SpeechSegment> Segments { get; }

    public SpeechToTextResult(
        string text,
        string language,
        TimeSpan duration,
        IReadOnlyList<SpeechSegment> segments)
    {
        Text = text;
        Language = language;
        Duration = duration;
        Segments = segments;
    }
}
