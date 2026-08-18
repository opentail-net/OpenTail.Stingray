namespace OpenTail.Stingray.Audio.Vad;

/// <summary>
/// Voice Activity Detection (VAD) interface for streaming and batch speech presence detection.
/// </summary>
public interface IVoiceActivityDetector : IDisposable
{
    /// <summary>
    /// Evaluates a 512-sample (31.25ms at 16kHz) audio frame and returns speech probability in [0.0, 1.0].
    /// </summary>
    float ProcessFrame(ReadOnlySpan<float> frame512);

    /// <summary>
    /// Resets the internal recurrent LSTM states (h_state, c_state) for a new audio stream.
    /// </summary>
    void Reset();

    /// <summary>
    /// Scans an entire audio waveform and returns speech timestamp segments.
    /// </summary>
    IReadOnlyList<VadSpeechSegment> DetectSegments(ReadOnlySpan<float> audio, VadParams? parameters = null);
}

public sealed record VadParams
{
    /// <summary>Speech probability threshold to trigger speech start/continuation. Default: 0.5.</summary>
    public float Threshold { get; init; } = 0.5f;

    /// <summary>Minimum duration of a speech segment in milliseconds. Default: 250ms.</summary>
    public int MinSpeechDurationMs { get; init; } = 250;

    /// <summary>Minimum silence duration to mark the end of a speech segment in milliseconds. Default: 100ms.</summary>
    public int MinSilenceDurationMs { get; init; } = 100;

    /// <summary>Padding added to the start and end of speech segments in milliseconds. Default: 30ms.</summary>
    public int SpeechPadMs { get; init; } = 30;

    /// <summary>Audio sample rate (16000 Hz standard). Default: 16000.</summary>
    public int SampleRate { get; init; } = 16000;
}

public sealed record VadSpeechSegment
{
    public int StartSample { get; init; }
    public int EndSample { get; init; }
    public float StartSeconds { get; init; }
    public float EndSeconds { get; init; }
    public float AvgProbability { get; init; }
}
