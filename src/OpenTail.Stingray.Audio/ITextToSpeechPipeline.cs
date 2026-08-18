namespace OpenTail.Stingray.Audio;

/// <summary>
/// Common interface for Text-to-Speech (TTS) synthesis engines.
/// </summary>
public interface ITextToSpeechPipeline : IDisposable
{
    string Architecture { get; }
    int DefaultSampleRate { get; }

    /// <summary>
    /// Synthesizes text to audio samples at native sample rate.
    /// </summary>
    AudioGenerationResult Generate(AudioGenerationRequest request);

    /// <summary>
    /// Synthesizes text in streaming fashion, yielding clause/sentence audio waveforms as they are generated.
    /// </summary>
    IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, CancellationToken ct = default);
}

public record AudioGenerationRequest
{
    public required string Text { get; init; }
    public string Voice { get; init; } = "af_heart";
    public float Speed { get; init; } = 1.0f;
    public string? OutputPath { get; init; }
    public string? ReferenceAudioPath { get; init; }
    public string? ReferenceText { get; init; }
    public Action<int, int>? Progress { get; init; }
}

public sealed class AudioGenerationResult
{
    public float[] Samples { get; }
    public int SampleRate { get; }
    public TimeSpan Duration => TimeSpan.FromSeconds((double)Samples.Length / SampleRate);

    public AudioGenerationResult(float[] samples, int sampleRate = 24000)
    {
        Samples = samples;
        SampleRate = sampleRate;
    }

    public void SaveWav(string path)
    {
        WavWriter.WriteWav(path, Samples, SampleRate);
    }

    public byte[] ToWavBytes()
    {
        return WavWriter.ToWavBytes(Samples, SampleRate);
    }
}
