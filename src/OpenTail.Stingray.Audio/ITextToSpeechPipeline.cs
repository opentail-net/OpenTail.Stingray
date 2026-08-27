using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

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

/// <summary>
/// Shared sentence-splitting logic for <see cref="ITextToSpeechPipeline.GenerateStreamAsync"/>
/// implementations that don't have a native token/frame streaming path: splits the request text
/// into clauses/sentences and calls the pipeline's synchronous <c>Generate</c> once per clause.
/// </summary>
public static class TtsStreamingHelper
{
    private static readonly Regex SentenceSplit = new(@"(?<=[.!?,;\n])\s+", RegexOptions.Compiled);

    public static async IAsyncEnumerable<float[]> SplitAndGenerateAsync(
        AudioGenerationRequest request,
        Func<AudioGenerationRequest, AudioGenerationResult> generate,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text)) yield break;

        var sentences = SentenceSplit.Split(request.Text);
        foreach (var s in sentences)
        {
            var trimmed = s.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            ct.ThrowIfCancellationRequested();

            var req = request with { Text = trimmed, OutputPath = null };
            var res = generate(req);
            if (res.Samples.Length > 0)
            {
                yield return res.Samples;
            }
            await Task.Yield();
        }
    }
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
