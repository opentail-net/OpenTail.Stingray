using System.Runtime.CompilerServices;

namespace OpenTail.Stingray.Audio.Wav2Vec2;

/// <summary>
/// <see cref="ISpeechToTextPipeline"/> over <see cref="Wav2Vec2CtcModel"/> (greedy CTC). Audio at another rate is
/// resampled to 16 kHz. Language/task/temperature/timestamps are not applicable to a CTC character model and are
/// ignored; the result is one segment covering the utterance.
/// </summary>
public sealed class Wav2Vec2CtcPipeline(Wav2Vec2CtcModel model) : ISpeechToTextPipeline
{
    public string Architecture => "Wav2Vec2ForCTC";
    public int SampleRate => 16000;

    public static Wav2Vec2CtcPipeline Load(string dir) => new(Wav2Vec2CtcModel.Load(dir));

    /// <summary>A checkpoint directory whose <c>config.json</c> declares <c>Wav2Vec2ForCTC</c>.</summary>
    public static bool IsWav2Vec2CtcDirectory(string path)
    {
        string cfg = Path.Combine(path, "config.json");
        return Directory.Exists(path) && File.Exists(cfg) && File.ReadAllText(cfg).Contains("\"Wav2Vec2ForCTC\"", StringComparison.Ordinal);
    }

    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        float[] audio = request.SampleRate == SampleRate
            ? request.AudioSamples
            : AudioResampler.Resample(request.AudioSamples, request.SampleRate, SampleRate);
        string text = model.Transcribe(audio);
        var duration = TimeSpan.FromSeconds(audio.Length / (double)SampleRate);
        return new SpeechToTextResult(text, "en", duration, [new SpeechSegment { Id = 0, Start = TimeSpan.Zero, End = duration, Text = text }]);
    }

    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        await foreach (var chunk in audioStream.WithCancellation(ct))
            buffer.AddRange(chunk.ToArray());
        if (buffer.Count == 0) yield break;
        foreach (var segment in Transcribe(baseRequest with { AudioSamples = [.. buffer] }).Segments) yield return segment;
    }

    public void Dispose() => model.Dispose();
}
