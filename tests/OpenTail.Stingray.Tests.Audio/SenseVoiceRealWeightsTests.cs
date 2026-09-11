using OpenTail.Stingray.Audio.SenseVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt exercise of the newly-ported SenseVoice-Small CTC ASR pipeline
/// (<see cref="SenseVoicePipeline"/>) against the real local checkpoint
/// (`F:\_models\sensevoice-small.int8.onnx`, resolved via the same absolute-path convention as
/// this session's other "fetched-outside-the-repo" sidecar files) and its real tokens.txt
/// (`F:\_models\sensevoice-small-tokens.txt`, downloaded from
/// `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` on Hugging Face -- confirmed
/// vocab_size=25055 matching the checkpoint's own real ONNX metadata), on the same real
/// LibriSpeech clip used throughout this session's other ASR ports (see
/// `CitrinetAsrRealWeightsTests`/`MarbleNetVadRealWeightsTests`).
/// </summary>
public sealed class SenseVoiceRealWeightsTests
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Transcribe_OnRealLibriSpeechClip_ProducesNonEmptyText()
    {
        const string onnxPath = @"F:\_models\sensevoice-small.int8.onnx";
        const string tokensPath = @"F:\_models\sensevoice-small-tokens.txt";
        Assert.SkipUnless(File.Exists(onnxPath), $"{onnxPath} not found");
        Assert.SkipUnless(File.Exists(tokensPath), $"{tokensPath} not found");

        string? wavPath = FindRepoFile("examples/audio.cpp/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
        Assert.SkipUnless(wavPath != null, "librispeech reference clip not found");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var pipeline = SenseVoicePipeline.TryLoad(onnxPath, tokensPath);
        Assert.NotNull(pipeline);

        var (rawSamples, rawRate, rawChannels) = WavReader.ReadWav(wavPath!);
        Assert.Equal(1, rawChannels);
        var waveform = rawRate == 16000
            ? rawSamples
            : AudioResampler.Resample(rawSamples, rawRate, 16000, channels: 1, ResampleQuality.BestQuality);

        var result = pipeline!.Transcribe(waveform);
        sw.Stop();

        Console.WriteLine($"[SenseVoice] {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"[SenseVoice] lang={result.Language} emotion={result.Emotion} event={result.Event}");
        Console.WriteLine($"[SenseVoice] transcript='{result.Text}'");

        Assert.False(string.IsNullOrWhiteSpace(result.Text));
    }
}
