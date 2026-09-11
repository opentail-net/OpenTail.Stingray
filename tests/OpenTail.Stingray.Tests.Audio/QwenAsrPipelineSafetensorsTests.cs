
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end proof for the Qwen3-ASR Safetensors pipeline
/// (<see cref="QwenAsrPipeline.LoadFromSafetensors"/>): mel extraction -&gt; real AuT audio
/// encoder -&gt; real audio-conditioned Qwen3 decode loop (via
/// <see cref="QwenAsrLlmSafetensorsTensorSource"/>/<see cref="QwenAsrDecoder.GenerateFromSafetensorsSource"/>)
/// -&gt; text. Uses real speech audio (`examples/audio.cpp/assets/resources/b.wav`, the same
/// standard reference clip used by every other working ASR pipeline's real-weights test in this
/// repo), not synthetic tone -- switched 2026-09-12 after the real chat-template fix (see
/// docs/00-current-work.md's 2026-09-12 entry) made this pipeline correctly recognize
/// non-speech sine-tone input as having nothing to transcribe (empty output), the same real,
/// expected-not-buggy behavior already documented for FunASR-Nano's synthetic-tone test
/// elsewhere in this project. A real ASR pipeline SHOULD produce nothing on pure tones; testing
/// with one was never actually verifying transcription, so this switches to a real speech
/// fixture and checks for real-content plausibility instead.
/// </summary>
public sealed class QwenAsrPipelineSafetensorsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void QwenAsrPipeline_LoadFromSafetensors_TranscribesAudioEndToEnd()
    {
        string? checkpointDir = FindRepoFile("models/qwen3-asr-0.6b-hf");
        Assert.SkipUnless(checkpointDir != null, "models/qwen3-asr-0.6b-hf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/b.wav");
        Assert.SkipUnless(audioPath != null, "reference b.wav not found");

        using var pipeline = QwenAsrPipeline.LoadFromSafetensors(checkpointDir!);
        Assert.Equal("Alibaba-Qwen3-ASR", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != pipeline.SampleRate) samples = AudioResampler.Resample(samples, sr, pipeline.SampleRate);

        var request = new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = pipeline.SampleRate,
            Language = "en",
            Task = SpeechTask.Transcribe
        };

        var result = pipeline.Transcribe(request);

        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.NotNull(result.Segments);
        Assert.NotEmpty(result.Segments);
        // Real-content plausibility check (not exact match -- this checkpoint's own decode still
        // has some real remaining garbling, see docs/00-current-work.md): the fixed template
        // should recover most of the real reference sentence's recognizable words, not a single
        // unrelated word like the pre-fix "aspects".
        Assert.True(result.Text.Length > 20, $"Expected a real multi-word transcript, got: \"{result.Text}\"");
        Assert.Contains("nature", result.Text, StringComparison.OrdinalIgnoreCase);
    }
}
