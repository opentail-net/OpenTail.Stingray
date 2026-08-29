
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end proof for the Qwen3-ASR Safetensors pipeline
/// (<see cref="QwenAsrPipeline.LoadFromSafetensors"/>): mel extraction -&gt; real AuT audio
/// encoder -&gt; real audio-conditioned Qwen3 decode loop (via
/// <see cref="QwenAsrLlmSafetensorsTensorSource"/>/<see cref="QwenAsrDecoder.GenerateFromSafetensorsSource"/>)
/// -&gt; text. Same structural bar as the existing GGUF end-to-end test
/// (`QwenAsrRealWeightsTests`): synthetic tone input, asserts the pipeline runs to completion
/// and returns real, non-empty segments -- not a transcription-content check (no real speech
/// sample piped through this specific ASR yet).
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

        using var pipeline = QwenAsrPipeline.LoadFromSafetensors(checkpointDir!);
        Assert.Equal("Alibaba-Qwen3-ASR", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        int sampleRate = 16000;
        int durationSec = 2;
        var pcm = new float[sampleRate * durationSec];
        for (int i = 0; i < pcm.Length; i++)
        {
            float t = (float)i / sampleRate;
            pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * 300.0f * t)
                   + 0.2f * MathF.Sin(2.0f * MathF.PI * 600.0f * t)
                   + 0.1f * MathF.Sin(2.0f * MathF.PI * 1200.0f * t);
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = sampleRate,
            Language = "en",
            Task = SpeechTask.Transcribe
        };

        var result = pipeline.Transcribe(request);

        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.True(result.Duration.TotalSeconds >= 1.9);
        Assert.NotNull(result.Segments);
        Assert.NotEmpty(result.Segments);
    }
}
