using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end proof for the Whisper Safetensors loader
/// (<see cref="WhisperGgmlModel.LoadFromSafetensors"/>): the real, canonical
/// `openai/whisper-tiny` Hugging Face checkpoint (`config.json`/`model.safetensors`/
/// `vocab.json`, confirmed to be `transformers`' actual native distribution, not a community
/// conversion) transcribes the same real ground-truth JFK speech sample correctly. Same
/// ground-truth assertions as the ggml/GGUF loader tests.
/// </summary>
public sealed class WhisperSafetensorsTests : HeavyTestBase
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
    public void WhisperPipeline_LoadFromSafetensors_TinyModel_TranscribesJfkSampleCorrectly()
    {
        string? checkpointDir = FindRepoFile("models/whisper-tiny-hf");
        string? wavPath = FindRepoFile("examples/whisper.cpp/samples/jfk.wav");
        Assert.SkipUnless(checkpointDir != null, "models/whisper-tiny-hf not found");
        Assert.SkipUnless(wavPath != null, "examples/whisper.cpp/samples/jfk.wav not found");

        using var pipeline = WhisperPipeline.LoadFromSafetensors(checkpointDir!);
        var (samples, sampleRate, _) = WavReader.ReadWav(wavPath!);

        var result = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = sampleRate,
            Language = "en",
            Task = SpeechTask.Transcribe,
            EnableTimestamps = true
        });

        string text = result.Text.ToLowerInvariant();
        Assert.Contains("fellow americans", text);
        Assert.Contains("ask not what your country can do for you", text);
        Assert.Contains("ask what you can do for your country", text);
    }
}
