using OpenTail.Stingray.Audio.Parakeet;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class ParakeetRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "parakeet-ctc-0.6b-q4_k.gguf";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void Parakeet_RealModelFile_GgufMetadataAndTensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Parakeet GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Parakeet GGUF must contain metadata");
    }

    [Fact]
    public void ParakeetPipeline_LoadRealGguf_TranscribesAudioEndToEnd()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var pipeline = ParakeetPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("NVIDIA-NeMo-Parakeet-ASR", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        int sampleRate = 16000;
        int durationSec = 2;
        var pcm = new float[sampleRate * durationSec];

        // Synthesize a speech-frequency carrier tone + harmonics
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
    }
}
