
namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class QwenAsrRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "qwen3-asr-0.6b-q4_k.gguf";

    private static string? FindModelPath(string modelFile) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(modelFile);

    [Fact]
    public void QwenAsr_RealWeights_GgufHeaderAndTensorMetadata_Valid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);

        Assert.True(model.Header.Version >= 2, "Expected GGUF version >= 2");
        Assert.True(model.Tensors.Count > 0, "GGUF model must contain tensors");
        Assert.True(model.Metadata.Count > 0, "GGUF model must contain metadata");
    }

    [Fact]
    public void QwenAsrPipeline_LoadRealGguf_TranscribesAudioEndToEnd()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var pipeline = QwenAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("Alibaba-Qwen3-ASR", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        int sampleRate = 16000;
        int durationSec = 2;
        var pcm = new float[sampleRate * durationSec];

        // A synthetic tone exercises the end-to-end path but is not speech, so an empty
        // transcript and segment list are valid model outputs.
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
        Assert.NotNull(result.Text);
    }
}
