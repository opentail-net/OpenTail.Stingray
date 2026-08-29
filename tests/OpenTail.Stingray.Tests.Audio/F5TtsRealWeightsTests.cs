
namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class F5TtsRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "f5tts_base.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void F5Tts_RealModelFile_SafetensorsValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "F5-TTS safetensors must contain tensors");
    }

    [Fact]
    public void F5TtsPipeline_LoadRealSafetensors_SynthesizesAudio()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var pipeline = F5TtsPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("F5-TTS", pipeline.Architecture);
        Assert.Equal(24000, pipeline.DefaultSampleRate);

        var request = new AudioGenerationRequest
        {
            Text = "Hello world! Flow matching text to speech.",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.True(result.Samples.Length > 0);
        Assert.True(result.Duration.TotalSeconds > 0.5);

        for (int i = 0; i < result.Samples.Length; i++)
        {
            Assert.False(float.IsNaN(result.Samples[i]), $"NaN in F5-TTS sample {i}");
            Assert.False(float.IsInfinity(result.Samples[i]), $"Infinity in F5-TTS sample {i}");
        }
    }
}
