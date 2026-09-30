
namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class MeloTtsRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "melotts-zh_en.onnx";

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            var pNested = Path.Combine(dir, "models", "_models", fileName);
            foreach (var candidate in new[] { p, pNested })
            {
                // A zero-byte placeholder in models/ must not hide a complete local checkpoint
                // in models/_models/.
                if (File.Exists(candidate) && new FileInfo(candidate).Length > 50L * 1024 * 1024)
                    return candidate;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void MeloTts_RealModelFile_OnnxHeaderValidAndSynthesizesSpeech()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        var fileInfo = new FileInfo(modelPath);
        Assert.True(fileInfo.Length > 50 * 1024 * 1024, "MeloTTS ONNX model file must be > 50MB");

        using var pipeline = MeloPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("MeloTTS", pipeline.Architecture);
        Assert.Equal(44100, pipeline.DefaultSampleRate);

        var request = new AudioGenerationRequest
        {
            Text = "MeloTTS high-speed multilingual text-to-speech synthesis.",
            Voice = "EN-US",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(44100, result.SampleRate);
        Assert.True(result.Samples.Length > 0);
        Assert.True(result.Duration.TotalSeconds > 0.5);

        for (int i = 0; i < result.Samples.Length; i++)
        {
            Assert.False(float.IsNaN(result.Samples[i]), $"NaN in MeloTTS sample {i}");
            Assert.False(float.IsInfinity(result.Samples[i]), $"Infinity in MeloTTS sample {i}");
        }
    }
}
