
namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class PiperRealWeightsTests : HeavyTestBase
{
    private const string ConfigFileName = "en_US-lessac-medium.onnx.json";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void Piper_RealConfigFile_LoadsAndSynthesizesSpeech()
    {
        string? configPath = FindModelPath(ConfigFileName);
        if (configPath is null) Assert.Skip("configPath not found (checkpoint or fixture missing).");

        var pipeline = PiperPipeline.FromConfigFile(configPath);
        Assert.NotNull(pipeline);
        Assert.Equal(22050, pipeline.DefaultSampleRate);

        var request = new AudioGenerationRequest
        {
            Text = "Hello world! Piper text-to-speech synthesis with real voice configuration.",
            Voice = "lessac",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);

        Assert.NotNull(result);
        Assert.Equal(22050, result.SampleRate);
        Assert.True(result.Samples.Length > 0);
        Assert.True(result.Duration.TotalSeconds > 0.5);

        for (int i = 0; i < result.Samples.Length; i++)
        {
            Assert.False(float.IsNaN(result.Samples[i]), $"NaN in Piper sample {i}");
            Assert.False(float.IsInfinity(result.Samples[i]), $"Infinity in Piper sample {i}");
        }
    }
}
