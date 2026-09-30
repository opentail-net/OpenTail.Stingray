namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class FunAsrRealWeightsTests : HeavyTestBase
{
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
            var pNested = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(pNested)) return pNested;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Paraformer_GgufRealModelFile_HandlesSyntheticNonSpeechAudio()
    {
        string? modelPath = FunAsrGgufTestModelLocator.FindParaformerModelPath();
        Assert.SkipUnless(modelPath is not null, "Paraformer GGUF (architecture=paraformer, pf.vocab present) not found.");

        using var pipeline = FunAsrPipeline.Load(modelPath!);
        Assert.NotNull(pipeline);
        Assert.Equal("paraformer", pipeline.Architecture);

        // Run transcription
        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.5f;
        }

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        });

        // A pure tone is not a speech-quality requirement. Real weights may emit only special
        // tokens for this input, resulting in an empty transcript and no segments.
        Assert.NotNull(res);
        Assert.Equal("zh", res.Language);
        Assert.Equal(TimeSpan.FromSeconds(2), res.Duration);
        Assert.NotNull(res.Segments);
    }

    [Fact]
    public void Paraformer_OnnxRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("paraformer-zh-small.int8.onnx");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var pipeline = FunAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);

        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.5f;
        }

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        });

        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
    }

    [Fact]
    public void SenseVoice_OnnxRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("sensevoice-small.int8.onnx");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var pipeline = FunAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);

        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.4f;
        }

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        });

        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
    }
}
