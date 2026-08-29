
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
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Paraformer_GgufRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("paraformer-q8.gguf");
        if (modelPath is null) return;

        using var pipeline = FunAsrPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("Alibaba-FunASR-Nano", pipeline.Architecture);

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

        // NOTE: this test's audio is a pure 440Hz sine tone, not real speech. The real
        // Paraformer model (see docs/audio-review-progress.md's FunASR section -- all four
        // stages now wired to real, golden-verified weights) can legitimately predict only
        // special tokens (<blank>/<s>/</s>/<unk>, all stripped by FunAsrTokenizer.Decode) for
        // non-speech audio, producing an empty transcript with a real (non-crashing, non-empty
        // Segments) result -- this is plausible real-model behavior, not a bug, unlike the old
        // fake pipeline which guaranteed non-empty placeholder text regardless of audio content.
        // Only assert the pipeline runs end-to-end without crashing and returns a structurally
        // valid result; do not assert non-empty text for non-speech input.
        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
    }

    [Fact]
    public void Paraformer_OnnxRealModelFile_LoadsAndTranscribes()
    {
        string? modelPath = FindModelPath("paraformer-zh-small.int8.onnx");
        if (modelPath is null) return;

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
        if (modelPath is null) return;

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
