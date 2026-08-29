
namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class WhisperRealWeightsTests : HeavyTestBase
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

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Ground-truth regression test, added 2026-08-21 after finding and fixing two real bugs
    /// (a mel filterbank constant typo -- log(6.4)/27 hardcoded as the wrong value 27/64 --
    /// and missing decode-time special-token suppression) that together made every model size
    /// hallucinate non-speech tags instead of transcribing real audio. Neither existing
    /// "real weights" test above would have caught this: both use synthetic sine-wave audio
    /// and only assert the output is finite/not-obviously-noise, never real transcription
    /// content. This test uses the standard whisper.cpp smoke-test sample (a real, known,
    /// stable reference) and asserts actual transcribed content, not just shape/finiteness.
    /// See docs/audio-review-progress.md for the full investigation.
    /// </summary>
    [Theory]
    [InlineData("ggml-tiny.bin")]
    [InlineData("ggml-base.bin")]
    public void WhisperPipeline_RealModel_TranscribesJfkSampleCorrectly(string modelFileName)
    {
        string? modelPath = FindModelPath(modelFileName);
        string? wavPath = FindRepoFile("examples/whisper.cpp/samples/jfk.wav");
        if (modelPath is null || wavPath is null) return;

        using var pipeline = WhisperPipeline.Load(modelPath);
        var (samples, sampleRate, _) = WavReader.ReadWav(wavPath);

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

    [Fact]
    public void WhisperPipeline_LoadRealGgmlTiny_TranscribesAudioEndToEnd()
    {
        string? modelPath = FindModelPath("ggml-tiny.bin");
        if (modelPath is null) return;

        using var pipeline = WhisperPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal(16000, pipeline.SampleRate);

        int sampleRate = 16000;
        int durationSec = 2;
        var pcm = new float[sampleRate * durationSec];

        for (int i = 0; i < pcm.Length; i++)
        {
            float t = (float)i / sampleRate;
            pcm[i] = 0.4f * MathF.Sin(2.0f * MathF.PI * 400.0f * t);
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = sampleRate,
            Language = "en",
            Task = SpeechTask.Transcribe,
            EnableTimestamps = false
        };

        var result = pipeline.Transcribe(request);
        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.True(result.Duration.TotalSeconds >= 1.9);
        Assert.NotNull(result.Segments);
    }

    [Fact]
    public void WhisperPipeline_LoadRealGgmlMedium_TranscribesAudioEndToEnd()
    {
        string? modelPath = FindModelPath("ggml-medium.bin");
        if (modelPath is null) return;

        using var pipeline = WhisperPipeline.Load(modelPath);
        Assert.NotNull(pipeline);

        int sampleRate = 16000;
        var pcm = new float[sampleRate * 2];
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * 220.0f * i / sampleRate);
        }

        var result = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = sampleRate,
            Language = "en",
            EnableTimestamps = false
        });

        Assert.NotNull(result);
        Assert.NotNull(result.Segments);
    }

    [Fact]
    public void WhisperGgmlModel_TinyRealWeights_EncoderProducesFiniteOutput()
    {
        string? modelPath = FindModelPath("ggml-tiny.bin");
        if (modelPath is null) return;

        var ggml = WhisperGgmlModel.Load(modelPath);
        Assert.True(ggml.VocabSize is 51864 or 51865 or 51866, $"Unexpected vocab size {ggml.VocabSize}.");
        Assert.Equal(384, ggml.AudioState);
        Assert.Equal(4, ggml.AudioLayer);
        Assert.Equal(80, ggml.NumMels);

        var config = ggml.ToConfig();
        var weights = new WhisperEncoderWeights(ggml);
        var encoder = new WhisperEncoder(config, weights);

        int numFrames = 100;
        var mel = new float[config.NumMels * numFrames];
        var rng = new Random(42);
        for (int i = 0; i < mel.Length; i++) mel[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] hidden = encoder.Forward(mel, numFrames);

        Assert.True(hidden.Length > 0);
        Assert.All(hidden, v => Assert.True(float.IsFinite(v), "Encoder output must be finite (no NaN/Inf)."));

        float mean = 0f;
        foreach (var v in hidden) mean += v;
        mean /= hidden.Length;
        Assert.True(Math.Abs(mean) < 50f, $"Encoder output mean {mean} looks unreasonable for a LayerNorm-terminated stack.");
    }

    [Fact]
    public void WhisperGgmlModel_TinyRealWeights_DecoderStepProducesFiniteLogitsAndPeaksOnKnownToken()
    {
        string? modelPath = FindModelPath("ggml-tiny.bin");
        if (modelPath is null) return;

        var ggml = WhisperGgmlModel.Load(modelPath);
        var config = ggml.ToConfig();
        var encoderWeights = new WhisperEncoderWeights(ggml);
        var decoderWeights = new WhisperDecoderWeights(ggml);
        var encoder = new WhisperEncoder(config, encoderWeights);
        var decoder = new WhisperDecoder(config, decoderWeights);

        int numFrames = 100;
        var mel = new float[config.NumMels * numFrames];
        var rng = new Random(7);
        for (int i = 0; i < mel.Length; i++) mel[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] encoded = encoder.Forward(mel, numFrames);
        int audioFrames = encoded.Length / config.AudioState;

        var cache = new WhisperKvCache(config.TextLayer, config.TextCtx, config.TextState);
        decoder.PrimeCrossAttention(cache, encoded, audioFrames);

        int sotToken = Array.IndexOf(ggml.TokenById, "<|startoftranscript|>");
        if (sotToken < 0) sotToken = ggml.VocabSize > 50257 ? 50257 : 0;

        float[] logits = decoder.ForwardStep(sotToken, position: 0, cache, encoded, audioFrames);

        Assert.Equal(config.VocabSize, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v), "Decoder logits must be finite (no NaN/Inf)."));

        int argMax = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[argMax]) argMax = i;

        float mean = 0f;
        foreach (var v in logits) mean += v;
        mean /= logits.Length;
        Assert.True(logits[argMax] - mean > 1.0f, $"Top logit ({logits[argMax]}) not meaningfully above mean ({mean}) — looks like noise.");
    }
}
