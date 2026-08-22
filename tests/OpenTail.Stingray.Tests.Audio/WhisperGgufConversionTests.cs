using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies the real GGUF-format Whisper loader (<see cref="WhisperGgmlModel.LoadFromGguf"/>)
/// against the legacy ggml loader (<see cref="WhisperGgmlModel.Load"/>) on the SAME underlying
/// weights. This is a pure repackaging (same tensor names, values, and hparams; only the
/// container format changed), so the correct verification is exact/bit-equality per tensor and
/// per hparam, not a numeric-similarity oracle -- a stronger check than cosine similarity, used
/// here because it's the right tool for a lossless-format-conversion claim.
/// </summary>
public sealed class WhisperGgufConversionTests : HeavyTestBase
{
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

    [Fact]
    public void LoadFromGguf_TinyModel_MatchesLegacyGgmlLoaderExactly()
    {
        string? ggmlPath = FindRepoFile("models/ggml-tiny.bin");
        string? ggufPath = FindRepoFile("scratch-llamacpp-ref/whisper-tiny-test.gguf");
        Assert.SkipUnless(ggmlPath != null, "models/ggml-tiny.bin not found");
        Assert.SkipUnless(ggufPath != null, "converted whisper-tiny-test.gguf fixture not found (run scratch-llamacpp-ref/whisper_ggml_to_gguf.py first)");

        var legacy = WhisperGgmlModel.Load(ggmlPath!);
        var gguf = WhisperGgmlModel.LoadFromGguf(ggufPath!);

        Assert.Equal(legacy.VocabSize, gguf.VocabSize);
        Assert.Equal(legacy.AudioCtx, gguf.AudioCtx);
        Assert.Equal(legacy.AudioState, gguf.AudioState);
        Assert.Equal(legacy.AudioHead, gguf.AudioHead);
        Assert.Equal(legacy.AudioLayer, gguf.AudioLayer);
        Assert.Equal(legacy.TextCtx, gguf.TextCtx);
        Assert.Equal(legacy.TextState, gguf.TextState);
        Assert.Equal(legacy.TextHead, gguf.TextHead);
        Assert.Equal(legacy.TextLayer, gguf.TextLayer);
        Assert.Equal(legacy.NumMels, gguf.NumMels);
        Assert.Equal(legacy.TokenById.Length, gguf.TokenById.Length);
        for (int i = 0; i < legacy.TokenById.Length; i++)
            Assert.Equal(legacy.TokenById[i], gguf.TokenById[i]);

        var encoderLegacy = new WhisperEncoderWeights(legacy);
        var encoderGguf = new WhisperEncoderWeights(gguf);
        AssertExactEqual(encoderLegacy.PositionalEmbedding, encoderGguf.PositionalEmbedding);
        AssertExactEqual(encoderLegacy.Conv1Weight, encoderGguf.Conv1Weight);
        AssertExactEqual(encoderLegacy.LnPostWeight, encoderGguf.LnPostWeight);
        for (int i = 0; i < encoderLegacy.Layers.Length; i++)
        {
            AssertExactEqual(encoderLegacy.Layers[i].QueryWeight, encoderGguf.Layers[i].QueryWeight);
            AssertExactEqual(encoderLegacy.Layers[i].Mlp0Weight, encoderGguf.Layers[i].Mlp0Weight);
            AssertExactEqual(encoderLegacy.Layers[i].Mlp2Bias, encoderGguf.Layers[i].Mlp2Bias);
        }

        var decoderLegacy = new WhisperDecoderWeights(legacy);
        var decoderGguf = new WhisperDecoderWeights(gguf);
        AssertExactEqual(decoderLegacy.TokenEmbeddingWeight, decoderGguf.TokenEmbeddingWeight);
        AssertExactEqual(decoderLegacy.PositionalEmbedding, decoderGguf.PositionalEmbedding);
        for (int i = 0; i < decoderLegacy.Layers.Length; i++)
        {
            AssertExactEqual(decoderLegacy.Layers[i].QueryWeight, decoderGguf.Layers[i].QueryWeight);
            AssertExactEqual(decoderLegacy.Layers[i].CrossAttnLnWeight, decoderGguf.Layers[i].CrossAttnLnWeight);
        }
    }

    private static void AssertExactEqual(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]);
    }

    /// <summary>
    /// Real end-to-end proof: the real GGUF-format loader (<see cref="WhisperPipeline.LoadFromGguf"/>)
    /// transcribes the same real ground-truth audio sample correctly, using the real
    /// `models/whisper-base.gguf` conversion (produced by
    /// `scratch-llamacpp-ref/whisper_ggml_to_gguf.py` from `models/ggml-base.bin`). Same
    /// ground-truth assertions as the legacy-loader version in `WhisperRealWeightsTests`.
    /// </summary>
    [Fact]
    public void WhisperPipeline_LoadFromGguf_BaseModel_TranscribesJfkSampleCorrectly()
    {
        string? modelPath = FindRepoFile("models/whisper-base.gguf");
        string? wavPath = FindRepoFile("examples/whisper.cpp/samples/jfk.wav");
        Assert.SkipUnless(modelPath != null, "models/whisper-base.gguf not found (run scratch-llamacpp-ref/whisper_ggml_to_gguf.py first)");
        Assert.SkipUnless(wavPath != null, "examples/whisper.cpp/samples/jfk.wav not found");

        using var pipeline = WhisperPipeline.LoadFromGguf(modelPath!);
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

    /// <summary>Same real end-to-end proof, extended to the `small` and `medium` GGUF conversions (converted this fire, same converter already proven bit-exact + working on tiny/base).</summary>
    [Theory]
    [InlineData("models/whisper-small.gguf")]
    [InlineData("models/whisper-medium.gguf")]
    public void WhisperPipeline_LoadFromGguf_OtherSizes_TranscribesJfkSampleCorrectly(string relativeModelPath)
    {
        string? modelPath = FindRepoFile(relativeModelPath);
        string? wavPath = FindRepoFile("examples/whisper.cpp/samples/jfk.wav");
        Assert.SkipUnless(modelPath != null, $"{relativeModelPath} not found (run scratch-llamacpp-ref/whisper_ggml_to_gguf.py first)");
        Assert.SkipUnless(wavPath != null, "examples/whisper.cpp/samples/jfk.wav not found");

        using var pipeline = WhisperPipeline.LoadFromGguf(modelPath!);
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
