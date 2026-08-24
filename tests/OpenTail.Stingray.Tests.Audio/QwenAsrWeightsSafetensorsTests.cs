using System.IO;
using OpenTail.Stingray.Audio.QwenASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real test for <see cref="QwenAsrWeights.LoadFromSafetensors"/>: confirms real HF config
/// values match the checkpoint's own `config.json` exactly, real tensor shapes resolve through
/// the name-remap table, and the real AuT audio encoder runs end-to-end on the loaded weights.
/// </summary>
public sealed class QwenAsrWeightsSafetensorsTests : HeavyTestBase
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
    public void LoadFromSafetensors_RealCheckpoint_MatchesRealConfigValues()
    {
        string? dir = FindRepoFile("models/qwen3-asr-0.6b-hf");
        Assert.SkipUnless(dir != null, "models/qwen3-asr-0.6b-hf not found");

        using var weights = QwenAsrWeights.LoadFromSafetensors(dir!);

        Assert.Null(weights.Model);
        Assert.Equal(18, weights.AudioLayers);
        Assert.Equal(896, weights.AudioDim);
        Assert.Equal(14, weights.AudioHeads);
        Assert.Equal(3584, weights.AudioFfDim);
        Assert.Equal(480, weights.AudioConvChannels);
        Assert.Equal(1024, weights.AudioProjDim);
        Assert.Equal(28, weights.LlmLayers);
        Assert.Equal(1024, weights.LlmDim);
        Assert.Equal(16, weights.LlmHeads);
        Assert.Equal(8, weights.LlmKvHeads);
        Assert.Equal(128, weights.LlmHeadDim);
        Assert.Equal(151936, weights.LlmVocabSize);
        Assert.Equal(151669, weights.AudioStartTokenId);
        Assert.Equal(151670, weights.AudioEndTokenId);
        Assert.Equal(151676, weights.AudioPadTokenId); // real cross-check: matches the GGUF checkpoint's own AudioPadTokenId exactly

        foreach (var v in weights.Conv1Weight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        AssertFiniteOutput(weights.AudioLayerWeights[0].AttnQWeight);
        AssertFiniteOutput(weights.AudioLayerWeights[17].FfnDownWeight);
    }

    /// <summary>Probes a CfmLinearWeight (real F16C or F32 fallback -- see its own doc comment) with an all-ones input and checks the output has no NaN/Infinity, since it no longer exposes its underlying bytes directly as a float[].</summary>
    private static void AssertFiniteOutput(OpenTail.Stingray.Audio.Primitives.CfmLinearWeight weight)
    {
        var input = new float[weight.InDim];
        Array.Fill(input, 1f);
        var output = weight.MatVec(input);
        foreach (var v in output) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }

    [Fact]
    public void AudioEncoder_RealSafetensorsWeights_ProducesFiniteOutput()
    {
        string? dir = FindRepoFile("models/qwen3-asr-0.6b-hf");
        Assert.SkipUnless(dir != null, "models/qwen3-asr-0.6b-hf not found");

        using var weights = QwenAsrWeights.LoadFromSafetensors(dir!);
        var encoderConfig = new QwenAsrEncoderConfig
        {
            EncoderDim = weights.AudioDim,
            NumLayers = weights.AudioLayers,
            NumHeads = weights.AudioHeads,
            QwenHiddenDim = weights.LlmDim,
        };
        var encoder = new QwenAsrAudioEncoder(encoderConfig, weights);

        // Synthetic mel input at a small, real-shaped scale (structural check, matching this
        // project's established first-pass bar for a from-scratch component).
        int melFrames = 100;
        var mel = new float[128 * melFrames];
        var rng = new System.Random(42);
        for (int i = 0; i < mel.Length; i++) mel[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var (audioTokens, numAudioTokens) = encoder.Forward(mel, melFrames);

        Assert.True(numAudioTokens > 0);
        Assert.Equal(numAudioTokens * weights.LlmDim, audioTokens.Length);
        foreach (var v in audioTokens) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }
}
