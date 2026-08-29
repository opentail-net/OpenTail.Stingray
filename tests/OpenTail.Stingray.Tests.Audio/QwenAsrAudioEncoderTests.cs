
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for the AuT audio encoder port (see docs/audio-review-
/// progress.md's QwenASR section). NOT yet golden-verified against a real oracle -- confirms
/// real weights load and the forward pass runs end-to-end without NaN/Inf, the same bar every
/// other pipeline's first real-weights test passed before golden verification followed.
/// </summary>
public sealed class QwenAsrAudioEncoderTests : HeavyTestBase
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
    public void Forward_RealWeights_ProducesFiniteOutputAtExpected8xDownsample()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var weights = new QwenAsrWeights(path!);
        var config = new QwenAsrEncoderConfig
        {
            InMelChannels = weights.NMels,
            EncoderDim = weights.AudioDim,
            NumLayers = weights.AudioLayers,
            NumHeads = weights.AudioHeads,
            QwenHiddenDim = weights.LlmDim,
        };
        using var encoder = new QwenAsrAudioEncoder(config, weights);

        int tMel = 64; // -> 8 encoder frames after 8x downsampling
        var mel = new float[tMel * weights.NMels];
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.1f * MathF.Sin(i * 0.05f);

        var (projected, numTokens) = encoder.Forward(mel, tMel);

        Assert.Equal(8, numTokens);
        Assert.Equal(numTokens * weights.LlmDim, projected.Length);
        foreach (var v in projected)
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }
}
