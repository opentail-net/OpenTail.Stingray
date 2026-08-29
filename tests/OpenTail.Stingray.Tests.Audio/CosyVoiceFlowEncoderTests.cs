
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for CosyVoice2's flow encoder port (see docs/audio-review-
/// progress.md's CosyVoice section). NOT yet golden-verified against a real oracle -- confirms
/// real weights load with the expected real shapes and the forward pass runs end-to-end
/// without NaN/Inf, the same bar every other pipeline's first real-weights test passed before
/// golden verification followed.
/// </summary>
public sealed class CosyVoiceFlowEncoderTests : HeavyTestBase
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
    public void Weights_LoadRealTensors_ExpectedRealShapes()
    {
        string? path = FindRepoFile("models/cosyvoice2_flow.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_flow.safetensors not found");

        using var w = new CosyVoiceFlowWeights(path!);

        Assert.Equal(512, w.HiddenDim);
        Assert.Equal(8, w.NumHeads);
        Assert.Equal(64, w.HeadDim);
        Assert.Equal(2048, w.FfnDim);
        Assert.Equal(80, w.MelChannels);
        Assert.Equal(192, w.SpkEmbedDim);
        Assert.Equal(6561, w.SpeechVocabSize);
        Assert.Equal(6, w.NumEncoders);
        Assert.Equal(4, w.NumUpEncoders);

        foreach (var v in w.InputEmbeddingWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        foreach (var v in w.EncoderProjWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }

    [Fact]
    public void Forward_RealWeights_ProducesFiniteOutputAtExpectedShape()
    {
        string? path = FindRepoFile("models/cosyvoice2_flow.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_flow.safetensors not found");

        using var w = new CosyVoiceFlowWeights(path!);

        var promptTokens = new[] { 12, 45, 100 };
        var speechTokens = new[] { 200, 350, 400, 12, 88 };

        var (mu, totalFrames) = CosyVoiceFlowEncoder.Forward(w, promptTokens, speechTokens);

        int expectedT = promptTokens.Length + speechTokens.Length;
        Assert.Equal(expectedT * 2, totalFrames); // Upsample1D doubles the sequence length
        Assert.Equal(w.MelChannels * totalFrames, mu.Length);
        foreach (var v in mu) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }

    [Fact]
    public void ProjectSpeakerEmbedding_RealWeights_ProducesFiniteMelDimVector()
    {
        string? path = FindRepoFile("models/cosyvoice2_flow.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_flow.safetensors not found");

        using var w = new CosyVoiceFlowWeights(path!);
        var xvector = new float[w.SpkEmbedDim];
        for (int i = 0; i < xvector.Length; i++) xvector[i] = 0.1f * MathF.Sin(i * 0.3f);

        var projected = CosyVoiceFlowEncoder.ProjectSpeakerEmbedding(w, xvector);

        Assert.Equal(w.MelChannels, projected.Length);
        foreach (var v in projected) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }
}
