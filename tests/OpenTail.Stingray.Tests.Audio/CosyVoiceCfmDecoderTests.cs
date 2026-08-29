
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for CosyVoice2's CFM flow-decoder port (see docs/audio-review-
/// progress.md's CosyVoice section). NOT yet golden-verified against a real oracle -- confirms
/// real weights load with the expected real shapes (cross-checked against the real
/// `cosyvoice2.yaml` config and the actual local checkpoint's tensor names/shapes) and the
/// Euler ODE solve runs end-to-end without NaN/Inf, the same bar every other pipeline's first
/// real-weights test passed before golden verification followed.
/// </summary>
public sealed class CosyVoiceCfmDecoderTests : HeavyTestBase
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

        using var flow = new CosyVoiceFlowWeights(path!);
        var w = new CosyVoiceCfmDecoderWeights(flow);

        Assert.Equal(12, w.MidStages.Length);
        Assert.Equal(4, w.DownStage.TransformerBlocks.Length);
        Assert.Equal(4, w.UpStage.TransformerBlocks.Length);
        Assert.NotNull(w.DownStage.ResampleConvWeight);
        Assert.NotNull(w.UpStage.ResampleConvWeight);
        Assert.Null(w.MidStages[0].ResampleConvWeight);

        Assert.Equal(1024 * 320, w.TimeMlpLinear1Weight.Length);
        Assert.Equal(80 * 256, w.FinalProjWeight.Length);

        foreach (var v in w.TimeMlpLinear1Weight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        foreach (var v in w.FinalProjWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }

    [Fact]
    public void Generate_RealWeights_ProducesFiniteMelAtExpectedShape()
    {
        string? path = FindRepoFile("models/cosyvoice2_flow.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_flow.safetensors not found");

        using var flow = new CosyVoiceFlowWeights(path!);
        var w = new CosyVoiceCfmDecoderWeights(flow);

        var promptTokens = new[] { 12, 45, 100 };
        var speechTokens = new[] { 200, 350, 400, 12, 88 };
        var (mu, totalFrames) = CosyVoiceFlowEncoder.Forward(flow, promptTokens, speechTokens);

        var spkEmbed = CosyVoiceFlowEncoder.ProjectSpeakerEmbedding(flow, new float[flow.SpkEmbedDim]);

        var rng = new Random(1234);
        var mel = CosyVoiceCfmDecoder.Generate(w, mu, spkEmbed, totalFrames, rng, nSteps: 10);

        Assert.Equal(CosyVoiceCfmDecoderWeights.OutChannels * totalFrames, mel.Length);
        foreach (var v in mel) Assert.False(float.IsNaN(v) || float.IsInfinity(v));

        bool anyNonZero = false;
        foreach (var v in mel) if (MathF.Abs(v) > 1e-6f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Generated mel looks degenerate (all zero).");
    }
}
