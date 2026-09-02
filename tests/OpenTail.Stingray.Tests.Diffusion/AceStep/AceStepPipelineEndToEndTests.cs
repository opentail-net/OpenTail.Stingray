using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// First real, genuine end-to-end ACE-Step Turbo text-to-music smoke test: real weights for all
/// four V1 components (Qwen3 text encoder, condition encoder, DiT, Oobleck VAE decoder) wired
/// through the real <see cref="AceStepPipeline"/>, at a short (2s) duration to keep the real
/// 8-step Euler-ODE loop's wall-clock cost low. Non-degeneracy receipt (finite, non-silent, real
/// 48kHz-stereo-shape-correct), not yet a numeric golden-parity test against a real `diffusers`
/// end-to-end reference run -- see docs/064-acestep-implementation-plan.md.
/// </summary>
public sealed class AceStepPipelineEndToEndTests
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
    public void Generate_RealWeights_ProducesFiniteNonSilentStereoAudio()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        string? ggufPath = FindRepoFile("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");
        Assert.SkipUnless(vaePath != null, "models/acestep-v15/vae.safetensors not found");
        Assert.SkipUnless(ggufPath != null, "models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf not found");

        using var turboLoader = SafetensorsLoader.Open(turboPath!);
        var ditWeights = AceStepDiTWeights.Load(turboLoader);
        var conditionWeights = AceStepConditionEncoderWeights.Load(turboLoader);
        var timbreWeights = AceStepTimbreEncoderWeights.Load(turboLoader);

        using var vaeLoader = SafetensorsLoader.Open(vaePath!);
        var vaeWeights = AceStepOobleckDecoderWeights.Load(vaeLoader);
        var vaeEncoderWeights = AceStepOobleckEncoderWeights.Load(vaeLoader);

        using var textEncoder = new AceStepQwen3TextEncoder(ggufPath!);

        var model = new AceStepModel
        {
            Transformer = ditWeights,
            Vae = vaeWeights,
            VaeEncoder = vaeEncoderWeights,
            TextEncoder = textEncoder,
            ConditionEncoder = conditionWeights,
            TimbreEncoder = timbreWeights,
        };
        var pipeline = new AceStepPipeline(model);

        var result = pipeline.Generate(new AceStepGenerationParams
        {
            Prompt = "A cinematic orchestral soundtrack with deep drums",
            Lyrics = "",
            Instrumental = true,
            DurationSeconds = 2f, // short on purpose -- real 8-step Euler loop, keep wall-clock low
            Seed = 1234,
        });

        Assert.Equal(AceStepConfig.VaeSampleRate, result.SampleRate);
        Assert.True(result.SampleCount > 0, "generated zero samples");
        Assert.Equal(result.Left.Length, result.Right.Length);

        foreach (var v in result.Left) Assert.True(float.IsFinite(v), "left channel contains NaN/Inf");
        foreach (var v in result.Right) Assert.True(float.IsFinite(v), "right channel contains NaN/Inf");

        double sumSq = 0;
        foreach (var v in result.Left) sumSq += (double)v * v;
        foreach (var v in result.Right) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / (result.Left.Length + result.Right.Length));
        Assert.True(rms > 1e-6, $"generated audio RMS ({rms}) is near-silent -- likely a wiring bug");
    }
}
