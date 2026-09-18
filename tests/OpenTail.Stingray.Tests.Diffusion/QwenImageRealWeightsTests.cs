using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight verification for Qwen-Image against the actual downloaded checkpoint
/// (city96/Qwen-Image-gguf, Q3_K_S). Per docs/086-video-model-verification-plan.md's extended
/// backlog: real DiT code, structural conformance tests only before this, never run against real
/// weights.
///
/// Verifies the DiT directly (not the full <see cref="QwenImagePipeline"/>.Generate, which also
/// calls VAE decode) -- as of 2026-09-18, the generic <c>VaeDecoder</c> class this pipeline
/// currently calls does not match Qwen-Image's real VAE tensor layout (a structurally different,
/// Wan-style causal VAE with a middle self-attention block -- see docs/086 for the real tensor
/// inventory). A dedicated Qwen-Image VAE decoder port is real, scoped, remaining work, not
/// attempted here. This test's job is to confirm what IS verified: the 60-layer DiT itself runs
/// clean and produces a numerically healthy (finite, non-degenerate) velocity prediction across a
/// real multi-step denoising loop with real weights.
/// </summary>
public sealed class QwenImageRealWeightsTests
{
    private const string ModelFileName = "qwen-image-Q3_K_S.gguf";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\_models\{fileName}",
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void QwenImageModel_RealWeights_ForwardProducesHealthyVelocity()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var weights = GgufWeightLoader.Open(modelPath);
        using var model = new QwenImageModel(weights);

        const int latH = 32, latW = 32, latC = 16;
        var latent = new float[latC * latH * latW];
        var rng = new Random(42);
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);

        int seqLen = 8;
        var textContext = new float[seqLen * QwenImageModel.ContextDim]; // placeholder zero-conditioning

        // Two-step Euler flow, matching QwenImagePipeline's own real denoising loop shape.
        var timesteps = new[] { 1000.0f, 500.0f };
        float[] velocity = [];
        foreach (var t in timesteps)
        {
            velocity = model.Forward(latent, t, textContext, latH, latW);
            for (int i = 0; i < latent.Length; i++) latent[i] -= 0.5f * velocity[i];
        }

        Assert.All(velocity, v => Assert.True(float.IsFinite(v), "DiT velocity output must be finite"));
        double sumSq = 0;
        foreach (var v in velocity) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / velocity.Length);
        Assert.True(rms > 1e-6, $"DiT velocity RMS too small ({rms}), likely degenerate/all-zero");
    }
}
