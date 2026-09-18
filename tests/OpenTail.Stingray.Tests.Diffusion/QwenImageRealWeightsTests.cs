using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight verification for Qwen-Image against the actual downloaded checkpoint
/// (city96/Qwen-Image-gguf, Q3_K_S). Per docs/086-video-model-verification-plan.md's extended
/// backlog: real DiT code, structural conformance tests only before this, never run against real
/// weights.
///
/// <see cref="QwenImageModel_RealWeights_ForwardProducesHealthyVelocity"/> verifies the DiT
/// directly. <see cref="QwenImagePipeline_RealWeights_FullGenerateProducesNonDegenerateImage"/>
/// verifies the full pipeline including VAE decode -- as of 2026-09-18, `QwenImagePipeline.Load`
/// uses <see cref="Wan.WanVaeDecoder3D"/> for VAE decode (not the generic `VaeDecoder`, which does
/// not match Qwen-Image's real VAE tensor layout). This works because Qwen-Image's real VAE is
/// architecturally identical to Wan2.1's -- confirmed directly against
/// `examples/stable-diffusion.cpp/src/model.h`'s `sd_version_uses_wan_vae(VERSION_QWEN_IMAGE)`,
/// which routes it through the literal same `WAN::WanVAERunner` class, not just a similar one. See
/// docs/086-video-model-verification-plan.md for the full tensor-shape evidence and reference
/// cross-check.
/// </summary>
public sealed class QwenImageRealWeightsTests
{
    private const string ModelFileName = "qwen-image-Q3_K_S.gguf";
    private const string VaeFileName = "qwen_image_vae.safetensors";

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

    [Fact]
    public void QwenImagePipeline_RealWeights_FullGenerateProducesNonDegenerateImage()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;
        string? vaePath = FindModelPath(VaeFileName);

        using var pipeline = QwenImagePipeline.Load(modelPath, vaePath);
        Assert.Equal("QwenImage", pipeline.Architecture);

        string outPath = Path.Combine(Path.GetTempPath(), "qwenimage_full_pipeline_test.png");
        pipeline.Generate(
            prompt: "a red apple on a white table",
            width: 256,
            height: 256,
            steps: 2,
            guidance: 1.0f,
            seed: 42,
            outputPath: outPath);

        Assert.True(File.Exists(outPath));
        var info = new FileInfo(outPath);
        Assert.True(info.Length > 1000, $"output PNG too small ({info.Length} bytes), likely degenerate");
    }
}
