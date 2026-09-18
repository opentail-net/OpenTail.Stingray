namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real smoke test confirming <see cref="OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(string,string,string?,IComputeBackend?)"/>
/// (the real-text-encoder overload, docs/089) constructs and disposes correctly against real
/// weights -- does NOT run a full Generate() (that's the existing, much slower
/// `QwenImageRealWeightsTests`), just confirms the additive wiring (new fields, new constructor
/// overload, Dispose ordering) doesn't crash on real checkpoints.
/// </summary>
public sealed class QwenImagePipelineTextConditioningWiringTests
{
    private static string? FindModelPath(string fileName)
    {
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
    public void Load_WithRealTextEncoder_ConstructsAndDisposesCleanly()
    {
        string? ditPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        string? textEncoderPath = FindModelPath("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
        string? vaePath = FindModelPath("qwen_image_vae.safetensors");
        Assert.SkipUnless(ditPath != null && textEncoderPath != null && vaePath != null,
            "Qwen Image DiT/text-encoder/VAE checkpoints not all found");

        using var pipeline = OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(ditPath!, textEncoderPath!, vaePath!);
        Assert.Equal("QwenImage", pipeline.Architecture);
    }
}
