using OpenTail.Stingray.Diffusion.StableDiffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd15PipelineTests
{
    [Fact]
    public void Sd15Pipeline_LoadsAndGeneratesImage_WhenModelExists()
    {
        string modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "models", "v1-5-pruned-emaonly.safetensors");
        string tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "models", "clip_tokenizer.json");

        if (!File.Exists(modelPath) || !File.Exists(tokenizerPath))
        {
            // Skip in CI environments where large 4GB model is absent
            return;
        }

        string outputPath = Path.Combine(AppContext.BaseDirectory, "test_sd15_output.png");
        if (File.Exists(outputPath)) File.Delete(outputPath);

        using var pipeline = StableDiffusionPipeline.Load(modelPath, tokenizerPath);

        // Run 2 denoising steps to verify the entire pipeline end-to-end
        pipeline.Generate(
            prompt: "A beautiful mountain lake at sunrise",
            width: 256,
            height: 256,
            steps: 2,
            guidance: 7.5f,
            seed: 42,
            outputPath: outputPath);

        Assert.True(File.Exists(outputPath), "Output PNG was not created");
        var bytes = File.ReadAllBytes(outputPath);
        Assert.True(bytes.Length > 100, "PNG file is too small");

        // Verify PNG magic header: 0x89 50 4E 47 0D 0A 1A 0A
        Assert.Equal(0x89, bytes[0]);
        Assert.Equal(0x50, bytes[1]);
        Assert.Equal(0x4E, bytes[2]);
        Assert.Equal(0x47, bytes[3]);
        Assert.Equal(0x0D, bytes[4]);
        Assert.Equal(0x0A, bytes[5]);
        Assert.Equal(0x1A, bytes[6]);
        Assert.Equal(0x0A, bytes[7]);
    }
}
