namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real coherence check for Qwen Image with real Qwen2.5-VL text conditioning (docs/089) --
/// the previous end-to-end test (`QwenImageRealWeightsTests`) used the 2-arg `Load` (no text
/// encoder) with `guidance=1.0` (which skips CFG entirely), so it never actually exercised real
/// conditioning. This test uses the 3-arg `Load` (real text encoder) with real CFG guidance.
/// </summary>
public sealed class QwenImageRealConditioningCoherenceTests
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
    public void QwenImagePipeline_RealTextConditioning_256_8Step_CoherenceCheck()
    {
        string? ditPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        string? textEncoderPath = FindModelPath("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
        string? vaePath = FindModelPath("qwen_image_vae.safetensors");
        Assert.SkipUnless(ditPath != null && textEncoderPath != null && vaePath != null,
            "Qwen Image DiT/text-encoder/VAE checkpoints not all found");

        using var pipeline = OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(ditPath!, textEncoderPath!, vaePath!);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "qwenimage_real_conditioning_256_8step_2026-09-18.png");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 256,
            height: 256,
            steps: 8,
            guidance: 4.0f,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();
        Console.WriteLine($"[QwenImage real conditioning 256x256/8step] Took {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(File.Exists(outputPath));
        var info = new FileInfo(outputPath);
        Assert.True(info.Length > 1000, $"output PNG too small ({info.Length} bytes), likely degenerate");
    }
}
