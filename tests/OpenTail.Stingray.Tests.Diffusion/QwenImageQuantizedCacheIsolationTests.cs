namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// SHORT, cheap isolation test (real weights, minimal size/steps) to localize why
/// QwenImageRealConditioningCoherenceTests has been silently dying with no exception/output after
/// QuantizedWeightCache was wired into QwenImageModel. Qwen Image's real checkpoint is Q3_K_S
/// quantized (qwen-image-Q3_K_S.gguf) -- DIFFERENT from FLUX.2's Q4_K_S, where the cache was
/// originally proven. This test uses the smallest possible real-weight generation (zero
/// conditioning, tiny resolution, 1 step) to get a fast pass/fail/crash signal without waiting
/// through a full real-conditioning multi-step run.
/// </summary>
public sealed class QwenImageQuantizedCacheIsolationTests
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
    public void QwenImagePipeline_TinyZeroCondSmoke_DoesNotCrash()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        string? vaePath = FindModelPath("qwen_image_vae.safetensors");
        Assert.SkipUnless(modelPath != null, "Qwen Image checkpoint not found");

        Console.WriteLine("[Isolation] Loading pipeline (zero-cond, no text encoder)...");
        using var pipeline = OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(modelPath!, vaePath);
        Console.WriteLine("[Isolation] Pipeline loaded. Starting 32x32/1-step generation...");

        string outputPath = Path.Combine(Path.GetTempPath(), "qwenimage_isolation_smoke.png");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple",
            width: 32,
            height: 32,
            steps: 1,
            guidance: 1.0f,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();
        Console.WriteLine($"[Isolation] Completed in {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(File.Exists(outputPath));
    }

    [Fact]
    public void QwenImagePipeline_MediumZeroCondSmoke_DoesNotCrash()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        string? vaePath = FindModelPath("qwen_image_vae.safetensors");
        Assert.SkipUnless(modelPath != null, "Qwen Image checkpoint not found");

        Console.WriteLine("[Isolation] Loading pipeline (zero-cond, no text encoder)...");
        using var pipeline = OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(modelPath!, vaePath);
        Console.WriteLine("[Isolation] Pipeline loaded. Starting 256x256/8-step generation (zero-cond -- isolating scale from the text encoder)...");

        string outputPath = Path.Combine(Path.GetTempPath(), "qwenimage_isolation_medium.png");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple",
            width: 256,
            height: 256,
            steps: 8,
            guidance: 4.0f,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();
        Console.WriteLine($"[Isolation] Completed in {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(File.Exists(outputPath));
    }
}
