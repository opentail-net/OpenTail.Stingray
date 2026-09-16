using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class FluxRealWeightsTests
{
    private static string? FindRepoFile(string relativePath)
    {
        string direct = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", relativePath);
        if (File.Exists(direct) || Directory.Exists(direct)) return direct;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void GenerateApple_4Steps_Flux_Vulkan()
    {
        string? ditPath = FindRepoFile("models/flux1-schnell/flux1-schnell-Q4_K_S.gguf");
        string? vaePath = FindRepoFile("models/flux1-schnell/ae.safetensors");
        string? clipPath = FindRepoFile("models/flux1-schnell/clip_l.safetensors");
        string? clipTok = FindRepoFile("models/flux1-schnell/tokenizer_clip/tokenizer.json");
        string? t5Path = FindRepoFile("models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors");
        string? t5Tok = FindRepoFile("models/flux1-schnell/tokenizer_t5/tokenizer.json");

        if (ditPath is null || vaePath is null || clipPath is null || clipTok is null || t5Path is null || t5Tok is null)
        {
            Console.WriteLine("[FluxRealWeightsTests] Checkpoints not found, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        var swLoad = Stopwatch.StartNew();
        using var pipeline = ImagePipeline.Load(ditPath, vaePath, clipPath, clipTok, t5Path, t5Tok, vulkan);
        swLoad.Stop();
        Console.WriteLine($"[FluxProfile Vulkan] Pipeline load took {swLoad.ElapsedMilliseconds} ms");

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "flux_apple_vulkan_512_4steps.png");

        // Pass 1 (Cold)
        var swCold = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            width: 512,
            height: 512,
            steps: 4,
            guidance: 1.0f,
            seed: 42,
            outputPath: outputPath);
        swCold.Stop();
        Console.WriteLine($"[FluxProfile Vulkan Pass 1 (Cold)] pipeline.Generate (4 steps 512x512) took {swCold.ElapsedMilliseconds} ms ({swCold.Elapsed.TotalSeconds:F1}s)");

        // Pass 2 (Warm)
        var swWarm = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            width: 512,
            height: 512,
            steps: 4,
            guidance: 1.0f,
            seed: 43,
            outputPath: outputPath);
        swWarm.Stop();
        Console.WriteLine($"[FluxProfile Vulkan Pass 2 (Warm)] pipeline.Generate (4 steps 512x512) took {swWarm.ElapsedMilliseconds} ms ({swWarm.Elapsed.TotalSeconds:F1}s)");

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}
