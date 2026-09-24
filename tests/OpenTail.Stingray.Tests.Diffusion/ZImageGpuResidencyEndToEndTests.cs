using System.Diagnostics;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real end-to-end measurement of Z-Image-Turbo's newly-wired GPU residency (docs/094
/// Phase 7): the resident <see cref="ZImageDiT.ApplyBlockGpu"/>/<see cref="ZImageGpuWeights"/>/
/// <see cref="ZImageGpuWorkspace"/> trio already existed and was parity-tested, but nothing in
/// <see cref="ZImageDiT.Forward"/> ever called it until this pass -- real production Vulkan runs
/// still went through the old per-op immediate-dispatch path. Baseline to compare against
/// (PerformanceLeague.md, 2026-09-13): 183.6s Vulkan, 256x256/4 steps, the SAME real coherent
/// red-apple config the 2026-09-12 sign-convention fix verified.
/// </summary>
public sealed class ZImageGpuResidencyEndToEndTests
{
    private readonly ITestOutputHelper _output;

    public ZImageGpuResidencyEndToEndTests(ITestOutputHelper output)
    {
        _output = output;
    }

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
        return null;
    }

    [Fact]
    public void ZImagePipeline_Gpu_256x256_4Steps_RealApple()
    {
        string? ditPath = FindModelPath("z_image_turbo-Q4_0.gguf") ?? FindModelPath("z_image_turbo-Q5_0.gguf");
        string? encoderPath = FindModelPath("Z-Image-AbliteratedV1.Q5_K_M.gguf");
        string vaePath = @"C:\Git-Public\OpenTail.Stingray\models\z-image-turbo\vae\diffusion_pytorch_model.safetensors";
        string tokenizerPath = @"C:\Git-Public\OpenTail.Stingray\models\z-image-turbo\tokenizer\tokenizer.json";

        if (ditPath is null || encoderPath is null || !File.Exists(vaePath) || !File.Exists(tokenizerPath))
        {
            _output.WriteLine("[ZImageGpuResidencyEndToEndTests] Checkpoints missing, skipping.");
            Console.WriteLine("[ZImageGpuResidencyEndToEndTests] Checkpoints missing, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        using var pipeline = ZImagePipeline.Load(ditPath, vaePath, encoderPath, tokenizerPath, vulkan);

        Environment.SetEnvironmentVariable("STINGRAY_ZIMAGE_DUMP_LATENT", "1");
        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "zimage_vulkan_restored.png");

        var sw = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            width: 256,
            height: 256,
            steps: 4,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();

        string msg = $"[ZImage GPU residency] Generate (256x256, 4 steps) took {sw.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(msg); Console.WriteLine(msg);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }

    [Fact]
    public void ZImagePipeline_Cpu_256x256_4Steps_RealApple()
    {
        string? ditPath = FindModelPath("z_image_turbo-Q4_0.gguf") ?? FindModelPath("z_image_turbo-Q5_0.gguf");
        string? encoderPath = FindModelPath("Z-Image-AbliteratedV1.Q5_K_M.gguf");
        string vaePath = @"C:\Git-Public\OpenTail.Stingray\models\z-image-turbo\vae\diffusion_pytorch_model.safetensors";
        string tokenizerPath = @"C:\Git-Public\OpenTail.Stingray\models\z-image-turbo\tokenizer\tokenizer.json";

        if (ditPath is null || encoderPath is null || !File.Exists(vaePath) || !File.Exists(tokenizerPath))
        {
            _output.WriteLine("[ZImageGpuResidencyEndToEndTests] Checkpoints missing, skipping.");
            return;
        }

        using var pipeline = ZImagePipeline.Load(ditPath, vaePath, encoderPath, tokenizerPath, backend: null);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "zimage_cpu_verified.png");

        var sw = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            width: 256,
            height: 256,
            steps: 4,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();

        string msg = $"[ZImage CPU] Generate (256x256, 4 steps) took {sw.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(msg); Console.WriteLine(msg);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }

    [Fact]
    public void ZImagePipeline_Cpu_64x64_4Steps_FastApple()
    {
        Environment.SetEnvironmentVariable("STINGRAY_ZIMAGE_DUMP_LATENT", "1");
        string? ditPath = FindModelPath("z_image_turbo-Q4_0.gguf") ?? FindModelPath("z_image_turbo-Q5_0.gguf");
        string? encoderPath = FindModelPath("Z-Image-AbliteratedV1.Q5_K_M.gguf");
        string vaePath = @"C:\Git-Public\OpenTail.Stingray\models\z-image-turbo\vae\diffusion_pytorch_model.safetensors";
        string tokenizerPath = @"C:\Git-Public\OpenTail.Stingray\models\z-image-turbo\tokenizer\tokenizer.json";

        if (ditPath is null || encoderPath is null || !File.Exists(vaePath) || !File.Exists(tokenizerPath))
        {
            _output.WriteLine("[ZImageGpuResidencyEndToEndTests] Checkpoints missing, skipping.");
            return;
        }

        using var pipeline = ZImagePipeline.Load(ditPath, vaePath, encoderPath, tokenizerPath, backend: null);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "zimage_cpu_64x64_test.png");

        var sw = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            width: 64,
            height: 64,
            steps: 4,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();

        string msg = $"[ZImage CPU] Generate (64x64, 4 steps) took {sw.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(msg); Console.WriteLine(msg);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}
