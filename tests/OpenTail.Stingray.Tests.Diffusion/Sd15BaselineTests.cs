using System.Diagnostics;
using OpenTail.Stingray.Diffusion.ControlNet;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd15BaselineTests
{
    private readonly ITestOutputHelper _output;

    public Sd15BaselineTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    [Fact]
    public void GenerateApple_20Steps_Sd15_Cpu()
    {
        string modelPath = FindModelPath(Path.Combine("models", "sd15", "v1-5-pruned-emaonly.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(modelPath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd15BaselineTests] Model checkpoint not found, skipping.");
            return;
        }

        var swTotal = Stopwatch.StartNew();
        using var pipeline = StableDiffusionPipeline.Load(modelPath, tokPath, backend: null);

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd15_apple_cpu_512_20steps.png");

        var swGen = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            negativePrompt: "",
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5f,
            seed: 42,
            outputPath: outputPath);
        swGen.Stop();
        swTotal.Stop();

        string msgGen = $"[Sd15Profile CPU] pipeline.Generate (20 steps 512x512) took {swGen.ElapsedMilliseconds} ms ({swGen.Elapsed.TotalSeconds:F1}s)";
        string msgTot = $"[Sd15Profile CPU] Total execution took {swTotal.ElapsedMilliseconds} ms ({swTotal.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgGen);
        _output.WriteLine(msgTot);
        Console.WriteLine(msgGen);
        Console.WriteLine(msgTot);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }

    [Fact]
    public void GenerateApple_20Steps_Sd15_Vulkan()
    {
        string modelPath = FindModelPath(Path.Combine("models", "sd15", "v1-5-pruned-emaonly.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(modelPath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd15BaselineTests] Model checkpoint not found, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        var swTotal = Stopwatch.StartNew();
        using var pipeline = StableDiffusionPipeline.Load(modelPath, tokPath, backend: vulkan);

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd15_apple_vulkan_512_20steps.png");

        var swGen = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            negativePrompt: "",
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5f,
            seed: 42,
            outputPath: outputPath);
        swGen.Stop();
        swTotal.Stop();

        string msgGen = $"[Sd15Profile Vulkan] pipeline.Generate (20 steps 512x512) took {swGen.ElapsedMilliseconds} ms ({swGen.Elapsed.TotalSeconds:F1}s)";
        string msgTot = $"[Sd15Profile Vulkan] Total execution took {swTotal.ElapsedMilliseconds} ms ({swTotal.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgGen);
        _output.WriteLine(msgTot);
        Console.WriteLine(msgGen);
        Console.WriteLine(msgTot);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }

    [Fact]
    public void GenerateApple_20Steps_Sd15_ControlNet_Cpu()
    {
        string modelPath = FindModelPath(Path.Combine("models", "sd15", "v1-5-pruned-emaonly.safetensors"));
        string cnPath = FindModelPath(Path.Combine("models", "controlnet-sd15-canny", "diffusion_pytorch_model.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(modelPath) || !File.Exists(cnPath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd15BaselineTests] Model or ControlNet checkpoint not found, skipping.");
            return;
        }

        using var pipeline = StableDiffusionPipeline.Load(modelPath, tokPath, backend: null);
        using var controlNet = ControlNetModel.Load(cnPath);

        // Simple synthetic circular edge hint (representing an apple outline)
        var hint = new float[3 * 512 * 512];
        float cx = 256f, cy = 256f, r = 100f;
        for (int y = 0; y < 512; y++)
        {
            for (int x = 0; x < 512; x++)
            {
                float d = MathF.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                if (MathF.Abs(d - r) < 2.0f)
                {
                    int idx = y * 512 + x;
                    hint[idx] = 1.0f;
                    hint[512 * 512 + idx] = 1.0f;
                    hint[2 * 512 * 512 + idx] = 1.0f;
                }
            }
        }

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd15_controlnet_apple_cpu_512_20steps.png");

        var swGen = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            negativePrompt: "",
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5f,
            seed: 42,
            controlNet: controlNet,
            controlHintRgb: hint,
            controlStrength: 1.0f,
            outputPath: outputPath);
        swGen.Stop();

        string msgGen = $"[Sd15ControlNet Profile CPU] pipeline.Generate took {swGen.ElapsedMilliseconds} ms ({swGen.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgGen);
        Console.WriteLine(msgGen);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }

    [Fact]
    public void GenerateApple_20Steps_Sd15_ControlNet_Vulkan()
    {
        string modelPath = FindModelPath(Path.Combine("models", "sd15", "v1-5-pruned-emaonly.safetensors"));
        string cnPath = FindModelPath(Path.Combine("models", "controlnet-sd15-canny", "diffusion_pytorch_model.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(modelPath) || !File.Exists(cnPath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd15BaselineTests] Model or ControlNet checkpoint not found, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        var swTotal = Stopwatch.StartNew();
        using var pipeline = StableDiffusionPipeline.Load(modelPath, tokPath, backend: vulkan);
        using var controlNet = ControlNetModel.Load(cnPath, backend: vulkan);

        // Simple synthetic circular edge hint (representing an apple outline)
        var hint = new float[3 * 512 * 512];
        float cx = 256f, cy = 256f, r = 100f;
        for (int y = 0; y < 512; y++)
        {
            for (int x = 0; x < 512; x++)
            {
                float d = MathF.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                if (MathF.Abs(d - r) < 2.0f)
                {
                    int idx = y * 512 + x;
                    hint[idx] = 1.0f;
                    hint[512 * 512 + idx] = 1.0f;
                    hint[2 * 512 * 512 + idx] = 1.0f;
                }
            }
        }

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd15_controlnet_apple_vulkan_512_20steps.png");

        var swGen = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a white table",
            negativePrompt: "",
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5f,
            seed: 42,
            controlNet: controlNet,
            controlHintRgb: hint,
            controlStrength: 1.0f,
            outputPath: outputPath);
        swGen.Stop();
        swTotal.Stop();

        string msgGen = $"[Sd15ControlNet Profile Vulkan] pipeline.Generate (20 steps 512x512) took {swGen.ElapsedMilliseconds} ms ({swGen.Elapsed.TotalSeconds:F1}s)";
        string msgTot = $"[Sd15ControlNet Profile Vulkan] Total execution took {swTotal.ElapsedMilliseconds} ms ({swTotal.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgGen);
        _output.WriteLine(msgTot);
        Console.WriteLine(msgGen);
        Console.WriteLine(msgTot);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}
