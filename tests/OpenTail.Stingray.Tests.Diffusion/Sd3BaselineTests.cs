using System.Diagnostics;
using OpenTail.Stingray.Diffusion.SD3;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd3BaselineTests
{
    private readonly ITestOutputHelper _output;

    public Sd3BaselineTests(ITestOutputHelper output)
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
    public void TestSd35GpuVsCpuParity()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        string clipLPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder", "model.fp16.safetensors"));
        string clipGPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder_2", "model.fp16.safetensors"));
        string vaePath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "vae", "diffusion_pytorch_model.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(ditPath) || !File.Exists(clipLPath) || !File.Exists(clipGPath) || !File.Exists(vaePath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd3BaselineTests] Checkpoints missing, skipping.");
            return;
        }

        using var cpuPipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: null);
        using var vulkan = new VulkanBackend();
        using var gpuPipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: vulkan);

        var rng = new Random(42);
        var latents = new float[16 * 32 * 32];
        for (int i = 0; i < latents.Length; i++) latents[i] = (float)rng.NextDouble() - 0.5f;

        var context = new float[77 * 4096];
        for (int i = 0; i < context.Length; i++) context[i] = (float)rng.NextDouble() * 0.1f;

        var pooledY = new float[2048];
        for (int i = 0; i < pooledY.Length; i++) pooledY[i] = (float)rng.NextDouble() * 0.1f;

        _output.WriteLine("[Sd3Parity] Running GPU forward pass (resident)...");
        var swGpu = Stopwatch.StartNew();
        var gpuOut = gpuPipeline.MMDiT.Forward(latents, 1000f, context, pooledY, 32, 32, 77);
        swGpu.Stop();
        string msgGpu = $"[Sd3Parity] GPU forward took {swGpu.ElapsedMilliseconds} ms ({swGpu.Elapsed.TotalSeconds:F2}s)";
        _output.WriteLine(msgGpu);
        Console.WriteLine(msgGpu);

        _output.WriteLine("[Sd3Parity] Running CPU forward pass...");
        var swCpu = Stopwatch.StartNew();
        var cpuOut = cpuPipeline.MMDiT.Forward(latents, 1000f, context, pooledY, 32, 32, 77);
        swCpu.Stop();
        string msgCpu = $"[Sd3Parity] CPU forward took {swCpu.ElapsedMilliseconds} ms ({swCpu.Elapsed.TotalSeconds:F2}s)";
        _output.WriteLine(msgCpu);
        Console.WriteLine(msgCpu);

        double dot = 0, normA = 0, normB = 0;
        float maxDiff = 0;
        for (int i = 0; i < gpuOut.Length; i++)
        {
            float diff = MathF.Abs(gpuOut[i] - cpuOut[i]);
            if (diff > maxDiff) maxDiff = diff;
            dot += gpuOut[i] * cpuOut[i];
            normA += gpuOut[i] * gpuOut[i];
            normB += cpuOut[i] * cpuOut[i];
        }
        double cosSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        string msgResult = $"[Sd3Parity] Cosine: {cosSim:F6}, MaxDiff: {maxDiff:F6}";
        _output.WriteLine(msgResult);
        Console.WriteLine(msgResult);

        Assert.True(cosSim > 0.99, $"Expected cosine similarity > 0.99, got {cosSim:F6}");
    }

    [Fact]
    public void GenerateApple_20Steps_Sd35_Cpu()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        string clipLPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder", "model.fp16.safetensors"));
        string clipGPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder_2", "model.fp16.safetensors"));
        string vaePath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "vae", "diffusion_pytorch_model.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(ditPath) || !File.Exists(clipLPath) || !File.Exists(clipGPath) || !File.Exists(vaePath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd3BaselineTests] Checkpoints missing, skipping.");
            return;
        }

        var swTotal = Stopwatch.StartNew();

        var swLoad = Stopwatch.StartNew();
        using var pipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: null);
        swLoad.Stop();
        string msgLoad = $"[Sd3Profile CPU] Pipeline load took {swLoad.ElapsedMilliseconds} ms";
        _output.WriteLine(msgLoad);
        Console.WriteLine(msgLoad);

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd35_medium_apple_cpu_256_20steps.png");

        var swGen = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 256,
            height: 256,
            steps: 20,
            guidance: 4.5f,
            seed: 42,
            outputPath: outputPath);
        swGen.Stop();
        swTotal.Stop();

        string msgGen = $"[Sd3Profile CPU] pipeline.Generate (20 steps 256x256) took {swGen.ElapsedMilliseconds} ms ({swGen.Elapsed.TotalSeconds:F1}s)";
        string msgTot = $"[Sd3Profile CPU] Total execution took {swTotal.ElapsedMilliseconds} ms ({swTotal.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgGen);
        _output.WriteLine(msgTot);
        Console.WriteLine(msgGen);
        Console.WriteLine(msgTot);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }

    [Fact]
    public void GenerateApple_20Steps_Sd35_Vulkan()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        string clipLPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder", "model.fp16.safetensors"));
        string clipGPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder_2", "model.fp16.safetensors"));
        string vaePath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "vae", "diffusion_pytorch_model.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));

        if (!File.Exists(ditPath) || !File.Exists(clipLPath) || !File.Exists(clipGPath) || !File.Exists(vaePath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd3BaselineTests] Checkpoints missing, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        var swTotal = Stopwatch.StartNew();

        var swLoad = Stopwatch.StartNew();
        using var pipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: vulkan);
        swLoad.Stop();
        string msgLoad = $"[Sd3Profile Vulkan] Pipeline load took {swLoad.ElapsedMilliseconds} ms";
        _output.WriteLine(msgLoad);
        Console.WriteLine(msgLoad);

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd35_medium_apple_vulkan_256_20steps.png");

        var swGen = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 256,
            height: 256,
            steps: 20,
            guidance: 4.5f,
            seed: 42,
            outputPath: outputPath);
        swGen.Stop();
        swTotal.Stop();

        string msgGen = $"[Sd3Profile Vulkan Pass 1 (Cold)] pipeline.Generate (20 steps 256x256) took {swGen.ElapsedMilliseconds} ms ({swGen.Elapsed.TotalSeconds:F1}s)";
        string msgTot = $"[Sd3Profile Vulkan Pass 1 (Cold)] Total execution took {swTotal.ElapsedMilliseconds} ms ({swTotal.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgGen);
        _output.WriteLine(msgTot);
        Console.WriteLine(msgGen);
        Console.WriteLine(msgTot);

        var swWarm = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 256,
            height: 256,
            steps: 20,
            guidance: 4.5f,
            seed: 43,
            outputPath: outputPath);
        swWarm.Stop();

        string msgWarm = $"[Sd3Profile Vulkan Pass 2 (Warm)] pipeline.Generate (20 steps 256x256) took {swWarm.ElapsedMilliseconds} ms ({swWarm.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msgWarm);
        Console.WriteLine(msgWarm);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}

