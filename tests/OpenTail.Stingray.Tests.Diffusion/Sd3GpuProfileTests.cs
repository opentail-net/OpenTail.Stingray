using OpenTail.Stingray.Diffusion.SD3;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real GPU-split profiling for SD3.5's GPU generation (docs/094 Phase 1, 2026-09-20 perf
/// follow-up): the user observed sparse, far-apart activity "blips" on a live GPU utilization
/// graph while watching a real generation run -- a classic signature of the GPU sitting idle
/// between bursts of work (waiting on CPU-side command recording or a synchronous fence) rather
/// than being compute-bound. This uses `VulkanBackend`'s existing (previously unused for SD3.5)
/// `STINGRAY_PROFILE_GPU_SPLIT=1` counters to get REAL numbers instead of guessing: how many
/// submit+fence-wait cycles a real generation does, how much wall-clock time those cycles cost in
/// total, and how much separate CPU-side staging-buffer memcpy time is spent -- must be run with
/// that env var set (this test does not set it itself; the process must already have it set
/// before the CLR starts, since VulkanBackend's counters are gated by a `static readonly` field
/// read once at class load).
/// </summary>
public sealed class Sd3GpuProfileTests
{
    private readonly ITestOutputHelper _output;

    public Sd3GpuProfileTests(ITestOutputHelper output)
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
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray\models\_models", Path.GetFileName(relativePath)),
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    [Fact]
    public void ProfileRealGeneration()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        string clipLPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder", "model.fp16.safetensors"));
        string clipGPath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "text_encoder_2", "model.fp16.safetensors"));
        string vaePath = FindModelPath(Path.Combine("models", "sd35-medium-aux", "vae", "diffusion_pytorch_model.safetensors"));
        string tokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));
        string t5Path = FindModelPath(Path.Combine("models", "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"));
        string t5TokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));

        if (!File.Exists(ditPath) || !File.Exists(clipLPath) || !File.Exists(clipGPath) || !File.Exists(vaePath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd3GpuProfile] Checkpoints missing, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        using var pipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: vulkan, t5EncoderPath: t5Path, t5TokenizerPath: t5TokPath);

        VulkanBackend.ResetGpuProfile();

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd35_medium_apple_gpu_profiled.png");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 256,
            height: 256,
            steps: 20,
            guidance: 4.5f,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();

        string msg = $"[Sd3GpuProfile] Total generate: {sw.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        VulkanBackend.PrintGpuProfile("SD3.5 full 20-step generation");
        VulkanBackend.PrintGpuMemoryProfile("SD3.5 full 20-step generation");
    }
}
