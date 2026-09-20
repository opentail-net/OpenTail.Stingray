using OpenTail.Stingray.Diffusion.SD3;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Per-step CPU-vs-GPU latent trajectory comparison (docs/094 Phase 1, 2026-09-20 GPU
/// investigation, follow-up to the CPU int8-quantization fix). The fix closed the SINGLE-forward-
/// pass parity gap almost completely (cosine 0.999724/maxDiff 0.118420 -> cosine 1.000000/maxDiff
/// 0.000898), yet the full 20-step Vulkan `Generate` run is still visually pure checkerboard noise
/// -- meaning the bug is NOT in `MMDiTModel.Forward`'s single-call math (now proven near-exact) but
/// somewhere in the multi-step trajectory: GPU-resident weight/context caching across steps, or an
/// error that starts small and compounds. This manually replicates `Sd3Pipeline.Generate`'s own
/// Euler loop for both backends with IDENTICAL real encoded text conditioning and initial noise,
/// dumping latent (x) mean/std/maxabs after every step to find exactly where CPU and GPU first
/// diverge in the real trajectory.
/// </summary>
public sealed class Sd3PerStepTrajectoryParityTests
{
    private readonly ITestOutputHelper _output;

    public Sd3PerStepTrajectoryParityTests(ITestOutputHelper output)
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
    public void PerStepLatentTrajectory_Cpu_Vs_Gpu()
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
            _output.WriteLine("[Sd3Trajectory] Checkpoints missing, skipping.");
            return;
        }

        using var cpuPipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: null, t5EncoderPath: t5Path, t5TokenizerPath: t5TokPath);
        using var vulkan = new VulkanBackend();
        using var gpuPipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: vulkan, t5EncoderPath: t5Path, t5TokenizerPath: t5TokPath);

        // Real text conditioning, computed once via the CPU-only encoders (ClipLEncoder/
        // OpenClipGEncoder/T5Encoder take no backend parameter at all -- always CPU regardless of
        // which MMDiT/VAE backend a pipeline uses), so both trajectories below start from
        // IDENTICAL condContext/pooledY -- any divergence can only come from MMDiT.Forward or the
        // Euler stepping, not text encoding.
        var prompt = "a red apple on a wooden table";
        var (condContext, condPooledY, numTextTokens) = cpuPipeline.EncodePromptForTesting(prompt);
        var (uncondContext, uncondPooledY, _) = cpuPipeline.EncodePromptForTesting("");

        const int latH = 32, latW = 32, latC = 16;
        int latentCount = latC * latH * latW;

        // Identical initial noise for both trajectories (real Box-Muller, same as Sd3Pipeline.Generate,
        // seeded identically).
        var rng = new Random(42);
        var x0 = new float[latentCount];
        for (int i = 0; i < latentCount - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            x0[i] = (float)(radius * Math.Cos(theta));
            x0[i + 1] = (float)(radius * Math.Sin(theta));
        }

        const int steps = 6; // enough to see divergence trend without paying the full 20-step cost
        const float guidance = 4.5f;
        float dt = 1.0f / 20; // same dt as a real 20-step run (steps use the SAME schedule granularity)

        var xCpu = (float[])x0.Clone();
        var xGpu = (float[])x0.Clone();

        static (double mean, double std, double maxAbs) Stats(float[] a)
        {
            double sum = 0, sumSq = 0, maxAbs = 0;
            foreach (var v in a) { sum += v; sumSq += (double)v * v; maxAbs = Math.Max(maxAbs, Math.Abs(v)); }
            double mean = sum / a.Length;
            double std = Math.Sqrt(sumSq / a.Length - mean * mean);
            return (mean, std, maxAbs);
        }

        static float MaxAbsDiff(float[] a, float[] b)
        {
            float m = 0;
            for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
            return m;
        }

        for (int step = 0; step < steps; step++)
        {
            float t = 1.0f - step * dt;
            float timestep = t * 1000.0f;

            var cpuCond = cpuPipeline.MMDiT.Forward(xCpu, timestep, condContext, condPooledY, latH, latW, numTextTokens);
            var cpuUncond = cpuPipeline.MMDiT.Forward(xCpu, timestep, uncondContext, uncondPooledY, latH, latW, numTextTokens);
            var gpuCond = gpuPipeline.MMDiT.Forward(xGpu, timestep, condContext, condPooledY, latH, latW, numTextTokens);
            var gpuUncond = gpuPipeline.MMDiT.Forward(xGpu, timestep, uncondContext, uncondPooledY, latH, latW, numTextTokens);

            for (int i = 0; i < xCpu.Length; i++)
            {
                float vCpu = cpuUncond[i] + guidance * (cpuCond[i] - cpuUncond[i]);
                xCpu[i] -= dt * vCpu;
                float vGpu = gpuUncond[i] + guidance * (gpuCond[i] - gpuUncond[i]);
                xGpu[i] -= dt * vGpu;
            }

            var (mc, sc, xc) = Stats(xCpu);
            var (mg, sg, xg) = Stats(xGpu);
            float maxDiff = MaxAbsDiff(xCpu, xGpu);
            string msg = $"[Sd3Trajectory] step {step}: CPU(mean={mc:F5} std={sc:F5} maxAbs={xc:F3}) GPU(mean={mg:F5} std={sg:F5} maxAbs={xg:F3}) latent maxDiff={maxDiff:F5}";
            _output.WriteLine(msg);
            Console.WriteLine(msg);
        }
    }
}
