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
        string t5Path = FindModelPath(Path.Combine("models", "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"));
        string t5TokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));

        if (!File.Exists(ditPath) || !File.Exists(clipLPath) || !File.Exists(clipGPath) || !File.Exists(vaePath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd3BaselineTests] Checkpoints missing, skipping.");
            return;
        }

        var swTotal = Stopwatch.StartNew();

        var swLoad = Stopwatch.StartNew();
        using var pipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: null, t5EncoderPath: t5Path, t5TokenizerPath: t5TokPath);
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
        string t5Path = FindModelPath(Path.Combine("models", "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"));
        string t5TokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));

        if (!File.Exists(ditPath) || !File.Exists(clipLPath) || !File.Exists(clipGPath) || !File.Exists(vaePath) || !File.Exists(tokPath))
        {
            _output.WriteLine("[Sd3BaselineTests] Checkpoints missing, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        var swTotal = Stopwatch.StartNew();

        var swLoad = Stopwatch.StartNew();
        using var pipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: vulkan, t5EncoderPath: t5Path, t5TokenizerPath: t5TokPath);
        swLoad.Stop();
        string msgLoad = $"[Sd3Profile Vulkan] Pipeline load took {swLoad.ElapsedMilliseconds} ms";
        _output.WriteLine(msgLoad);
        Console.WriteLine(msgLoad);

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "sd35_medium_apple_proof_20260916.png");

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

    /// <summary>
    /// Bisects the real, still-open SD3.5 GPU-vs-CPU divergence (docs/094 Phase 1, maxDiff 0.118420
    /// on the full 24-block forward) by re-running the SAME single-forward parity check truncated
    /// at increasing block counts (via <see cref="OpenTail.Stingray.Diffusion.SD3.MMDiTModel.MaxBlockIndexForDiagnostic"/>)
    /// to find the first block where CPU and GPU intermediate state actually diverges, instead of
    /// only comparing the final (24-block) output.
    /// </summary>
    [Fact]
    public void BisectGpuCpuDivergence_Sd35()
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

        int[] depths = { -1, 0, 1, 2, 3, 5, 8, 12, 16, 20, 23 };
        foreach (int maxBlock in depths)
        {
            cpuPipeline.MMDiT.MaxBlockIndexForDiagnostic = maxBlock;
            gpuPipeline.MMDiT.MaxBlockIndexForDiagnostic = maxBlock;
            bool skipDual = Environment.GetEnvironmentVariable("STINGRAY_SD3_SKIP_DUALATTN") == "1";
            cpuPipeline.MMDiT.SkipDualAttnForDiagnostic = skipDual;
            gpuPipeline.MMDiT.SkipDualAttnForDiagnostic = skipDual;

            var cpuOut = cpuPipeline.MMDiT.Forward(latents, 1000f, context, pooledY, 32, 32, 77);
            var gpuOut = gpuPipeline.MMDiT.Forward(latents, 1000f, context, pooledY, 32, 32, 77);

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
            string msg = $"[Sd3Bisect] blocks=0..{maxBlock} ({maxBlock + 1} blocks): cosine={cosSim:F6} maxDiff={maxDiff:F6}";
            _output.WriteLine(msg);
            Console.WriteLine(msg);
        }
    }

    /// <summary>
    /// Finer-grained bisection WITHIN block 0 (docs/094 Phase 1, 2026-09-20): the coarse
    /// per-block bisection above found the CPU/GPU divergence is already huge (maxDiff~10) after
    /// just ONE block, and before-block-0 state matches almost exactly (maxDiff 0.000351) -- so
    /// the bug is inside block 0's own computation, not accumulated drift. This compares
    /// intermediate GPU vs. CPU state at three named stop-points inside block 0 (modulation,
    /// QKV/QK-norm, joint attention output) to find exactly which stage first diverges.
    /// </summary>
    [Fact]
    public void BisectGpuCpuDivergence_Sd35_WithinBlock0()
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

        static (double cos, float maxDiff) Compare(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0;
            float maxDiff = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = MathF.Abs(a[i] - b[i]);
                if (diff > maxDiff) maxDiff = diff;
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            return (dot / (Math.Sqrt(na) * Math.Sqrt(nb)), maxDiff);
        }

        foreach (var stage in new[] { "mod", "qkv", "attn" })
        {
            cpuPipeline.MMDiT.MaxBlockIndexForDiagnostic = 0;
            gpuPipeline.MMDiT.MaxBlockIndexForDiagnostic = 0;
            cpuPipeline.MMDiT.DiagnosticStopStage = stage;
            gpuPipeline.MMDiT.DiagnosticStopStage = stage;

            cpuPipeline.MMDiT.Forward(latents, 1000f, context, pooledY, 32, 32, 77);
            gpuPipeline.MMDiT.Forward(latents, 1000f, context, pooledY, 32, 32, 77);

            if (stage == "mod")
            {
                var (cosMod, diffMod) = Compare(cpuPipeline.MMDiT.DiagnosticImgModOut!, gpuPipeline.MMDiT.DiagnosticImgModOut!);
                var (cosTMod, diffTMod) = Compare(cpuPipeline.MMDiT.DiagnosticTxtModOut!, gpuPipeline.MMDiT.DiagnosticTxtModOut!);
                string msg = $"[Sd3Bisect0] stage=mod: ImgMod cosine={cosMod:F6} maxDiff={diffMod:F6} | TxtMod cosine={cosTMod:F6} maxDiff={diffTMod:F6}";
                _output.WriteLine(msg); Console.WriteLine(msg);
            }
            else if (stage == "qkv")
            {
                var (cosQ, diffQ) = Compare(cpuPipeline.MMDiT.DiagnosticQOut!, gpuPipeline.MMDiT.DiagnosticQOut!);
                var (cosK, diffK) = Compare(cpuPipeline.MMDiT.DiagnosticKOut!, gpuPipeline.MMDiT.DiagnosticKOut!);
                string msg = $"[Sd3Bisect0] stage=qkv: Q cosine={cosQ:F6} maxDiff={diffQ:F6} | K cosine={cosK:F6} maxDiff={diffK:F6}";
                _output.WriteLine(msg); Console.WriteLine(msg);
            }
            else
            {
                var cpuA = cpuPipeline.MMDiT.DiagnosticAttnOut!;
                var gpuA = gpuPipeline.MMDiT.DiagnosticAttnOut!;
                var (cosA, diffA) = Compare(cpuA, gpuA);
                string msg = $"[Sd3Bisect0] stage=attn: AttnOut cosine={cosA:F6} maxDiff={diffA:F6}";
                _output.WriteLine(msg); Console.WriteLine(msg);

                const int hiddenSize = 1536, headDim = 64;
                int worstIdx = 0; float worstDiff = 0;
                for (int i = 0; i < cpuA.Length; i++)
                {
                    float d = MathF.Abs(cpuA[i] - gpuA[i]);
                    if (d > worstDiff) { worstDiff = d; worstIdx = i; }
                }
                int worstToken = worstIdx / hiddenSize;
                int worstDim = worstIdx % hiddenSize;
                int worstHead = worstDim / headDim;
                string msg2 = $"[Sd3Bisect0] worst attn element: token={worstToken} head={worstHead} dimInHead={worstDim % headDim} cpu={cpuA[worstIdx]:F4} gpu={gpuA[worstIdx]:F4}";
                _output.WriteLine(msg2); Console.WriteLine(msg2);

                // Count how many tokens have ANY element with diff > 1.0 (outlier concentration check).
                int outlierTokens = 0;
                for (int t = 0; t < cpuA.Length / hiddenSize; t++)
                {
                    bool isOutlier = false;
                    for (int d = 0; d < hiddenSize; d++)
                        if (MathF.Abs(cpuA[t * hiddenSize + d] - gpuA[t * hiddenSize + d]) > 1.0f) { isOutlier = true; break; }
                    if (isOutlier) outlierTokens++;
                }
                string msg3 = $"[Sd3Bisect0] tokens with any |diff|>1.0: {outlierTokens} / {cpuA.Length / hiddenSize}";
                _output.WriteLine(msg3); Console.WriteLine(msg3);
            }
        }
    }
}

