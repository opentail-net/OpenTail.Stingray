using System.Diagnostics;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Pipeline;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd15GpuParityTests
{
    private readonly ITestOutputHelper _output;

    public Sd15GpuParityTests(ITestOutputHelper output)
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
    public void TestMultiHeadAttention_4096()
    {
        int qSeq = 4096, kvSeq = 4096, numHeads = 8, headDim = 40;
        int dim = numHeads * headDim;
        var rng = new Random(42);
        var q = new float[qSeq * dim];
        var k = new float[kvSeq * dim];
        var v = new float[kvSeq * dim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        using var backend = new VulkanBackend();
        var qGpu = backend.Upload(q, TensorShape.D1(q.Length));
        var kGpu = backend.Upload(k, TensorShape.D1(k.Length));
        var vGpu = backend.Upload(v, TensorShape.D1(v.Length));
        var outGpu = backend.Allocate(TensorShape.D1(qSeq * dim));
        backend.MultiHeadAttention(outGpu, qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
        var res = new float[qSeq * dim];
        backend.Download(outGpu, res);
        int nans = 0;
        for (int i = 0; i < res.Length; i++) if (float.IsNaN(res[i])) nans++;
        Console.WriteLine($"[TestMultiHeadAttention_4096] NaNs: {nans}/{res.Length}, sample: {res[0]}");
        Assert.Equal(0, nans);
    }

    [Fact]
    public void UNet_SingleForwardPass_MatchesCpuReference()
    {
        string modelPath = FindModelPath(Path.Combine("models", "sd15", "v1-5-pruned-emaonly.safetensors"));
        if (!File.Exists(modelPath))
        {
            _output.WriteLine("[Sd15GpuParityTests] Model checkpoint not found, skipping.");
            return;
        }

        int latH = 64;
        int latW = 64;
        int inC = 4;
        int contextDim = 768;
        int contextLen = 77;

        var rng = new Random(42);
        var x = new float[inC * latH * latW];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var context = new float[contextLen * contextDim];
        for (int i = 0; i < context.Length; i++) context[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float timestep = 500f;

        using var weights = SafetensorsLoader.Open(modelPath);
        var unetLoader = new PrefixWeightLoader(weights, "model.diffusion_model.");

        // 1. GPU forward pass (resident ForwardGpu)
        Console.WriteLine("[Sd15GpuParityTests] Running GPU UNet forward pass (resident)...");
        var swGpu = Stopwatch.StartNew();
        using var backend = new VulkanBackend();
        using var unetGpu = new UNet2DConditionModel(unetLoader, prefix: "", backend: backend);
        var gpuOut = unetGpu.Forward(x, timestep, context, latH, latW);
        swGpu.Stop();
        Console.WriteLine($"[Sd15GpuParityTests] GPU cold (load+upload+eval) completed in {swGpu.Elapsed.TotalSeconds:F2}s");

        Console.WriteLine("[Sd15GpuParityTests] Running GPU UNet forward pass (warm pass 2)...");
        var swWarm = Stopwatch.StartNew();
        gpuOut = unetGpu.Forward(x, timestep, context, latH, latW);
        swWarm.Stop();
        Console.WriteLine($"[Sd15GpuParityTests] GPU warm completed in {swWarm.Elapsed.TotalSeconds:F2}s");

        int gpuNaN = 0;
        float gpuMin = float.MaxValue, gpuMax = float.MinValue;
        for (int i = 0; i < gpuOut.Length; i++)
        {
            if (float.IsNaN(gpuOut[i])) gpuNaN++;
            else { gpuMin = MathF.Min(gpuMin, gpuOut[i]); gpuMax = MathF.Max(gpuMax, gpuOut[i]); }
        }
        Console.WriteLine($"[Sd15GpuParityTests] GPU: min={gpuMin:F4}, max={gpuMax:F4}, NaNs={gpuNaN}/{gpuOut.Length}");
        Assert.Equal(0, gpuNaN);

        // 2. CPU forward pass
        Console.WriteLine("[Sd15GpuParityTests] Running CPU UNet forward pass...");
        var swCpu = Stopwatch.StartNew();
        using var unetCpu = new UNet2DConditionModel(unetLoader, prefix: "", backend: null);
        var cpuOut = unetCpu.Forward(x, timestep, context, latH, latW);
        swCpu.Stop();
        Console.WriteLine($"[Sd15GpuParityTests] CPU completed in {swCpu.Elapsed.TotalSeconds:F2}s");

        // 3. Compute stats & check for NaN
        int cpuNaN = 0;
        float cpuMin = float.MaxValue, cpuMax = float.MinValue;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            if (float.IsNaN(cpuOut[i])) cpuNaN++;
            else { cpuMin = MathF.Min(cpuMin, cpuOut[i]); cpuMax = MathF.Max(cpuMax, cpuOut[i]); }
        }
        Console.WriteLine($"[Sd15GpuParityTests] CPU: min={cpuMin:F4}, max={cpuMax:F4}, NaNs={cpuNaN}/{cpuOut.Length}");
        Assert.Equal(0, cpuNaN);

        Assert.Equal(cpuOut.Length, gpuOut.Length);
        double dot = 0.0, normA = 0.0, normB = 0.0;
        float maxAbsDiff = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float a = cpuOut[i];
            float b = gpuOut[i];
            dot += (double)a * b;
            normA += (double)a * a;
            normB += (double)b * b;
            float diff = MathF.Abs(a - b);
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }

        double cosineSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Console.WriteLine($"[Sd15GpuParityTests] Cosine Similarity: {cosineSim:F6}");
        Console.WriteLine($"[Sd15GpuParityTests] Max Absolute Diff: {maxAbsDiff:F6}");

        Assert.True(cosineSim > 0.999, $"Expected cosine similarity > 0.999, got {cosineSim}");
    }
}
