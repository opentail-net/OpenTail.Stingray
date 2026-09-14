using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights numeric parity test for CosyVoice3's GPU-residency reuse of F5-TTS's own
/// block-level GPU code (<see cref="CosyVoice3GpuWeights"/>/<see cref="CosyVoice3DiTModel.
/// RunBackboneGpu"/>, docs/080's CosyVoice entry) against the existing CPU path
/// (<see cref="CosyVoice3DiTModel.RunBackbone"/>). Mirrors <c>F5GpuResidencyParityTests</c>'s
/// pattern: real weights, a small-but-real frame count, direct numeric comparison.
/// </summary>
public sealed class CosyVoice3GpuResidencyParityTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void RunBackboneGpu_MatchesRunBackboneCpu_Numerically()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        if (path is null) return;

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var model = GgufModel.Open(path);
        var w = new CosyVoice3DiTWeights(model);

        const int t = 32;
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var rng = new Random(42);
        var h = new float[t * dim];
        for (int i = 0; i < h.Length; i++) h[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(CosyVoice3DiTModel.RotaryInvFreq(), t);
        float timestep = 0.5f;

        var cpuResult = CosyVoice3DiTModel.RunBackbone(w, (float[])h.Clone(), timestep, t, rotaryCos, rotarySin, backend: null);

        using var gpuWeights = new CosyVoice3GpuWeights(vulkan, w);
        using var ws = new F5GpuWorkspace(vulkan, t, rotaryCos, rotarySin);
        var gpuResult = CosyVoice3DiTModel.RunBackboneGpu(gpuWeights, ws, w, (float[])h.Clone(), timestep, t, vulkan);

        Assert.Equal(cpuResult.Length, gpuResult.Length);
        float maxDiff = 0f;
        double sumAbs = 0;
        for (int i = 0; i < cpuResult.Length; i++)
        {
            float diff = MathF.Abs(cpuResult[i] - gpuResult[i]);
            if (diff > maxDiff) maxDiff = diff;
            sumAbs += MathF.Abs(cpuResult[i]);
        }
        float meanAbs = (float)(sumAbs / cpuResult.Length);

        // Both paths use plain float32 here (CosyVoice3 has no Q8_0 quantization, unlike F5), so a
        // tighter tolerance than F5's own 25% full-pipeline one is expected -- this is purely
        // fp16-GPU-vs-fp32-CPU rounding over 22 layers, no quantization-scheme mismatch involved.
        Assert.True(maxDiff < 0.1f * Math.Max(1f, meanAbs),
            $"Max difference {maxDiff} too large relative to mean magnitude {meanAbs} (CPU[0]={cpuResult[0]:F6}, GPU[0]={gpuResult[0]:F6})");
    }

    [Fact]
    public void Benchmark_RunBackboneGpu_Vs_Cpu_RealTiming()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        if (path is null) return;
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var model = GgufModel.Open(path);
        var w = new CosyVoice3DiTWeights(model);

        const int t = 200;
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var rng = new Random(42);
        var h = new float[t * dim];
        for (int i = 0; i < h.Length; i++) h[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(CosyVoice3DiTModel.RotaryInvFreq(), t);
        float timestep = 0.5f;

        using var gpuWeights = new CosyVoice3GpuWeights(vulkan, w);
        using var ws = new F5GpuWorkspace(vulkan, t, rotaryCos, rotarySin);

        // Warm up.
        _ = CosyVoice3DiTModel.RunBackboneGpu(gpuWeights, ws, w, (float[])h.Clone(), timestep, t, vulkan);
        _ = CosyVoice3DiTModel.RunBackbone(w, (float[])h.Clone(), timestep, t, rotaryCos, rotarySin, backend: null);

        const int reps = 5;
        var swGpu = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
            _ = CosyVoice3DiTModel.RunBackboneGpu(gpuWeights, ws, w, (float[])h.Clone(), timestep, t, vulkan);
        swGpu.Stop();

        var swCpu = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
            _ = CosyVoice3DiTModel.RunBackbone(w, (float[])h.Clone(), timestep, t, rotaryCos, rotarySin, backend: null);
        swCpu.Stop();

        double gpuMs = swGpu.Elapsed.TotalMilliseconds / reps;
        double cpuMs = swCpu.Elapsed.TotalMilliseconds / reps;
        Console.WriteLine($"[CosyVoice3 GPU Residency Benchmark] t={t} frames, {reps} reps: GPU={gpuMs:F1}ms/call, CPU={cpuMs:F1}ms/call, speedup={cpuMs / gpuMs:F2}x");
    }
}
