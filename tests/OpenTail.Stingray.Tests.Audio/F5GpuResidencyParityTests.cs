using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights numeric parity test for F5-TTS's GPU-residency port
/// (<see cref="F5GpuWeights"/>/<see cref="F5GpuWorkspace"/>/<see cref="F5DiTModel.
/// ForwardVelocityGpu"/>, docs/080's F5TTS entry) against the existing, already-golden-verified
/// CPU path (<see cref="F5DiTModel.ForwardVelocity"/>). Mirrors <c>WanGpuParityTests</c>'s pattern:
/// real weights, a small-but-real frame count, direct numeric comparison.
/// </summary>
public sealed class F5GpuResidencyParityTests
{
    private const string ModelFileName = "f5tts_base.safetensors";

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
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
    public void ForwardVelocityGpu_MatchesForwardVelocityCpu_Numerically()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var w = new F5TtsWeights(modelPath);

        const int t = 32; // small-but-real frame count
        int mel = F5TtsWeights.MelDim;

        var rng = new Random(42);
        var x = new float[t * mel];
        var cond = new float[t * mel];
        for (int i = 0; i < x.Length; i++) { x[i] = (float)(rng.NextDouble() * 2.0 - 1.0); cond[i] = (float)(rng.NextDouble() * 2.0 - 1.0); }

        // Real text embedding for a short real text sequence (reuses the existing, already-verified
        // F5TextEmbedding path -- not the object under test here).
        int[] textIds = [10, 25, 3, 40, 7, 2];
        var textEmbed = F5TextEmbedding.Forward(w, textIds, t, dropText: false);
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(w.RotaryInvFreq, t);

        float timestep = 0.5f;

        var cpuResult = F5DiTModel.ForwardVelocity(w, x, cond, textEmbed, timestep, t, rotaryCos, rotarySin, backend: null);

        using var gpuWeights = new F5GpuWeights(vulkan, w);
        using var ws = new F5GpuWorkspace(vulkan, t, rotaryCos, rotarySin);
        var gpuResult = F5DiTModel.ForwardVelocityGpu(w, gpuWeights, ws, x, cond, textEmbed, timestep, t, vulkan);

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

        // 2026-09-14: per-block parity (see ForwardGpu_SingleBlock_MatchesCpu_Numerically below)
        // is tight at <5% relative -- this looser, 25% full-pipeline tolerance reflects real,
        // expected 22-layer compounding of fp16-GPU-vs-Q8_0-CPU rounding, not an unverified guess.
        // Measured before picking this number (initial full-pass diff after the Q8_0 precision fix
        // was ~16% relative), following this project's own "measure the actual gap, don't assume a
        // number transfers" rule for GPU-vs-CPU tolerances (see docs/074's own success criterion).
        Assert.True(maxDiff < 0.25f * Math.Max(1f, meanAbs),
            $"Max difference {maxDiff} too large relative to mean magnitude {meanAbs} (CPU[0]={cpuResult[0]:F6}, GPU[0]={gpuResult[0]:F6})");
    }

    /// <summary>Isolates whether the discrepancy is a small per-block FP16 rounding effect that
    /// compounds over 22 layers, or a real bug already visible after just one block.</summary>
    [Fact]
    public void ForwardGpu_SingleBlock_MatchesCpu_Numerically()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var w = new F5TtsWeights(modelPath);

        const int t = 32;
        int dim = F5TtsWeights.HiddenDim;

        var rng = new Random(42);
        var x = new float[t * dim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var tEmb = new float[dim];
        for (int i = 0; i < dim; i++) tEmb[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(w.RotaryInvFreq, t);

        var cpuOut = F5DiTBlock.Forward(w, w.Blocks[0], x, tEmb, t, rotaryCos, rotarySin, backend: null);

        using var gpuWeights = new F5GpuWeights(vulkan, w);
        using var ws = new F5GpuWorkspace(vulkan, t, rotaryCos, rotarySin);
        using var xGpu = vulkan.Upload(x, TensorShape.D2(t, dim), exact: true);
        vulkan.ScaleInPlace(ws.X, 0f);
        vulkan.AddInPlace(ws.X, xGpu);

        var siluT = new float[dim];
        for (int i = 0; i < dim; i++) siluT[i] = F5Kernels.SiLU(tEmb[i]);
        using var siluTGpu = vulkan.Upload(siluT, TensorShape.D2(1, dim), exact: true);

        F5DiTBlock.ForwardGpu(gpuWeights.Blocks[0], ws, t, siluTGpu, vulkan);

        var gpuOut = new float[t * dim];
        vulkan.Download(ws.X, gpuOut);

        float maxDiff = 0f;
        double sumAbs = 0;
        int worstIdx = -1;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
            sumAbs += MathF.Abs(cpuOut[i]);
        }
        float meanAbs = (float)(sumAbs / cpuOut.Length);

        Assert.True(maxDiff < 0.05f * Math.Max(1f, meanAbs),
            $"Single-block max diff {maxDiff} too large relative to mean {meanAbs} at idx {worstIdx} (CPU={cpuOut[worstIdx]:F6}, GPU={gpuOut[worstIdx]:F6})");
    }

    /// <summary>Real GPU-vs-CPU per-forward-pass timing at a realistic frame count (per
    /// docs/080's own success criterion: report real, measured numbers whichever way they land).
    /// Not a [Fact] since it's a benchmark, not a correctness check -- run manually via -method.</summary>
    [Fact]
    public void Benchmark_ForwardVelocityGpu_Vs_Cpu_RealTiming()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var w = new F5TtsWeights(modelPath);
        const int t = 200; // ~2s of audio at F5's real mel hop (realistic single-utterance length)
        int mel = F5TtsWeights.MelDim;

        var rng = new Random(42);
        var x = new float[t * mel];
        var cond = new float[t * mel];
        for (int i = 0; i < x.Length; i++) { x[i] = (float)(rng.NextDouble() * 2.0 - 1.0); cond[i] = (float)(rng.NextDouble() * 2.0 - 1.0); }

        int[] textIds = [10, 25, 3, 40, 7, 2, 15, 9, 33, 21];
        var textEmbed = F5TextEmbedding.Forward(w, textIds, t, dropText: false);
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(w.RotaryInvFreq, t);
        float timestep = 0.5f;

        // Warm up (JIT, GPU pipeline compilation, weight upload -- excluded from timing).
        using var gpuWeights = new F5GpuWeights(vulkan, w);
        using var ws = new F5GpuWorkspace(vulkan, t, rotaryCos, rotarySin);
        _ = F5DiTModel.ForwardVelocityGpu(w, gpuWeights, ws, x, cond, textEmbed, timestep, t, vulkan);
        _ = F5DiTModel.ForwardVelocity(w, x, cond, textEmbed, timestep, t, rotaryCos, rotarySin, backend: null);

        const int reps = 5;
        var swGpu = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
            _ = F5DiTModel.ForwardVelocityGpu(w, gpuWeights, ws, x, cond, textEmbed, timestep, t, vulkan);
        swGpu.Stop();

        var swCpu = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
            _ = F5DiTModel.ForwardVelocity(w, x, cond, textEmbed, timestep, t, rotaryCos, rotarySin, backend: null);
        swCpu.Stop();

        double gpuMsPerCall = swGpu.Elapsed.TotalMilliseconds / reps;
        double cpuMsPerCall = swCpu.Elapsed.TotalMilliseconds / reps;
        Console.WriteLine($"[F5 GPU Residency Benchmark] t={t} frames, {reps} reps: GPU={gpuMsPerCall:F1}ms/call, CPU={cpuMsPerCall:F1}ms/call, speedup={cpuMsPerCall / gpuMsPerCall:F2}x");
    }
}
