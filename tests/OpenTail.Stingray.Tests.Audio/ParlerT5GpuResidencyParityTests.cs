using OpenTail.Stingray.Audio.Parler;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights numeric parity test for Parler's T5 encoder GPU-residency reuse of FLUX's own
/// T5-XXL GPU code (<see cref="ParlerT5GpuWeights"/>/<see cref="ParlerT5GpuWorkspace"/>/
/// <see cref="T5Encoder.EncodeGpu"/>, docs/080's Parler entry) against the existing CPU path
/// (<see cref="T5Encoder.Forward"/>).
/// </summary>
public sealed class ParlerT5GpuResidencyParityTests
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
    public void EncodeGpu_MatchesForwardCpu_Numerically()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        if (modelPath is null) return;

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var w = new T5EncoderWeights(loader);

        // A short, real-ish token sequence (real vocab ids, arbitrary but plausible).
        int[] tokenIds = [37, 25, 19, 27, 9, 45, 8, 1];
        int t = tokenIds.Length;
        int dim = T5EncoderWeights.DModel;
        int ffDim = T5EncoderWeights.DFf;

        var cpuOutput = T5Encoder.Forward(w, tokenIds, backend: null);
        var cpuFlat = new float[t * dim];
        for (int i = 0; i < t; i++) Array.Copy(cpuOutput[i], 0, cpuFlat, i * dim, dim);

        var relBiasFlat = T5Encoder.ComputeRelativePositionBiasFlat(w, t);
        using var gpuWeights = new ParlerT5GpuWeights(vulkan, name => loader.ReadF32($"text_encoder.{name}"), T5EncoderWeights.NumLayers, dim, ffDim);
        using var gpuWorkspace = new ParlerT5GpuWorkspace(vulkan, t, relBiasFlat, dim, T5EncoderWeights.NumHeads, ffDim);

        var gpuFlat = T5Encoder.EncodeGpu(w, tokenIds, gpuWeights, gpuWorkspace, vulkan);

        Assert.Equal(cpuFlat.Length, gpuFlat.Length);
        float maxDiff = 0f;
        double sumAbs = 0;
        for (int i = 0; i < cpuFlat.Length; i++)
        {
            float diff = MathF.Abs(cpuFlat[i] - gpuFlat[i]);
            if (diff > maxDiff) maxDiff = diff;
            sumAbs += MathF.Abs(cpuFlat[i]);
        }
        float meanAbs = (float)(sumAbs / cpuFlat.Length);

        Assert.True(maxDiff < 0.15f * Math.Max(1f, meanAbs),
            $"Max difference {maxDiff} too large relative to mean magnitude {meanAbs} (CPU[0]={cpuFlat[0]:F6}, GPU[0]={gpuFlat[0]:F6})");
    }

    /// <summary>Isolates whether the discrepancy is real-bug-sized already after one layer, or a
    /// small compounding effect over 24 layers (mirroring F5TTS's own debugging pattern).</summary>
    [Fact]
    public void EncodeGpu_SingleLayerActivationTrace()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        if (modelPath is null) return;
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var w = new T5EncoderWeights(loader);

        int[] tokenIds = [37, 25, 19, 27, 9, 45, 8, 1];
        int t = tokenIds.Length;
        int dim = T5EncoderWeights.DModel;
        int ffDim = T5EncoderWeights.DFf;

        var relBiasFlat = T5Encoder.ComputeRelativePositionBiasFlat(w, t);
        using var gpuWeights = new ParlerT5GpuWeights(vulkan, name => loader.ReadF32($"text_encoder.{name}"), T5EncoderWeights.NumLayers, dim, ffDim);
        using var gpuWorkspace = new ParlerT5GpuWorkspace(vulkan, t, relBiasFlat, dim, T5EncoderWeights.NumHeads, ffDim);

        // Embedding lookup (shared by both paths, host-side).
        var xHost = new float[t * dim];
        for (int i = 0; i < t; i++) Array.Copy(w.SharedEmbedding, (long)tokenIds[i] * dim, xHost, (long)i * dim, dim);

        using var xInit = vulkan.Upload(xHost, TensorShape.D2(t, dim), exact: true);
        vulkan.ScaleInPlace(gpuWorkspace.X, 0f);
        vulkan.AddInPlace(gpuWorkspace.X, xInit);

        var jaggedBias = new float[T5EncoderWeights.NumHeads][,];
        for (int h = 0; h < T5EncoderWeights.NumHeads; h++)
        {
            jaggedBias[h] = new float[t, t];
            for (int i = 0; i < t; i++) for (int j = 0; j < t; j++) jaggedBias[h][i, j] = relBiasFlat[(h * t + i) * t + j];
        }

        // GPU: run exactly ONE layer, checking after attention residual too.
        var lw = gpuWeights.Layers[0];
        vulkan.RmsNormBatched(gpuWorkspace.XNorm, gpuWorkspace.X, lw.LayerNorm0Weight, dim, t, eps: 1e-6f);
        vulkan.Sgemm(gpuWorkspace.Q, gpuWorkspace.XNorm, lw.QWeight, t, dim, dim);
        vulkan.Sgemm(gpuWorkspace.K, gpuWorkspace.XNorm, lw.KWeight, t, dim, dim);
        vulkan.Sgemm(gpuWorkspace.V, gpuWorkspace.XNorm, lw.VWeight, t, dim, dim);
        vulkan.T5MultiHeadAttentionRelBias(gpuWorkspace.AttnOut, gpuWorkspace.Q, gpuWorkspace.K, gpuWorkspace.V, gpuWorkspace.RelPosBias, t, t, T5EncoderWeights.NumHeads, T5EncoderWeights.DKv);
        vulkan.Sgemm(gpuWorkspace.XNorm, gpuWorkspace.AttnOut, lw.OWeight, t, dim, dim);
        vulkan.AddInPlace(gpuWorkspace.X, gpuWorkspace.XNorm);

        var gpuAfterAttn = new float[t * dim];
        vulkan.Download(gpuWorkspace.X, gpuAfterAttn);

        vulkan.RmsNormBatched(gpuWorkspace.XNorm, gpuWorkspace.X, lw.LayerNorm1Weight, dim, t, eps: 1e-6f);
        vulkan.Sgemm(gpuWorkspace.Gate, gpuWorkspace.XNorm, lw.Wi0Weight, t, dim, ffDim);
        vulkan.Sgemm(gpuWorkspace.Val, gpuWorkspace.XNorm, lw.Wi1Weight, t, dim, ffDim);
        vulkan.GeluTanhMul(gpuWorkspace.Gate, gpuWorkspace.Val);
        vulkan.Sgemm(gpuWorkspace.FfOut, gpuWorkspace.Gate, lw.WoWight, t, ffDim, dim);
        vulkan.AddInPlace(gpuWorkspace.X, gpuWorkspace.FfOut);

        var gpuOut = new float[t * dim];
        vulkan.Download(gpuWorkspace.X, gpuOut);

        var cpuAfterAttn = T5Encoder.RunAttentionOnlyForTest(xHost, w.Layers[0], t, jaggedBias);
        float attnMaxDiff = 0f;
        double attnSumAbs = 0;
        for (int i = 0; i < cpuAfterAttn.Length; i++)
        {
            float diff = MathF.Abs(cpuAfterAttn[i] - gpuAfterAttn[i]);
            if (diff > attnMaxDiff) attnMaxDiff = diff;
            attnSumAbs += MathF.Abs(cpuAfterAttn[i]);
        }
        Console.WriteLine($"[Parler T5 SingleLayer Trace] AFTER ATTENTION ONLY: maxDiff={attnMaxDiff:F6} meanAbs={(float)(attnSumAbs / cpuAfterAttn.Length):F6}");

        var cpuOut = T5Encoder.RunSingleLayerForTest(xHost, w.Layers[0], t, jaggedBias);

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
        Console.WriteLine($"[Parler T5 SingleLayer Trace] maxDiff={maxDiff:F6} meanAbs={meanAbs:F6} worstIdx={worstIdx} CPU={cpuOut[worstIdx]:F6} GPU={gpuOut[worstIdx]:F6}");
    }

    [Fact]
    public void Benchmark_EncodeGpu_Vs_Cpu_RealTiming()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        if (modelPath is null) return;
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var w = new T5EncoderWeights(loader);

        // A realistic prompt-length token count for TTS conditioning text.
        var rng = new Random(42);
        int t = 64;
        int dim = T5EncoderWeights.DModel;
        int ffDim = T5EncoderWeights.DFf;
        var tokenIds = new int[t];
        for (int i = 0; i < t; i++) tokenIds[i] = rng.Next(10, 30000);

        var relBiasFlat = T5Encoder.ComputeRelativePositionBiasFlat(w, t);
        using var gpuWeights = new ParlerT5GpuWeights(vulkan, name => loader.ReadF32($"text_encoder.{name}"), T5EncoderWeights.NumLayers, dim, ffDim);
        using var gpuWorkspace = new ParlerT5GpuWorkspace(vulkan, t, relBiasFlat, dim, T5EncoderWeights.NumHeads, ffDim);

        // Warm up.
        _ = T5Encoder.EncodeGpu(w, tokenIds, gpuWeights, gpuWorkspace, vulkan);
        _ = T5Encoder.Forward(w, tokenIds, backend: null);

        const int reps = 5;
        var swGpu = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
            _ = T5Encoder.EncodeGpu(w, tokenIds, gpuWeights, gpuWorkspace, vulkan);
        swGpu.Stop();

        var swCpu = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
            _ = T5Encoder.Forward(w, tokenIds, backend: null);
        swCpu.Stop();

        double gpuMs = swGpu.Elapsed.TotalMilliseconds / reps;
        double cpuMs = swCpu.Elapsed.TotalMilliseconds / reps;
        Console.WriteLine($"[Parler T5 GPU Residency Benchmark] t={t} tokens, {reps} reps: GPU={gpuMs:F1}ms/call, CPU={cpuMs:F1}ms/call, speedup={cpuMs / gpuMs:F2}x");
    }
}
