using OpenTail.Stingray.Core;
using System.Diagnostics;
using OpenTail.Stingray.Diffusion.HunyuanVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// HunyuanVideo transformer: GPU blocks (<c>HunyuanVideoModel.Gpu.cs</c>, raw fp8 weights through
/// <c>Shaders.SgemmFp8W</c>) against the CPU path on the REAL 720p cfg-distilled fp8 checkpoint, one
/// full forward at a small latent (1 frame, 16x16 latent = 64 image tokens) with synthetic text
/// context. Both paths multiply the identical decoded weights; differences are fp32 accumulation
/// order across 60 blocks. Skips (visibly) when the checkpoint or a Vulkan device is absent.
/// </summary>
public sealed class HunyuanVideoGpuParityTests
{
    private readonly ITestOutputHelper _out;
    public HunyuanVideoGpuParityTests(ITestOutputHelper output) => _out = output;

    private static string? FindCheckpoint()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string p = Path.Combine(dir.FullName, "models", "hunyuanvideo", "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors");
            if (File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }

    [Fact]
    public void GpuBlocks_MatchCpu_OnRealFp8Checkpoint()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "Heavy: set STINGRAY_RUN_HEAVY_TESTS=1 (loads the ~13 GB fp8 HunyuanVideo checkpoint).");
        string? path = FindCheckpoint();
        Assert.SkipUnless(path is not null, "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors not present");

        global::OpenTail.Stingray.Vulkan.VulkanBackend gpu;
        try { gpu = new global::OpenTail.Stingray.Vulkan.VulkanBackend(); }
        catch (Exception ex) { Assert.Skip($"Vulkan device could not be created: {ex.Message}"); throw; }
        using var _gpu = gpu;
        using var loader = SafetensorsLoader.Open(path!);
        using var model = new HunyuanVideoModel(loader, backend: gpu);
        Assert.True(model.UseGpu, "a Vulkan backend should enable the GPU blocks by default");

        const int frames = 1, latH = 16, latW = 16, nTxt = 24;
        var rng = new Random(7);
        var latent = new float[16 * frames * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);
        var text = new float[nTxt * HunyuanVideoModel.TextDim];
        for (int i = 0; i < text.Length; i++) text[i] = (float)((rng.NextDouble() * 2 - 1) * 0.5);
        var pooled = new float[768];
        for (int i = 0; i < pooled.Length; i++) pooled[i] = (float)(rng.NextDouble() * 2 - 1);

        var sw = Stopwatch.StartNew();
        var gpuOut = model.Forward(latent, 500f, text, frames, latH, latW, pooled, 6000f);
        double gpuFirst = sw.Elapsed.TotalSeconds;
        sw.Restart();
        gpuOut = model.Forward(latent, 500f, text, frames, latH, latW, pooled, 6000f);
        double gpuSecond = sw.Elapsed.TotalSeconds;

        model.UseGpu = false;
        sw.Restart();
        var cpuOut = model.Forward(latent, 500f, text, frames, latH, latW, pooled, 6000f);
        double cpuSec = sw.Elapsed.TotalSeconds;

        double dot = 0, nc = 0, ng = 0, diff = 0;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            Assert.True(float.IsFinite(gpuOut[i]), $"non-finite GPU output at {i}");
            dot += (double)cpuOut[i] * gpuOut[i]; nc += (double)cpuOut[i] * cpuOut[i]; ng += (double)gpuOut[i] * gpuOut[i];
            double e = cpuOut[i] - gpuOut[i]; diff += e * e;
        }
        double cos = dot / Math.Sqrt(nc * ng), relL2 = Math.Sqrt(diff / nc);
        _out.WriteLine($"GPU {gpuFirst:F1}s (first, incl. upload) / {gpuSecond:F1}s, CPU {cpuSec:F1}s; cosine {cos:F6}, rel L2 {relL2:E3}");
        Console.WriteLine($"[Hunyuan GPU parity] GPU {gpuFirst:F1}s first / {gpuSecond:F1}s, CPU {cpuSec:F1}s; cosine {cos:F6}, rel L2 {relL2:E3}");
        Assert.True(cos > 0.9999, $"cosine {cos}");
        Assert.True(relL2 < 1e-2, $"rel L2 {relL2}");
    }
}
