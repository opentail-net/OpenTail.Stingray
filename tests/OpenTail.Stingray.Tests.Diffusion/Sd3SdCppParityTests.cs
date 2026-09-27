using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// SD3.5 Medium end to end (real CLIP-L + OpenCLIP-G + T5-XXL conditioning and MMDiT) against stable-diffusion.cpp
/// (<c>sd-cli --backend cpu</c>, the vendored <c>examples/stable-diffusion.cpp</c> with its git-ignored dump patches;
/// docs/102 #13). Fixture (<c>TestData/Sd3SdCppGolden</c>): the reference's initial noise [16, 32, 32] and its latent
/// after ONE Euler step (sigma 1 -> 0, so the combined velocity = noise - latent) for "a red apple on a wooden table",
/// 256², CFG 4.5 with an empty negative prompt, seed 42, real <c>sd3.5_medium-Q4_K_M.gguf</c>.
/// </summary>
public sealed class Sd3SdCppParityTests
{
    private static string? FindRepoPath(params string[] parts)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine([dir, .. parts]);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    // Measured 2026-09-27: CPU 0.998754 (rel L2 5.0%, norm ratio 0.9961) -- the conditioning's small encoder
    // differences (CLIP-G 0.996, T5 0.998) amplified by CFG 4.5.
    [Fact]
    public void OneStepCfgVelocity_MatchesSdCppCpu() => Run(backend: null, minCosine: 0.998);

    [Fact]
    public void OneStepCfgVelocity_MatchesSdCppCpu_OnVulkan()
    {
        OpenTail.Stingray.Vulkan.VulkanBackend? vk;
        try { vk = new OpenTail.Stingray.Vulkan.VulkanBackend(); }
        catch (Exception e) { Assert.Skip($"Vulkan unavailable: {e.Message}"); return; }
        using (vk) Run(vk, minCosine: VulkanMinCosine);
    }

    // Measured 2026-09-27: Vulkan 0.998767, same as CPU for one step (the 20-step CPU/GPU spread in STATUS accumulates over steps).
    private const double VulkanMinCosine = 0.998;

    private static void Run(OpenTail.Stingray.Core.IComputeBackend? backend, double minCosine)
    {
        string? mmdit = FindRepoPath("models", "_models", "sd3.5_medium-Q4_K_M.gguf");
        string? aux = FindRepoPath("models", "sd35-medium-aux");
        string? schnell = FindRepoPath("models", "flux1-schnell");
        string? golden = FindRepoPath("tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "Sd3SdCppGolden");
        Assert.SkipUnless(mmdit is not null && aux is not null && schnell is not null && golden is not null,
            "SD3.5 Medium checkpoint, models/sd35-medium-aux, models/flux1-schnell (T5 + tokenizers) or the sd.cpp fixture not found");

        string noisePath = Path.Combine(golden!, "noise.f32");
        var noise = ReadFloats(noisePath);
        var refLatent = ReadFloats(Path.Combine(golden!, "latent_1step_cfg45_cpu.f32"));
        string tmp = Path.Combine(Path.GetTempPath(), $"sd3_sdcpp_parity_{Environment.ProcessId}");
        Directory.CreateDirectory(tmp);
        string oursPath = Path.Combine(tmp, "latent.f32");

        Environment.SetEnvironmentVariable("STINGRAY_SD3_INJECT_NOISE_PATH", noisePath);
        Environment.SetEnvironmentVariable("STINGRAY_SD3_DUMP_LATENT_PATH", oursPath);
        try
        {
            using var pipe = Sd3Pipeline.LoadSeparate(
                Path.Combine(aux!, "text_encoder", "model.fp16.safetensors"),
                Path.Combine(aux!, "text_encoder_2", "model.fp16.safetensors"),
                mmdit!,
                Path.Combine(aux!, "vae", "diffusion_pytorch_model.safetensors"),
                Path.Combine(schnell!, "tokenizer_clip", "tokenizer.json"),
                backend: backend,
                t5EncoderPath: Path.Combine(schnell!, "t5xxl_fp8_e4m3fn.safetensors"),
                t5TokenizerPath: Path.Combine(schnell!, "tokenizer_t5", "tokenizer.json"));
            pipe.Generate("a red apple on a wooden table", negativePrompt: "", width: 256, height: 256, steps: 1,
                guidance: 4.5f, seed: 42, outputPath: Path.Combine(tmp, "out.png"));
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_SD3_INJECT_NOISE_PATH", null);
            Environment.SetEnvironmentVariable("STINGRAY_SD3_DUMP_LATENT_PATH", null);
        }

        var ours = ReadFloats(oursPath);
        Assert.Equal(refLatent.Length, ours.Length);
        double dot = 0, nr = 0, no = 0, diff = 0;
        for (int i = 0; i < noise.Length; i++)
        {
            double r = noise[i] - refLatent[i], o = noise[i] - ours[i];
            dot += r * o; nr += r * r; no += o * o; diff += (r - o) * (r - o);
        }
        double cos = dot / Math.Sqrt(nr * no), rel = Math.Sqrt(diff / nr), normRatio = Math.Sqrt(no / nr);
        Console.WriteLine($"[SD3.5 {(backend is null ? "CPU" : "Vulkan")} vs sd.cpp CPU] 1-step CFG 4.5 velocity cosine {cos:F6}, relative L2 {rel:F4}, norm ratio {normRatio:F4}");
        Assert.True(cos > minCosine, $"cosine {cos:F6} (min {minCosine})");
        Assert.InRange(normRatio, 0.97, 1.03);
    }
}
