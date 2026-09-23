using OpenTail.Stingray.Diffusion.Flux2;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real end-to-end FLUX.2 generation with the DiT's 8 double-stream blocks running on real
/// Vulkan GPU (docs/091, wired 2026-09-19 via <see cref="Flux2Pipeline.Load"/>'s new
/// <c>ditBackend</c> parameter) -- the 48 single-stream blocks and everything else still run on
/// CPU (memory-budget reasons, docs/091). This is a correctness/wiring smoke test: confirms GPU is
/// now a real, selectable, exercised code path in the actual generation pipeline, not just an
/// isolated unit-level parity/benchmark test. Real, measured finding on this project's own dev
/// iGPU is that CPU currently wins end-to-end (docs/091/PerformanceLeague.md) -- this test proves
/// the GPU path is wired and correct so it can be measured and improved further.
/// </summary>
public sealed class Flux2GpuWiredEndToEndTests
{
    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
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
    public void Flux2Pipeline_GpuDoubleBlocks_EndToEndProducesFiniteImage()
    {
        string? ditPath = FindModelPath("flux2-dev-Q4_K_S.gguf");
        string? mistralPath = FindModelPath("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf");
        string? vaePath = FindModelPath("flux2-vae.safetensors");
        Assert.SkipUnless(ditPath != null && mistralPath != null && vaePath != null,
            "FLUX.2 DiT/Mistral/VAE checkpoints not all found");

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        Environment.SetEnvironmentVariable("STINGRAY_FLUX2_GPU_SINGLE_BLOCKS", "1");
        try
        {
            using var pipeline = Flux2Pipeline.Load(ditPath!, mistralPath!, vaePath!, ditBackend: vulkan);

            string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "flux2_gpu_wired_smoke_2026-09-19.png");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var rgb = pipeline.Generate(new Flux2GenerationRequest
            {
                Prompt = "a red apple on a wooden table",
                Width = 128,
                Height = 128,
                Steps = 4,
                Guidance = 3.5f,
                Seed = 42,
                OutputPath = outputPath,
            });
            sw.Stop();
            Console.WriteLine($"[Flux2 GPU-wired E2E] Took {sw.Elapsed.TotalSeconds:F1}s");

            Assert.True(rgb.Length > 0);
            foreach (var v in rgb) Assert.True(float.IsFinite(v), "FLUX.2 GPU-wired end-to-end output contains NaN/Inf");
            Assert.True(File.Exists(outputPath));
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_FLUX2_GPU_SINGLE_BLOCKS", null);
        }
    }
}
