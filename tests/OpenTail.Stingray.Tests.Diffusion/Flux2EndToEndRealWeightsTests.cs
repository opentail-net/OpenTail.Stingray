namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real end-to-end FLUX.2 generation attempt (docs/087): real DiT + real Mistral-24B text
/// conditioning + real VAE decode, wired together via <see cref="OpenTail.Stingray.Diffusion.Flux2.Flux2Pipeline.Load"/>.
/// Small resolution and few steps (this is a correctness/wiring smoke test, not a quality or
/// performance benchmark -- CPU-only DiT alone is ~68s/step at 512x512, so this test uses a
/// small size and few steps to stay a reasonable smoke-test duration).
/// </summary>
public sealed class Flux2EndToEndRealWeightsTests
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

    [Fact]
    public void Flux2Pipeline_RealWeights_EndToEndGenerateProducesFiniteImage()
    {
        string? ditPath = FindModelPath("flux2-dev-Q4_K_S.gguf");
        string? mistralPath = FindModelPath("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf");
        string? vaePath = FindModelPath("flux2-vae.safetensors");
        Assert.SkipUnless(ditPath != null && mistralPath != null && vaePath != null,
            "FLUX.2 DiT/Mistral/VAE checkpoints not all found");

        using var pipeline = OpenTail.Stingray.Diffusion.Flux2.Flux2Pipeline.Load(ditPath!, mistralPath!, vaePath!);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "flux2_first_e2e_smoke_2026-09-18.png");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var rgb = pipeline.Generate(new OpenTail.Stingray.Diffusion.Flux2.Flux2GenerationRequest
        {
            Prompt = "a red apple on a wooden table",
            Width = 64,
            Height = 64,
            Steps = 2,
            Guidance = 3.5f,
            Seed = 42,
            OutputPath = outputPath,
        });
        sw.Stop();
        Console.WriteLine($"[Flux2 E2E smoke] Took {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(rgb.Length > 0);
        foreach (var v in rgb) Assert.True(float.IsFinite(v), "FLUX.2 end-to-end output contains NaN/Inf");
        Assert.True(File.Exists(outputPath));
    }

    /// <summary>
    /// Moderate real quality check (docs/088): 128x128/4-step CPU run, enough real denoising steps
    /// to get a genuine visual signal (not just a wiring smoke test) without the hours a full
    /// 512x512/20-step CPU run would take -- the user's real requirement is a Vulkan GPU run
    /// (FLUX.2 has no GPU-residency wiring yet, a separate large task), this is a cheap interim
    /// data point on whether the real conditioning/DiT/VAE wiring produces coherent structure.
    /// </summary>
    [Fact]
    public void Flux2Pipeline_RealWeights_ModerateQualityCheck()
    {
        string? ditPath = FindModelPath("flux2-dev-Q4_K_S.gguf");
        string? mistralPath = FindModelPath("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf");
        string? vaePath = FindModelPath("flux2-vae.safetensors");
        Assert.SkipUnless(ditPath != null && mistralPath != null && vaePath != null,
            "FLUX.2 DiT/Mistral/VAE checkpoints not all found");

        using var pipeline = OpenTail.Stingray.Diffusion.Flux2.Flux2Pipeline.Load(ditPath!, mistralPath!, vaePath!);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "flux2_quality_check_128_4step_2026-09-18.png");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var rgb = pipeline.Generate(new OpenTail.Stingray.Diffusion.Flux2.Flux2GenerationRequest
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
        Console.WriteLine($"[Flux2 quality check 128x128/4step] Took {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(rgb.Length > 0);
        foreach (var v in rgb) Assert.True(float.IsFinite(v), "FLUX.2 quality-check output contains NaN/Inf");
        Assert.True(File.Exists(outputPath));
    }
}
