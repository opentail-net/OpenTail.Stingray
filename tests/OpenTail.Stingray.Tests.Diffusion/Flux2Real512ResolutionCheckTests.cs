namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Tests the resolution-sensitivity hypothesis for FLUX.2's residual periodic grid/tiling artifact
/// (docs/087/088, found 2026-09-18 after the timestep-blindness fix): the real reference
/// (`examples/flux2/src/flux2/sampling.py`'s `limit_pixels = 1024**2`) expects ~1024x1024-scale
/// generation. Every real end-to-end run so far has been at 64x64 or 128x128 -- 64x-256x smaller in
/// pixel area than the model's expected regime, a far more extreme mismatch than LTX-Video's own
/// documented resolution-sensitivity precedent (which caused real corruption in the UNMODIFIED
/// official pipeline at 256x256 vs its ~720p+ trained regime). This test runs at 512x512/20-step,
/// the model's real production resolution class, to see whether the grid artifact is a genuine
/// remaining bug or an under-resolution artifact like LTX-Video's was.
/// </summary>
public sealed class Flux2Real512ResolutionCheckTests
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
    public void Flux2Pipeline_RealWeights_512Resolution20StepCheck()
    {
        string? ditPath = FindModelPath("flux2-dev-Q4_K_S.gguf");
        string? mistralPath = FindModelPath("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf");
        string? vaePath = FindModelPath("flux2-vae.safetensors");
        Assert.SkipUnless(ditPath != null && mistralPath != null && vaePath != null,
            "FLUX.2 DiT/Mistral/VAE checkpoints not all found");

        using var pipeline = OpenTail.Stingray.Diffusion.Flux2.Flux2Pipeline.Load(ditPath!, mistralPath!, vaePath!);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "flux2_512_20step_resolution_check_2026-09-18.png");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var rgb = pipeline.Generate(new OpenTail.Stingray.Diffusion.Flux2.Flux2GenerationRequest
        {
            Prompt = "a red apple on a wooden table",
            Width = 512,
            Height = 512,
            Steps = 20,
            Guidance = 3.5f,
            Seed = 42,
            OutputPath = outputPath,
        });
        sw.Stop();
        Console.WriteLine($"[Flux2 512x512/20step resolution check] Took {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(rgb.Length > 0);
        foreach (var v in rgb) Assert.True(float.IsFinite(v), "FLUX.2 end-to-end output contains NaN/Inf");
        Assert.True(File.Exists(outputPath));
    }
}
