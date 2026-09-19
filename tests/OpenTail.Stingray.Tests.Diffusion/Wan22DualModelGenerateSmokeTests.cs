using System.Diagnostics;
using OpenTail.Stingray.Diffusion.Wan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real end-to-end <see cref="WanPipeline.Generate"/> run exercising Wan2.2-A14B's actual
/// dual-model low/high-noise swap (docs/094 Phase 8) -- the real next step after
/// <see cref="Wan22DualModelRealWeightsTests"/> verified each transformer loads and forward-passes
/// correctly in isolation. Uses Wan2.1's own VAE checkpoint (`Wan2.1_VAE.safetensors`) -- per Wan's
/// real public release notes, the T2V-A14B variant shares Wan2.1's 16-channel/8x-spatial VAE (only
/// the separate TI2V-5B variant uses a different, higher-compression VAE), so this is expected to
/// be compatible, not verified against a written spec inside this repo.
/// </summary>
public sealed class Wan22DualModelGenerateSmokeTests
{
    private readonly ITestOutputHelper _output;

    public Wan22DualModelGenerateSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var pNested = Path.Combine(dir, "models", "wan2.1", fileName);
            if (File.Exists(pNested)) return pNested;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Wan22_DualModel_Generate_SmallSmoke_ProducesFiniteFrame()
    {
        string? lowNoisePath = FindModelPath("wan2.2_t2v_low_noise_14B_Q4_K_S.gguf");
        string? highNoisePath = FindModelPath("wan2.2_t2v_high_noise_14B_Q4_K_S.gguf");
        string? vaePath = FindModelPath("Wan2.1_VAE.safetensors");
        if (lowNoisePath is null || highNoisePath is null || vaePath is null)
        {
            _output.WriteLine("[Wan22DualModelGenerateSmokeTests] Checkpoints missing, skipping.");
            Console.WriteLine("[Wan22DualModelGenerateSmokeTests] Checkpoints missing, skipping.");
            return;
        }

        using var pipeline = WanPipeline.Load(lowNoisePath, vaePath);
        using var highWeights = GgufWeightLoader.Open(highNoisePath);
        using var highModel = new WanModel(highWeights, prefix: "");

        string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "docs", "diffusion-samples", "wan22_a14b_dualmodel_smoke_2026-09-19.png");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        var sw = Stopwatch.StartNew();
        var frames = pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 128,
            height: 128,
            numFrames: 1,
            steps: 2,
            guidance: 6.0f,
            seed: 42,
            outputPath: outputPath,
            highNoiseTransformer: highModel,
            highNoiseBoundary: 0.5f);
        sw.Stop();
        string msg = $"[Wan2.2 dual-model] Generate (128x128, 1 frame, 2 steps, zero-conditioning) took {sw.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.Single(frames);
        Assert.All(frames[0], v => Assert.True(float.IsFinite(v), "VAE output must be finite"));
        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}
