using System.Diagnostics;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real UMT5 text conditioning for Wan2.2's dual-model Low/High-Noise swap (docs/094 Phase 8).
/// <see cref="Wan22DualModelGenerateSmokeTests"/> already proved the dual-model swap runs
/// end-to-end, but only with all-zero text conditioning (`textContext`/`negativeTextContext` left
/// null, matching <c>WanPipeline.Generate</c>'s own documented fallback) -- despite passing a
/// real-looking prompt string, which is cosmetic only unless a real embedding is supplied. This is
/// the first Wan2.2 test to encode the prompt with the real UMT5-XXL encoder
/// (`models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors`, already present on this machine and
/// shared with Wan2.1 per its own release notes), matching the exact real-conditioning path
/// `ImageCommand.RunWan` uses for the CLI (fixed-length 226-token zero-padded embedding, found and
/// fixed 2026-09-14 per docs/081's own writeup -- reusing that exact convention here, not
/// reinventing it).
/// </summary>
public sealed class Wan22DualModelRealTextConditioningTests
{
    private readonly ITestOutputHelper _output;

    public Wan22DualModelRealTextConditioningTests(ITestOutputHelper output)
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

    private const int FixedTxtLen = 226;
    private const int TxtDim = 4096;

    private static float[] ZeroPadEmbedding(float[] raw)
    {
        var padded = new float[FixedTxtLen * TxtDim];
        Array.Copy(raw, padded, Math.Min(raw.Length, FixedTxtLen * TxtDim));
        return padded;
    }

    [Fact]
    public void Wan22_DualModel_Generate_RealUmt5TextConditioning_ProducesFiniteFrame()
    {
        string? lowNoisePath = FindModelPath("wan2.2_t2v_low_noise_14B_Q4_K_S.gguf");
        string? highNoisePath = FindModelPath("wan2.2_t2v_high_noise_14B_Q4_K_S.gguf");
        string? vaePath = FindModelPath("Wan2.1_VAE.safetensors");
        string? umt5EncoderPath = FindModelPath("models_t5_umt5-xxl-enc-bf16.safetensors");
        string? umt5TokenizerPath = FindModelPath("umt5-tokenizer.json");
        if (lowNoisePath is null || highNoisePath is null || vaePath is null
            || umt5EncoderPath is null || umt5TokenizerPath is null)
        {
            _output.WriteLine("[Wan22DualModelRealTextConditioningTests] Checkpoints missing, skipping.");
            Console.WriteLine("[Wan22DualModelRealTextConditioningTests] Checkpoints missing, skipping.");
            return;
        }

        var swEncode = Stopwatch.StartNew();
        var tokenizer = T5Tokenizer.FromFile(umt5TokenizerPath, maxLen: 512);
        using var umt5 = new UMT5Encoder(umt5EncoderPath);
        var condContext = ZeroPadEmbedding(umt5.Encode(tokenizer.Tokenize("a red apple on a wooden table")));
        var uncondContext = ZeroPadEmbedding(umt5.Encode(tokenizer.Tokenize("")));
        swEncode.Stop();
        string encMsg = $"[Wan2.2 real text cond] UMT5 encode (cond+uncond) took {swEncode.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(encMsg);
        Console.WriteLine(encMsg);

        Assert.All(condContext, v => Assert.True(float.IsFinite(v), "UMT5 cond embedding must be finite"));
        Assert.All(uncondContext, v => Assert.True(float.IsFinite(v), "UMT5 uncond embedding must be finite"));
        double condSumSq = 0;
        foreach (var v in condContext) condSumSq += (double)v * v;
        double condRms = Math.Sqrt(condSumSq / condContext.Length);
        _output.WriteLine($"[Wan2.2 real text cond] Real UMT5 cond-embedding RMS: {condRms:E6}");
        Console.WriteLine($"[Wan2.2 real text cond] Real UMT5 cond-embedding RMS: {condRms:E6}");
        Assert.True(condRms > 1e-6, $"UMT5 cond embedding RMS too small ({condRms}), likely degenerate/all-zero");

        using var pipeline = WanPipeline.Load(lowNoisePath, vaePath);
        using var highWeights = GgufWeightLoader.Open(highNoisePath);
        using var highModel = new WanModel(highWeights, prefix: "");

        string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "docs", "diffusion-samples", "wan22_a14b_dualmodel_realtext_2026-09-20.png");
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
            textContext: condContext,
            negativeTextContext: uncondContext,
            highNoiseTransformer: highModel,
            highNoiseBoundary: 0.5f);
        sw.Stop();
        string msg = $"[Wan2.2 real text cond] Generate (128x128, 1 frame, 2 steps, REAL UMT5 conditioning) took {sw.Elapsed.TotalSeconds:F1}s";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.Single(frames);
        Assert.All(frames[0], v => Assert.True(float.IsFinite(v), "VAE output must be finite"));
        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}
