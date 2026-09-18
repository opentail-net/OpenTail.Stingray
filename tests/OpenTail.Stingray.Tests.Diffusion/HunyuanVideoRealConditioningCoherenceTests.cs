using OpenTail.Stingray.Diffusion.HunyuanVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real end-to-end HunyuanVideo generation with REAL LLaMA-3-8B text conditioning (docs/088) --
/// the first time all three real components (DiT, LLaMA text encoder, VAE) run together. Compares
/// against the earlier zero-conditioning sample
/// (`hunyuanvideo_red-apple-on-white-table_256x256_4steps_zero-cond_2026-09-18.png`).
/// </summary>
public sealed class HunyuanVideoRealConditioningCoherenceTests
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
    public void HunyuanVideoPipeline_RealTextConditioning_256_4Step_CoherenceCheck()
    {
        string? modelPath = FindModelPath("hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors");
        string? vaePath = FindModelPath("hunyuan_video_vae_bf16.safetensors");
        string? textEncoderPath = FindModelPath("llava-llama-3-8b-v1_1-int4.gguf");
        if (modelPath is null || vaePath is null || textEncoderPath is null) return;

        using var pipeline = HunyuanVideoPipeline.Load(modelPath, textEncoderPath, vaePath);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var frames = pipeline.Generate(
            prompt: "A cinematic hyperrealistic photo of a red apple on a white table",
            width: 256,
            height: 256,
            numFrames: 1,
            steps: 4,
            guidance: 6.0f,
            seed: 42);
        sw.Stop();
        Console.Error.WriteLine($"[HunyuanVideo real conditioning 256x256/4step] Took {sw.Elapsed.TotalSeconds:F1}s");

        Assert.Single(frames);
        var frame = frames[0];
        Assert.All(frame, v => Assert.True(float.IsFinite(v), "VAE output must be finite"));

        double sumSq = 0;
        foreach (var v in frame) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / frame.Length);
        Assert.True(rms > 1e-4, $"VAE output RMS too small ({rms}), likely degenerate/all-zero");

        string outDir = Path.Combine(Directory.GetCurrentDirectory(), "docs", "diffusion-samples");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "hunyuanvideo_real_conditioning_256_4step_2026-09-18.png");

        var rescaled = new float[frame.Length];
        for (int i = 0; i < frame.Length; i++)
            rescaled[i] = Math.Clamp((frame[i] + 1.0f) * 0.5f, 0f, 1f);

        PngWriter.Write(outPath, rescaled, 256, 256);
        Console.Error.WriteLine($"[HunyuanVideo real conditioning] saved to {outPath}");
    }
}
