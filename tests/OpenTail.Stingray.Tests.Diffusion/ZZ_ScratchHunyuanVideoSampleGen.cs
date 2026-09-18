using OpenTail.Stingray.Diffusion.HunyuanVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class ZZ_ScratchHunyuanVideoSampleGen
{
    [Fact]
    public void GenerateRealSample_256x256()
    {
        string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\hunyuanvideo\hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors";
        string vaePath = @"C:\Git-Public\OpenTail.Stingray\models\hunyuanvideo\hunyuan_video_vae_bf16.safetensors";
        if (!File.Exists(modelPath) || !File.Exists(vaePath)) return;

        using var pipeline = HunyuanVideoPipeline.Load(modelPath, vaePath);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var frames = pipeline.Generate(
            prompt: "A cinematic hyperrealistic photo of a red apple on a white table",
            width: 256,
            height: 256,
            numFrames: 1,
            steps: 4,
            guidance: 1.0f,
            seed: 42);
        sw.Stop();
        Console.Error.WriteLine($"[HunyuanVideo sample] generated in {sw.Elapsed.TotalSeconds:F1}s");

        Assert.Single(frames);
        var frame = frames[0];
        Assert.All(frame, v => Assert.True(float.IsFinite(v)));

        string outDir = @"C:\Git-Public\OpenTail.Stingray\docs\diffusion-samples";
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "hunyuanvideo_red-apple-on-white-table_256x256_4steps_zero-cond_2026-09-18.png");

        // HunyuanVaeDecoder3D returns raw [-1,1] CHW pixels (standard diffusers VAE convention,
        // no internal rescale) -- PngWriter expects [0,1] CHW, so rescale here.
        var rescaled = new float[frame.Length];
        for (int i = 0; i < frame.Length; i++)
            rescaled[i] = Math.Clamp((frame[i] + 1.0f) * 0.5f, 0f, 1f);

        OpenTail.Stingray.Diffusion.PngWriter.Write(outPath, rescaled, 256, 256);
        Console.Error.WriteLine($"[HunyuanVideo sample] saved to {outPath}");
    }
}
