using OpenTail.Stingray.Diffusion.HunyuanVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class HunyuanVideoRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors";
    private const string VaeFileName = "hunyuan_video_vae_bf16.safetensors";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void HunyuanVideo_RealModelFile_LoadsAndExposesTensors()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var loader = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(loader);
        Assert.True(loader.TensorCount > 0, "HunyuanVideo safetensors must contain tensors");
    }

    [Fact]
    public void HunyuanVideoPipeline_LoadRealModel_ExecutesForwardPass()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");
        string? vaePath = FindModelPath(VaeFileName);

        using var pipeline = HunyuanVideoPipeline.Load(modelPath, vaePath);
        Assert.NotNull(pipeline);
        Assert.Equal("HunyuanVideo", pipeline.Architecture);

        // Generate a 1-frame 32x32 1-step test pass
        var frames = pipeline.Generate(
            prompt: "A cinematic hyperrealistic video of waves crashing against dramatic cliffs",
            width: 32,
            height: 32,
            numFrames: 1,
            steps: 1,
            guidance: 1.0f,
            seed: 42);

        Assert.NotNull(frames);
        Assert.NotEmpty(frames);
        Assert.Single(frames);
        Assert.Equal(32 * 32 * 3, frames[0].Length);

        // Real VAE decode must produce finite, non-degenerate pixel output (not all-zero/NaN).
        var frame = frames[0];
        Assert.All(frame, v => Assert.True(float.IsFinite(v), "VAE output must be finite"));
        double sumSq = 0;
        foreach (var v in frame) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / frame.Length);
        Assert.True(rms > 1e-4, $"VAE output RMS too small ({rms}), likely degenerate/all-zero");
    }
}
