using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LtxVideoRealWeightsTests
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void LtxVideo_RealModelFile_LoadsAndExposesPipeline()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(loader);
        Assert.True(loader.TensorCount > 0, "LTX-Video safetensors must contain tensors");

        using var pipeline = LtxVideoPipeline.Load(modelPath);
        Assert.NotNull(pipeline);
        Assert.Equal("LTX-Video", pipeline.Architecture);
    }

    /// <summary>Checks `LtxVideoModel.DetectConfig` against the real v0.9.1 checkpoint's own tensor
    /// shapes (verified directly against the safetensors JSON header --
    /// docs/055-ltx-video-implementation-plan.md's tensor inventory).</summary>
    [Fact]
    public void LtxVideo_RealModelFile_DetectConfigMatchesKnownArchitecture()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        Assert.Equal(128, model.InChannels);
        Assert.Equal(128, model.OutChannels);
        Assert.Equal(2048, model.HiddenSize);
        Assert.Equal(32, model.NumHeads);
        Assert.Equal(64, model.HeadDim);
        Assert.Equal(28, model.NumLayers);
        Assert.Equal(2048, model.CrossAttentionDim);
        Assert.Equal(4096, model.CaptionChannels);
        Assert.False(model.CrossAttentionAdaln);
        Assert.False(model.SelfAttentionGated);
        Assert.False(model.CrossAttentionGated);
    }

    private static string? FindRepoFile(string relativePath)
    {
        string direct = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", relativePath.Replace('/', Path.DirectorySeparatorChar));
        if (File.Exists(direct) || Directory.Exists(direct)) return direct;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void GenerateApple_20Steps_LtxVideo_Cpu()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? textEncoderDir = FindRepoFile("models/ltx-t5/text_encoder");
        string? tokenizerJsonPath = FindRepoFile("models/ltx-t5/tokenizer/tokenizer.json");
        if (modelPath is null || textEncoderDir is null || tokenizerJsonPath is null) return;

        using var pipeline = LtxVideoPipeline.Load(
            modelPath,
            textEncoderDir: textEncoderDir,
            tokenizerJsonPath: tokenizerJsonPath);

        string outputPath = Path.Combine(@"C:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "ltx_video_apple_padded128_20step.png");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        pipeline.Generate(new OpenTail.Stingray.Diffusion.ImageGenerationRequest
        {
            Prompt = "a red apple on a wooden table",
            Width = 512,
            Height = 512,
            Steps = 20,
            Guidance = 3.0f,
            Seed = 42,
            OutputPath = outputPath
        });
        sw.Stop();
        string msg = $"[LtxVideo CPU 20-step] Took {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F1}s)";
        Console.WriteLine(msg);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}

