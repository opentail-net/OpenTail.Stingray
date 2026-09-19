using System.Diagnostics;
using OpenTail.Stingray.Diffusion.QwenImage;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real end-to-end GPU generation for Qwen Image (docs/094 Phase 2) -- settles whether
/// <see cref="QwenImageModel.ForwardGpu"/>'s marginal parity number (cosine 0.990, maxDiff 0.218,
/// see <see cref="QwenImageGpuParityTests"/>) actually produces a coherent image or not. Small
/// resolution/step count deliberately (this is a real-weights smoke test, not a quality bar) --
/// zero-conditioning like <c>QwenImageRealWeightsTests</c>'s own CPU equivalent, not a real prompt,
/// so a real text encoder pass isn't required.
/// </summary>
public sealed class QwenImageGpuEndToEndSmokeTests
{
    private readonly ITestOutputHelper _output;

    public QwenImageGpuEndToEndSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\_models\{fileName}",
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }
        return null;
    }

    [Fact]
    public void QwenImagePipeline_Gpu_SmallSmoke_ProducesImage()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        string? vaePath = FindModelPath("qwen_image_vae.safetensors");
        if (modelPath is null || vaePath is null)
        {
            _output.WriteLine("[QwenImageGpuEndToEndSmokeTests] Checkpoints missing, skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();
        using var pipeline = QwenImagePipeline.Load(modelPath, vaePath, backend: vulkan);

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "docs", "diffusion-samples", "qwenimage_gpu_smoke_2026-09-19.png");

        var sw = Stopwatch.StartNew();
        pipeline.Generate(
            prompt: "a red apple on a wooden table",
            negativePrompt: "",
            width: 128,
            height: 128,
            steps: 4,
            guidance: 2.5f,
            seed: 42,
            outputPath: outputPath);
        sw.Stop();

        string msg = $"[QwenImageGpuSmoke] Generate (4 steps 128x128, zero-conditioning) took {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F1}s)";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.True(File.Exists(outputPath), $"Expected output file at {outputPath}");
    }
}
