using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight GPU-vs-CPU parity check for <see cref="QwenImageModel.ForwardGpu"/>
/// (docs/094 Phase 2's first GPU port for this model). Same discipline as every other model's
/// first GPU parity test in this codebase (e.g. <c>Sd3BaselineTests.TestSd35GpuVsCpuParity</c>,
/// <c>FluxGpuVsCpuForwardBisectDebugTest</c>) -- run this and inspect the real cosine/maxDiff
/// numbers BEFORE trusting any timing number for this GPU path.
/// </summary>
public sealed class QwenImageGpuParityTests
{
    private readonly ITestOutputHelper _output;

    public QwenImageGpuParityTests(ITestOutputHelper output)
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
    public void TestQwenImageGpuVsCpuParity()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        if (modelPath is null)
        {
            _output.WriteLine("[QwenImageGpuParityTests] Checkpoint missing, skipping.");
            return;
        }

        using var cpuWeights = GgufWeightLoader.Open(modelPath);
        using var cpuModel = new QwenImageModel(cpuWeights);
        using var vulkan = new VulkanBackend();
        using var gpuWeights = GgufWeightLoader.Open(modelPath);
        using var gpuModel = new QwenImageModel(gpuWeights, backend: vulkan);

        const int latH = 32, latW = 32, latC = 16;
        var rng = new Random(42);
        var latent = new float[latC * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);

        int seqLen = 8;
        var textContext = new float[seqLen * QwenImageModel.ContextDim];
        for (int i = 0; i < textContext.Length; i++) textContext[i] = (float)(rng.NextDouble() * 0.1);

        _output.WriteLine("[QwenImageParity] Running GPU forward pass...");
        var swGpu = Stopwatch.StartNew();
        var gpuOut = gpuModel.Forward(latent, 1000f, textContext, latH, latW);
        swGpu.Stop();
        string msgGpu = $"[QwenImageParity] GPU forward took {swGpu.ElapsedMilliseconds} ms ({swGpu.Elapsed.TotalSeconds:F2}s)";
        _output.WriteLine(msgGpu);
        Console.WriteLine(msgGpu);

        _output.WriteLine("[QwenImageParity] Running CPU forward pass...");
        var swCpu = Stopwatch.StartNew();
        var cpuOut = cpuModel.Forward(latent, 1000f, textContext, latH, latW);
        swCpu.Stop();
        string msgCpu = $"[QwenImageParity] CPU forward took {swCpu.ElapsedMilliseconds} ms ({swCpu.Elapsed.TotalSeconds:F2}s)";
        _output.WriteLine(msgCpu);
        Console.WriteLine(msgCpu);

        double dot = 0, normA = 0, normB = 0;
        float maxDiff = 0;
        for (int i = 0; i < gpuOut.Length; i++)
        {
            float diff = MathF.Abs(gpuOut[i] - cpuOut[i]);
            if (diff > maxDiff) maxDiff = diff;
            dot += gpuOut[i] * cpuOut[i];
            normA += gpuOut[i] * gpuOut[i];
            normB += cpuOut[i] * cpuOut[i];
        }
        double cosSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        string msgResult = $"[QwenImageParity] Cosine: {cosSim:F6}, MaxDiff: {maxDiff:F6}";
        _output.WriteLine(msgResult);
        Console.WriteLine(msgResult);

        Assert.True(cosSim > 0.99, $"Expected cosine similarity > 0.99, got {cosSim:F6}");
    }
}
