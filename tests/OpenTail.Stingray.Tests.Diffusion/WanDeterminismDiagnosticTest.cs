using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Checks whether WanModel.Forward (CPU) and ForwardGpu are actually deterministic for identical
/// inputs -- a real, previously-untried category (docs/081, update #14's remaining candidates):
/// a race condition or nondeterministic parallel-reduction/dispatch-ordering bug could produce
/// exactly the "plausible magnitude, no coherent structure" symptom observed all session if the
/// same "clean" latent produces a DIFFERENT prediction on repeat calls.
/// </summary>
public sealed class WanDeterminismDiagnosticTest
{
    private static string? FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    [Fact]
    public void Forward_Cpu_SameInputTwice_ProducesIdenticalOutput()
    {
        string? ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        if (ditPath is null || !File.Exists(ditPath)) { Console.WriteLine("Checkpoint missing, skipping."); return; }

        var loader1 = SafetensorsLoader.Open(ditPath);
        using var model = new WanModel(loader1, "");

        int latH = 32, latW = 32, latC = 16, numFrames = 1;
        var latent = new float[latC * numFrames * latH * latW];
        var rng = new Random(7);
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var textCtx = new float[226 * 4096];
        for (int i = 0; i < 4096; i++) textCtx[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        var out1 = model.Forward((float[])latent.Clone(), 500f, textCtx, numFrames, latH, latW);
        var out2 = model.Forward((float[])latent.Clone(), 500f, textCtx, numFrames, latH, latW);

        float maxDiff = 0f;
        for (int i = 0; i < out1.Length; i++)
        {
            float d = MathF.Abs(out1[i] - out2[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Console.WriteLine($"[Determinism/CPU] maxDiff between two identical-input calls = {maxDiff:E6}");
        Assert.True(maxDiff < 1e-4f, $"CPU Forward is NOT deterministic: maxDiff={maxDiff}");
    }

    [Fact]
    public void ForwardGpu_SameInputTwice_ProducesIdenticalOutput()
    {
        string? ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        if (ditPath is null || !File.Exists(ditPath)) { Console.WriteLine("Checkpoint missing, skipping."); return; }

        using var vulkan = new VulkanBackend();
        var loader = SafetensorsLoader.Open(ditPath);
        using var model = new WanModel(loader, "", backend: vulkan);
        using var gpuWeights = model.GetOrCreateGpuWeights();

        int latH = 32, latW = 32, latC = 16, numFrames = 1;
        int patchH = latH / 2, patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;
        int numTxt = 226;

        var (ropeCos, ropeSin) = WanRoPE.Compute3DRoPECompact(numFrames, patchH, patchW, headDim: 128);
        using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, 1536, 8960, 30, numTxt, ropeCos, ropeSin, headDim: 128);

        var latent = new float[latC * numFrames * latH * latW];
        var rng = new Random(7);
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var textCtx = new float[numTxt * 4096];
        for (int i = 0; i < 4096; i++) textCtx[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        model.PrecomputeCrossKvCacheGpu(textCtx, gpuWs, gpuWeights, vulkan);
        var out1 = model.ForwardGpu((float[])latent.Clone(), 500f, textCtx, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);
        var out2 = model.ForwardGpu((float[])latent.Clone(), 500f, textCtx, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);

        float maxDiff = 0f;
        for (int i = 0; i < out1.Length; i++)
        {
            float d = MathF.Abs(out1[i] - out2[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Console.WriteLine($"[Determinism/GPU] maxDiff between two identical-input calls = {maxDiff:E6}");
        Assert.True(maxDiff < 1e-3f, $"GPU ForwardGpu is NOT deterministic: maxDiff={maxDiff}");
    }
}
