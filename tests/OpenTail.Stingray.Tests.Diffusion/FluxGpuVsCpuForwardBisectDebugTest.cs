using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real-weights numerical parity check: <see cref="FluxDiT.ForwardGpu"/> (GPU-resident path) vs
/// <see cref="FluxDiT.Forward"/> (CPU, known-correct — has produced real coherent FLUX.1-schnell
/// images in this session) for a single denoising step. Uses a deliberately small synthetic token
/// count (nImg=16, nTxt=8) so it runs in ~1 minute (mostly checkpoint load) rather than the
/// ~160s/step a real 512x512 generation costs, while still exercising the real weights and the
/// full 19 double + 38 single block graph.
///
/// This test caught a real bug on 2026-09-13: the GPU-resident path's shared <c>VisionGelu</c>
/// shader collapsed to all-NaN output because FLUX's MLP activations reach |v|~17, past a range
/// where this GPU driver's tanh() overflows exp(2x) to Inf and returns Inf/Inf=NaN — fixed by
/// clamping the tanh argument in <c>Shaders.VisionGelu</c> (a mathematical no-op; tanh already
/// saturates to +-1 well before |arg|=20). See <see cref="FluxDiT.DebugHook"/> for the
/// per-stage instrumentation used to bisect it — left in place (zero overhead when unset) since
/// it proved useful for narrowing a whole-buffer-NaN regression to one shader in ~5 iterations.
/// </summary>
public sealed class FluxGpuVsCpuForwardBisectDebugTest
{
    private static string? FindRepoRoot()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "docs"))) return dir;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindFluxCheckpoint()
    {
        var repoRoot = FindRepoRoot();
        if (repoRoot is null) return null;
        foreach (var c in new[]
        {
            "models/flux1-schnell/flux1-schnell-Q4_K_S.gguf",
            "models/flux1-schnell-q4_k.gguf",
        })
        {
            var full = Path.Combine(repoRoot, c);
            if (File.Exists(full)) return full;
        }
        return null;
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void ForwardGpu_MatchesForwardCpu_SingleStep_RealWeights()
    {
        var modelPath = FindFluxCheckpoint();
        if (modelPath is null) return; // no local checkpoint -- skip silently, consistent with this repo's RealWeights convention

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return; // no GPU on this machine -- skip

        var model = GgufModel.Open(modelPath);
        var p = FluxParams.FromMetadata(model.Metadata);

        const int heightPatches = 4, widthPatches = 4; // nImg = 16 (tiny, fast)
        const int nTxt = 8;
        int nImg = heightPatches * widthPatches;
        int d = p.HiddenSize;

        var imgIds = Flux2DRoPE.ImagePatchIds(heightPatches, widthPatches);
        var txtIds = new int[nTxt * 2]; // all zero, matches real FLUX convention

        var rng = new Random(2026_09_13);
        float[] txtEmbeds = new float[nTxt * p.ContextDim];
        for (int i = 0; i < txtEmbeds.Length; i++) txtEmbeds[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        float[] pooledEmbed = new float[p.VecDim];
        for (int i = 0; i < pooledEmbed.Length; i++) pooledEmbed[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        float[] noise = new float[nImg * p.InChannels];
        for (int i = 0; i < noise.Length; i++) noise[i] = (float)(rng.NextDouble() * 2 - 1);

        const float timestep = 0.5f;
        const float guidance = 3.5f;

        // ── CPU reference (known-correct baseline) ──
        var cpuBackend = new CpuBackend();
        using var cpuDit = new FluxDiT(model, p, cpuBackend);
        float[] velCpu = cpuDit.Forward(noise, imgIds, txtEmbeds, txtIds, pooledEmbed, timestep, guidance);

        // ── GPU-resident path under test ──
        using var gpuDit = new FluxDiT(model, p, vulkan);
        var weights = gpuDit.GetOrCreateGpuWeights();
        var visionOps = (IVisionOpsBackend)vulkan;
        var imageOps = (IImageOpsBackend)vulkan;

        int nSeq = nTxt + nImg;
        var allIds = new int[nSeq * 2];
        imgIds.CopyTo(allIds, nTxt * 2);
        var (ropeC, ropeS) = Flux2DRoPE.BuildFreqs(allIds, nSeq, p.HeadDim);
        var (imgRopeC, imgRopeS) = Flux2DRoPE.BuildFreqs(imgIds, nImg, p.HeadDim);

        using var ws = new FluxGpuWorkspace(vulkan, nSeq, nImg, nTxt, d, imgRopeC, imgRopeS, ropeC, ropeS);
        using var initNoiseGpu = vulkan.Upload(noise, TensorShape.D2(nImg, p.InChannels), exact: true);
        imageOps.ScaleInPlace(ws.Latent, 0f);
        vulkan.AddInPlace(ws.Latent, initNoiseGpu);

        gpuDit.PrecomputeTxtGpu(txtEmbeds, ws, weights, visionOps);
        var swGpu = System.Diagnostics.Stopwatch.StartNew();
        var velGpuTensor = gpuDit.ForwardGpu(ws.Latent, txtEmbeds, pooledEmbed, timestep, guidance, ws, weights, visionOps);
        swGpu.Stop();
        Console.WriteLine($"[FluxParityTest] ForwardGpu (57 blocks) took {swGpu.ElapsedMilliseconds} ms ({swGpu.Elapsed.TotalSeconds:F2}s)");

        float[] velGpu = new float[nImg * p.OutChannels];
        vulkan.Download(velGpuTensor, velGpu);

        for (int i = 0; i < velCpu.Length; i++)
        {
            Assert.False(float.IsNaN(velGpu[i]) || float.IsInfinity(velGpu[i]), $"velGpu[{i}] is NaN/Inf");
            // FP16-weight GPU path vs FP32 CPU path: real precision gap, not a bug. 5e-2 absolute
            // tolerance observed empirically (max diff ~5.5e-3 on values of magnitude ~1-1.5).
            Assert.True(MathF.Abs(velCpu[i] - velGpu[i]) < 5e-2f,
                $"velCpu[{i}]={velCpu[i]} vs velGpu[{i}]={velGpu[i]}, diff={MathF.Abs(velCpu[i] - velGpu[i]):E4}");
        }
    }
}
