using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Flux2;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real GPU-vs-CPU parity check for FLUX.2's SingleStreamBlock GPU execution:
/// compares unified hidden states after single block 0 between the real CPU path
/// (<see cref="Flux2DiT.ApplySingleBlockReal"/>) and the new GPU path
/// (<see cref="Flux2DiT.SingleBlockGpu"/>), using real weights from the checkpoint.
/// </summary>
public sealed class Flux2SingleBlockGpuParityTests
{
    private const string ModelFileName = "flux2-dev-Q4_K_S.gguf";

    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", ModelFileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void SingleBlockGpu_MatchesCpuReference_RealWeights()
    {
        string? modelPath = FindModelPath();
        Assert.SkipUnless(modelPath != null, "models/_models/flux2-dev-Q4_K_S.gguf not found");

        using var backend = TryCreateVulkan();
        if (backend is null) return;

        using var weights = GgufWeightLoader.Open(modelPath!);
        var p = new Flux2Params();
        using var dit = new Flux2DiT(weights, p);

        int d = p.HiddenSize;
        int headDim = p.HeadDim;
        int mlpHidden = (int)(d * p.MlpRatio);

        // Small synthetic token counts
        const int patchH = 4, patchW = 4;
        int nImg = patchH * patchW; // 16
        const int nTxt = 8;
        int nSeq = nTxt + nImg;

        var rng = new Random(4242);
        var unifiedCpu = new float[nSeq * d];
        for (int i = 0; i < unifiedCpu.Length; i++) unifiedCpu[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] vec = dit.ComputeModulationVec(timestep: 0.5f, pooledEmbed: [], guidance: 3.5f);
        float[][] modSingle = dit.ComputeModulation("single_stream_modulation.lin.weight", vec, d, 3);

        var txtPositions = new int[nTxt * 4];
        for (int i = 0; i < nTxt; i++) txtPositions[i * 4 + 3] = i;

        var imgPositions = new int[nImg * 4];
        int idx = 0;
        for (int y = 0; y < patchH; y++)
            for (int x = 0; x < patchW; x++)
            {
                imgPositions[idx * 4 + 1] = y;
                imgPositions[idx * 4 + 2] = x;
                idx++;
            }

        var (txtCos, txtSin) = Flux2RoPE.BuildContextFreqs(txtPositions, nTxt, p.AxesDim, p.Theta);
        var (imgCos, imgSin) = Flux2RoPE.BuildContextFreqs(imgPositions, nImg, p.AxesDim, p.Theta);

        var unifiedCos = new float[nSeq * headDim];
        var unifiedSin = new float[nSeq * headDim];
        txtCos.AsSpan().CopyTo(unifiedCos.AsSpan(0, nTxt * headDim));
        txtSin.AsSpan().CopyTo(unifiedSin.AsSpan(0, nTxt * headDim));
        imgCos.AsSpan().CopyTo(unifiedCos.AsSpan(nTxt * headDim, nImg * headDim));
        imgSin.AsSpan().CopyTo(unifiedSin.AsSpan(nTxt * headDim, nImg * headDim));

        // --- 1. CPU Reference ---
        var expectedCpu = (float[])unifiedCpu.Clone();
        dit.ApplySingleBlockReal(0, expectedCpu, modSingle, unifiedCos, unifiedSin, nSeq);

        // --- 2. GPU Path ---
        Func<string, float[]> getWeight = weights.ReadF32;
        using var gpuWeights = new Flux2GpuWeights(backend, getWeight, p, includeSingleBlocks: false);

        var combinedPositions = new int[nSeq * 4];
        Array.Copy(txtPositions, 0, combinedPositions, 0, txtPositions.Length);
        Array.Copy(imgPositions, 0, combinedPositions, txtPositions.Length, imgPositions.Length);
        var (ropeCosCompact, ropeSinCompact) = Flux2RoPE.BuildContextFreqsCompact(combinedPositions, nSeq, p.AxesDim, p.Theta);

        using var ws = new Flux2GpuWorkspace(backend, nImg, nTxt, d, mlpHidden, p.HeadDim, ropeCosCompact, ropeSinCompact);

        var visionOps = (IVisionOpsBackend)backend;
        var imageOps = (IImageOpsBackend)backend;

        // Compute single modulation
        var siluVec = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(siluVec);
        using var siluVecGpu = backend.Upload(siluVec, TensorShape.D1(d));
        dit.ComputeSharedSingleModulationGpu(visionOps, ws, gpuWeights, siluVecGpu);

        // Load single block 0 on demand (streaming path)
        using var bw = gpuWeights.LoadSingleBlock(0);

        // Upload unified state into ws.Unified
        ws.SetUnified(backend, unifiedCpu);

        // Run single block
        dit.SingleBlockGpu(ws, bw, visionOps, imageOps);
        backend.Synchronize();

        var actualGpu = new float[nSeq * d];
        backend.Download(ws.Unified, actualGpu);

        // Cosine similarity check
        double dot = 0, normRef = 0, normGpu = 0;
        for (int i = 0; i < expectedCpu.Length; i++)
        {
            dot += expectedCpu[i] * actualGpu[i];
            normRef += expectedCpu[i] * expectedCpu[i];
            normGpu += actualGpu[i] * actualGpu[i];
        }
        double cosine = dot / (Math.Sqrt(normRef) * Math.Sqrt(normGpu));
        Assert.True(cosine > 0.999, $"SingleBlock cosine similarity was {cosine:F7}, expected > 0.999");
    }
}
