using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Flux2;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real, production-scale GPU-vs-CPU timing comparison for FLUX.2's double-block loop (docs/091's
/// step 7, 2026-09-19). The correctness check (<see cref="Flux2DoubleBlockGpuParityTests"/>) used a
/// tiny synthetic scale (16 image + 8 text tokens) deliberately for speed -- this test measures
/// whether the GPU path is actually faster at a realistic 512x512-equivalent scale (1024 image
/// tokens, 256 text tokens), which is a completely separate question from correctness.
///
/// Per CLAUDE.md rule 13: this machine's GPU is an integrated Radeon (Ryzen 5700G) sharing system
/// RAM with the CPU, not a discrete GPU with its own VRAM/bandwidth advantage -- do NOT assume the
/// GPU wins. A real precedent (MiniMax-Music3's DiT) measured this same iGPU 2.5x SLOWER than CPU
/// due to per-call dispatch overhead. This test's job is to measure, not to confirm an assumption.
/// </summary>
public sealed class Flux2DoubleBlockGpuBenchmarkTests
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
    public void DoubleBlockLoop_ProductionScale_GpuVsCpuTiming()
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

        // Production-scale: 512x512 image at 16x16 patch = 32x32 = 1024 image tokens; a realistic
        // Mistral-conditioning text token count.
        const int patchH = 32, patchW = 32;
        int nImg = patchH * patchW; // 1024
        const int nTxt = 256;
        int nSeq = nTxt + nImg;

        var rng = new Random(2091);
        float[] RandArray(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            return a;
        }

        var imgCpu = RandArray(nImg * d);
        var txtCpu = RandArray(nTxt * d);

        float[] vec = dit.ComputeModulationVec(timestep: 0.5f, pooledEmbed: [], guidance: 3.5f);
        float[][] modImg = dit.ComputeModulation("double_stream_modulation_img.lin.weight", vec, d, 6);
        float[][] modTxt = dit.ComputeModulation("double_stream_modulation_txt.lin.weight", vec, d, 6);

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

        // ── CPU timing: best-of-3 full 8-block loops ──
        double cpuBestMs = double.MaxValue;
        for (int trial = 0; trial < 3; trial++)
        {
            var imgT = (float[])imgCpu.Clone();
            var txtT = (float[])txtCpu.Clone();
            var sw = Stopwatch.StartNew();
            for (int layer = 0; layer < p.DepthDoubleBlocks; layer++)
                dit.ApplyDoubleBlockReal(layer, imgT, txtT, modImg, modTxt, imgCos, imgSin, txtCos, txtSin, nImg, nTxt);
            sw.Stop();
            cpuBestMs = Math.Min(cpuBestMs, sw.Elapsed.TotalMilliseconds);
        }

        // ── GPU timing: weight upload cost measured SEPARATELY from the per-step compute loop --
        // real question is whether upload is a one-time cost amortized across a multi-step
        // denoising loop, or paid every call. ──
        var uploadSw = Stopwatch.StartNew();
        using var gpuWeights = new Flux2GpuWeights(backend, weights.ReadF32, p, includeSingleBlocks: false);
        uploadSw.Stop();

        var combinedPositions = new int[nSeq * 4];
        Array.Copy(txtPositions, 0, combinedPositions, 0, txtPositions.Length);
        Array.Copy(imgPositions, 0, combinedPositions, txtPositions.Length, imgPositions.Length);
        var (ropeCosCompact, ropeSinCompact) = Flux2RoPE.BuildContextFreqsCompact(combinedPositions, nSeq, p.AxesDim, p.Theta);

        using var ws = new Flux2GpuWorkspace(backend, nImg, nTxt, d, mlpHidden, headDim, ropeCosCompact, ropeSinCompact,
            initialImgHidden: imgCpu, initialTxtHidden: txtCpu);

        var siluVec = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(siluVec);
        var siluVecGpu = backend.Upload(siluVec, TensorShape.D1(d));

        double gpuBestMs = double.MaxValue;
        try
        {
            // Warm-up call (first dispatch pays one-time pipeline-compile cost).
            dit.ComputeSharedDoubleModulationGpu(backend, ws, gpuWeights, siluVecGpu);
            for (int layer = 0; layer < p.DepthDoubleBlocks; layer++)
                dit.DoubleBlockGpu(ws, gpuWeights.DoubleBlocks[layer], backend, backend);
            backend.Synchronize();

            for (int trial = 0; trial < 3; trial++)
            {
                var sw = Stopwatch.StartNew();
                dit.ComputeSharedDoubleModulationGpu(backend, ws, gpuWeights, siluVecGpu);
                for (int layer = 0; layer < p.DepthDoubleBlocks; layer++)
                    dit.DoubleBlockGpu(ws, gpuWeights.DoubleBlocks[layer], backend, backend);
                backend.Synchronize();
                sw.Stop();
                gpuBestMs = Math.Min(gpuBestMs, sw.Elapsed.TotalMilliseconds);
            }
        }
        finally
        {
            backend.Free(siluVecGpu);
        }

        double ratio = gpuBestMs / cpuBestMs;
        Console.WriteLine($"[Flux2 DoubleBlock benchmark] nImg={nImg} nTxt={nTxt} | CPU best={cpuBestMs:F1}ms | GPU best={gpuBestMs:F1}ms (weight upload {uploadSw.Elapsed.TotalMilliseconds:F1}ms, NOT included in per-step GPU timing above) | ratio={ratio:F2}x {(ratio < 1 ? "(GPU faster)" : "(CPU faster)")}");

        // No pass/fail assertion on the ratio itself -- this test's job is to MEASURE and report,
        // per CLAUDE.md rule 13 ("do not assume GPU wins"). Only assert the measurement itself is
        // sane (both times positive and finite).
        Assert.True(cpuBestMs > 0 && double.IsFinite(cpuBestMs));
        Assert.True(gpuBestMs > 0 && double.IsFinite(gpuBestMs));
    }
}
