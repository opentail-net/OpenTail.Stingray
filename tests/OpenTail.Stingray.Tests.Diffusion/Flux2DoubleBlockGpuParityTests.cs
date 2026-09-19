using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Flux2;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real GPU-vs-CPU parity check for FLUX.2's double-block-only GPU residency (docs/091, 2026-09-19):
/// compares img/txt hidden states after all 8 double blocks between the real CPU path
/// (<see cref="Flux2DiT.ApplyDoubleBlockReal"/>, looped 8 times) and the new GPU path
/// (<see cref="Flux2DiT.DoubleBlockGpu"/>, looped 8 times over <see cref="Flux2GpuWeights"/> with
/// <c>includeSingleBlocks: false</c>), using the real checkpoint's own weights. Uses a small
/// synthetic token count (not production resolution) to keep this test fast -- this verifies
/// correctness of the double-block math itself, not end-to-end timing (that's a separate,
/// not-yet-done step per docs/091's implementation order).
/// </summary>
public sealed class Flux2DoubleBlockGpuParityTests
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
    public void DoubleBlockGpu_MatchesCpuReference_RealWeights()
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

        // Small synthetic token counts -- correctness check, not a production-scale timing run.
        const int patchH = 4, patchW = 4;
        int nImg = patchH * patchW; // 16
        const int nTxt = 8;
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

        // Real conditioning vector (synthetic timestep/pooled -- FLUX.2 has no CLIP pooled input,
        // VecInDim=0, so pooledEmbed is unused when real weights are present per ComputeVec/
        // ComputeModulationVec's own doc comments).
        float[] vec = dit.ComputeModulationVec(timestep: 0.5f, pooledEmbed: [], guidance: 3.5f);
        float[][] modImg = dit.ComputeModulation("double_stream_modulation_img.lin.weight", vec, d, 6);
        float[][] modTxt = dit.ComputeModulation("double_stream_modulation_txt.lin.weight", vec, d, 6);

        // Real 4-axis position scheme: text tokens at (0,0,0,l=sequential), image tokens at
        // (0,h,w,0) -- matching Flux2Pipeline.Generate's own real construction.
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

        // ── CPU reference: loop the real per-block method 8 times ──
        var imgAfterCpu = (float[])imgCpu.Clone();
        var txtAfterCpu = (float[])txtCpu.Clone();
        for (int layer = 0; layer < p.DepthDoubleBlocks; layer++)
        {
            dit.ApplyDoubleBlockReal(layer, imgAfterCpu, txtAfterCpu, modImg, modTxt, imgCos, imgSin, txtCos, txtSin, nImg, nTxt);
        }

        // ── GPU: same math, real weights, double-blocks-only residency ──
        Func<string, float[]> getWeight = weights.ReadF32;
        using var gpuWeights = new Flux2GpuWeights(backend, getWeight, p, includeSingleBlocks: false);

        // Combined [txt, img] compact RoPE table (both streams rotated in FLUX.2 -- see
        // Flux2DiT.DoubleBlockGpu's own doc comment).
        var combinedPositions = new int[nSeq * 4];
        Array.Copy(txtPositions, 0, combinedPositions, 0, txtPositions.Length);
        Array.Copy(imgPositions, 0, combinedPositions, txtPositions.Length, imgPositions.Length);
        var (ropeCosCompact, ropeSinCompact) = Flux2RoPE.BuildContextFreqsCompact(combinedPositions, nSeq, p.AxesDim, p.Theta);

        using var ws = new Flux2GpuWorkspace(backend, nImg, nTxt, d, mlpHidden, headDim, ropeCosCompact, ropeSinCompact,
            initialImgHidden: imgCpu, initialTxtHidden: txtCpu);

        // SiLU(vec) computed on CPU (cheap for a single d-length vector -- see
        // ComputeSharedDoubleModulationGpu's own doc comment for why this doesn't need a shader).
        var siluVec = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(siluVec);
        var siluVecGpu = backend.Upload(siluVec, TensorShape.D1(d));

        try
        {
            dit.ComputeSharedDoubleModulationGpu(backend, ws, gpuWeights, siluVecGpu);

            for (int layer = 0; layer < p.DepthDoubleBlocks; layer++)
            {
                dit.DoubleBlockGpu(ws, gpuWeights.DoubleBlocks[layer], backend, backend);
            }

            var imgAfterGpu = new float[nImg * d];
            var txtAfterGpu = new float[nTxt * d];
            backend.Download(ws.ImgHidden, imgAfterGpu);
            backend.Download(ws.TxtHidden, txtAfterGpu);

            double imgMaxDiff = 0, txtMaxDiff = 0;
            for (int i = 0; i < imgAfterCpu.Length; i++)
                imgMaxDiff = Math.Max(imgMaxDiff, Math.Abs(imgAfterCpu[i] - imgAfterGpu[i]));
            for (int i = 0; i < txtAfterCpu.Length; i++)
                txtMaxDiff = Math.Max(txtMaxDiff, Math.Abs(txtAfterCpu[i] - txtAfterGpu[i]));

            double imgCos2 = CosineSim(imgAfterCpu, imgAfterGpu);
            double txtCos2 = CosineSim(txtAfterCpu, txtAfterGpu);

            Console.WriteLine($"[Flux2 DoubleBlockGpu parity] img: maxDiff={imgMaxDiff:E4} cosine={imgCos2:F7} | txt: maxDiff={txtMaxDiff:E4} cosine={txtCos2:F7}");

            Assert.True(imgCos2 > 0.999, $"img hidden state diverges: cosine={imgCos2}, maxDiff={imgMaxDiff}");
            Assert.True(txtCos2 > 0.999, $"txt hidden state diverges: cosine={txtCos2}, maxDiff={txtMaxDiff}");
        }
        finally
        {
            backend.Free(siluVecGpu);
        }
    }

    private static double CosineSim(float[] a, float[] b)
    {
        double dot = 0, sa = 0, sb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            sa += (double)a[i] * a[i];
            sb += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(sa) * Math.Sqrt(sb) + 1e-12);
    }
}
