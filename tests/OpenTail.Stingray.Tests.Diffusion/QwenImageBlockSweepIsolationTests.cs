using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Phase 2b of docs/094's follow-up investigation (2026-09-20): is block 29 uniquely bad, or does
/// every block show ~this much intrinsic CPU/GPU disagreement when isolated and fed identical real
/// input? <see cref="QwenImageBlock29Fp32IsolationTests"/> found a SINGLE block, given a correct
/// real input, already diverges from CPU by cosine 0.92-0.97 -- comparable to or worse than the full
/// 60-block model's own end-to-end cosine (0.990109). This test answers whether that's special to
/// block 29 or generic to the architecture, by running the identical single-block isolation
/// (production FP16 precision only -- FP32 was already found to change nothing) across a spread of
/// blocks: 10, 20 (early), 28, 29, 30 (local to the whole-model bisection's own onset point), and 50
/// (late). Captures every block's CPU state in ONE forward pass (via <see cref="QwenImageModel.OnBlockStateCpu"/>)
/// rather than re-running the CPU trajectory per target block, since the isolated GPU test itself is
/// the only per-block cost that matters.
///
/// Reports cosine (as before) PLUS relative L2 error (||GPU-CPU|| / ||CPU||) and RMSE, per an
/// external second-opinion review's point that cosine alone can be a misleading single metric on
/// rows dominated by a huge common-magnitude residual component.
/// </summary>
public sealed class QwenImageBlockSweepIsolationTests
{
    private static readonly int[] TargetBlocks = { 10, 20, 28, 29, 30, 50 };

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

    private static (double cosine, double relL2, double rmse, double maxDiff) Compare(float[] cpu, float[] gpu)
    {
        double dot = 0, na = 0, nb = 0, sqDiff = 0, maxDiff = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            double diff = cpu[i] - gpu[i];
            double absDiff = Math.Abs(diff);
            if (absDiff > maxDiff) maxDiff = absDiff;
            sqDiff += diff * diff;
            dot += (double)cpu[i] * gpu[i];
            na += (double)cpu[i] * cpu[i];
            nb += (double)gpu[i] * gpu[i];
        }
        double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
        double relL2 = Math.Sqrt(sqDiff) / (Math.Sqrt(na) + 1e-30);
        double rmse = Math.Sqrt(sqDiff / cpu.Length);
        return (cosine, relL2, rmse, maxDiff);
    }

    [Fact]
    public void SweepMultipleBlocks_FindsWhetherBlock29IsUniquelyAnomalous()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        if (modelPath is null)
        {
            Console.WriteLine("[BlockSweep] Checkpoint missing, skipping.");
            return;
        }

        const int latH = 32, latW = 32, latC = 16;
        var rng = new Random(42);
        var latent = new float[latC * latH * latW];
        for (int i = 0; i < latent.Length - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            latent[i] = (float)(radius * Math.Cos(theta));
            latent[i + 1] = (float)(radius * Math.Sin(theta));
        }

        int seqLen = 8;
        var textContext = new float[seqLen * QwenImageModel.ContextDim];
        for (int i = 0; i < textContext.Length; i++) textContext[i] = (float)(rng.NextDouble() * 2 - 1) * 0.02f;

        const int patchH = latH / QwenImageModel.PatchSize, patchW = latW / QwenImageModel.PatchSize;
        const float timestep = 500f;

        // Capture EVERY block's post-block [img,txt] state in one CPU pass -- state[b] is the
        // output of block b (= the input to block b+1).
        var stateImg = new float[60][];
        var stateTxt = new float[60][];

        using var cpuWeights = GgufWeightLoader.Open(modelPath);
        using var cpuModel = new QwenImageModel(cpuWeights);
        cpuModel.OnBlockStateCpu = (b, img, txt) =>
        {
            stateImg[b] = (float[])img.Clone();
            stateTxt[b] = (float[])txt.Clone();
        };

        Console.WriteLine("[BlockSweep] Running full CPU forward, capturing all 60 blocks' state...");
        cpuModel.Forward(latent, timestep, textContext, latH, latW);

        using var vulkan = new VulkanBackend();
        using var gpuModel = new QwenImageModel(GgufWeightLoader.Open(modelPath), backend: vulkan);

        Console.WriteLine("[BlockSweep] Block | ImgCosine | ImgRelL2 | ImgRMSE | ImgMaxDiff | TxtCosine | TxtRelL2");
        var results = new List<(int block, double imgCos, double imgRelL2, double txtCos, double txtRelL2)>();

        foreach (int target in TargetBlocks)
        {
            var imgIn = stateImg[target - 1];
            var txtIn = stateTxt[target - 1];
            int numImgTokens = imgIn.Length / QwenImageModel.HiddenDim;
            int numTxtTokens = txtIn.Length / QwenImageModel.HiddenDim;

            var (gpuImgOut, gpuTxtOut) = gpuModel.RunSingleBlockGpuForTest(
                target, imgIn, txtIn, timestep, numImgTokens, numTxtTokens, patchH, patchW, forceFp32ForThisBlock: false);

            var (imgCos, imgRelL2, imgRmse, imgMaxDiff) = Compare(stateImg[target], gpuImgOut);
            var (txtCos, txtRelL2, txtRmse, txtMaxDiff) = Compare(stateTxt[target], gpuTxtOut);

            results.Add((target, imgCos, imgRelL2, txtCos, txtRelL2));
            Console.WriteLine($"[BlockSweep] {target,5} | {imgCos:F6} | {imgRelL2:F6} | {imgRmse:E3} | {imgMaxDiff:E3} | {txtCos:F6} | {txtRelL2:F6}");
        }

        // Real interpretation, printed for the record rather than asserted (this is a diagnostic
        // test, not a correctness gate): is block 29 an outlier relative to the other five, or is
        // ~this much intrinsic per-block divergence typical?
        double meanOtherCos = 0; int otherCount = 0;
        double block29Cos = 0;
        foreach (var r in results)
        {
            if (r.block == 29) block29Cos = r.imgCos;
            else { meanOtherCos += r.imgCos; otherCount++; }
        }
        meanOtherCos /= Math.Max(1, otherCount);
        Console.WriteLine($"[BlockSweep] CONCLUSION: block 29 img cosine={block29Cos:F6} vs mean of other 5 blocks={meanOtherCos:F6} (diff={meanOtherCos - block29Cos:F6}). " +
            (Math.Abs(meanOtherCos - block29Cos) > 0.02
                ? "Block 29 looks like a real outlier -- worth a block-29-specific investigation."
                : "Block 29 looks TYPICAL of intrinsic per-block CPU/GPU divergence -- the whole-model bisection's 'onset at block 29' was likely a threshold artifact of the cumulative trajectory, not a block-29-specific defect. Point investigation at a systemic cause (attention algorithm, GEMM reduction order) instead."));
    }
}
