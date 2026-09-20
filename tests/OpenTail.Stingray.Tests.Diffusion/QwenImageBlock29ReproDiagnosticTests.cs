using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic-only test (docs/094 Phase 2 follow-up, 2026-09-20): two supposedly-identical
/// block-29 single-block isolation experiments produced wildly different numbers --
/// <see cref="QwenImageBlock29Fp32IsolationTests"/> found cosine 0.968982/maxDiff 1.47e7, while
/// <see cref="QwenImageBlockSweepIsolationTests"/> found cosine 0.999997/maxDiff 2.36e5 for the SAME
/// block, SAME seed, SAME input generation code. That is a real, unexplained ~30x discrepancy that
/// must be resolved before trusting either number. The two tests differ in exactly one structural
/// way: the isolation test builds a FRESH VulkanBackend/QwenImageModel/GgufWeightLoader dedicated to
/// GPU-only work and calls RunSingleBlockGpuForTest twice on it (Lane A then Lane B); the sweep test
/// builds ONE GPU model/backend and reuses it across six sequential single-block calls. This test
/// runs BOTH patterns in the SAME process against the SAME CPU-captured reference, to isolate
/// whether the discrepancy is: (a) fresh-vs-reused GPU model/backend instance state, (b) call-order
/// (first-ever dispatch on a backend vs a later one), or (c) genuine GPU non-determinism between
/// runs of the identical computation.
/// </summary>
public sealed class QwenImageBlock29ReproDiagnosticTests
{
    private const int BlockIndex = 29;

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

    private static (double cosine, double maxDiff) Compare(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs(a[i] - b[i]);
            if (diff > maxDiff) maxDiff = diff;
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30), maxDiff);
    }

    [Fact]
    public void Block29_FreshVsReusedGpuInstance_SameCpuReference()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        if (modelPath is null)
        {
            Console.WriteLine("[Block29Repro] Checkpoint missing, skipping.");
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

        float[]? imgIn29 = null, txtIn29 = null, cpuOut29Img = null, cpuOut29Txt = null;
        using var cpuWeights = GgufWeightLoader.Open(modelPath);
        using var cpuModel = new QwenImageModel(cpuWeights);
        cpuModel.OnBlockStateCpu = (b, img, txt) =>
        {
            if (b == BlockIndex - 1) { imgIn29 = (float[])img.Clone(); txtIn29 = (float[])txt.Clone(); }
            else if (b == BlockIndex) { cpuOut29Img = (float[])img.Clone(); cpuOut29Txt = (float[])txt.Clone(); }
        };
        Console.WriteLine("[Block29Repro] Running CPU forward...");
        cpuModel.Forward(latent, timestep, textContext, latH, latW);
        Assert.NotNull(imgIn29); Assert.NotNull(cpuOut29Img);

        int numImgTokens = imgIn29!.Length / QwenImageModel.HiddenDim;
        int numTxtTokens = txtIn29!.Length / QwenImageModel.HiddenDim;

        // Pattern 1: REUSED instance, called TWICE in a row (mirrors QwenImageBlock29Fp32IsolationTests).
        using (var vulkanA = new VulkanBackend())
        using (var gpuModelA = new QwenImageModel(GgufWeightLoader.Open(modelPath), backend: vulkanA))
        {
            var (out1Img, _) = gpuModelA.RunSingleBlockGpuForTest(BlockIndex, imgIn29, txtIn29, timestep, numImgTokens, numTxtTokens, patchH, patchW, forceFp32ForThisBlock: false);
            var (cos1, maxDiff1) = Compare(cpuOut29Img!, out1Img);
            Console.WriteLine($"[Block29Repro] Pattern1-Call1 (fresh instance, 1st call): cosine={cos1:F6} maxDiff={maxDiff1:E4}");

            var (out2Img, _) = gpuModelA.RunSingleBlockGpuForTest(BlockIndex, imgIn29, txtIn29, timestep, numImgTokens, numTxtTokens, patchH, patchW, forceFp32ForThisBlock: false);
            var (cos2, maxDiff2) = Compare(cpuOut29Img!, out2Img);
            Console.WriteLine($"[Block29Repro] Pattern1-Call2 (SAME instance, 2nd call, same inputs): cosine={cos2:F6} maxDiff={maxDiff2:E4}");

            var (callSelfCos, callSelfMaxDiff) = Compare(out1Img, out2Img);
            Console.WriteLine($"[Block29Repro] Pattern1 Call1-vs-Call2 (both GPU, same instance, should be identical if deterministic): cosine={callSelfCos:F6} maxDiff={callSelfMaxDiff:E4}");
        }

        // Pattern 2: BRAND NEW instance created fresh for a single call (isolates "first-ever
        // dispatch on a freshly constructed backend" as a variable).
        using (var vulkanB = new VulkanBackend())
        using (var gpuModelB = new QwenImageModel(GgufWeightLoader.Open(modelPath), backend: vulkanB))
        {
            var (out3Img, _) = gpuModelB.RunSingleBlockGpuForTest(BlockIndex, imgIn29, txtIn29, timestep, numImgTokens, numTxtTokens, patchH, patchW, forceFp32ForThisBlock: false);
            var (cos3, maxDiff3) = Compare(cpuOut29Img!, out3Img);
            Console.WriteLine($"[Block29Repro] Pattern2 (brand-new instance, single call): cosine={cos3:F6} maxDiff={maxDiff3:E4}");
        }

        Console.WriteLine("[Block29Repro] If all cosines above agree closely (~same value), the discrepancy versus the earlier two tests is NOT instance-reuse or call-order -- points to a real bug in one of those two test files' own comparison logic, or genuine run-to-run GPU non-determinism (compare against the two prior tests' logged numbers by re-running them after this).");
    }
}
