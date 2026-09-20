using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Phase 2 of docs/094's follow-up investigation (2026-09-20, jointly designed with an external
/// second-opinion review): the single-block cross-injection experiment. Freezes the REAL CPU
/// trajectory's [img,txt] state entering block 29 (the bisection's own measured divergence-onset
/// block on a realistic-scale input) and runs ONLY block 29 in isolation on both backends, fed the
/// identical input -- rather than each backend accumulating its own drift over 29 blocks first.
/// This isolates block 29's OWN numerical behavior from everything upstream of it.
///
/// Two lanes, both against the CPU's own block-29 output (computed from the SAME frozen input) as
/// the reference:
/// A) Normal GPU block 29 (FP16 weights, the production precision)
/// B) GPU block 29 with FP32 weights for that block ONLY (isolates weight-precision specifically;
///    deliberately does NOT touch the resident 60-layer weight set -- see
///    <see cref="QwenImageModel.RunSingleBlockGpuForTest"/>'s own doc comment on why, given this
///    model's ~41GB full-resident memory footprint observed to cause real RAM pressure on this
///    machine during a full end-to-end run).
///
/// If (A) already agrees closely with CPU, block 29 in isolation is not the divergence source
/// (contradicting the whole-model bisection's own onset-block finding, which would be a real,
/// interesting result in itself -- meaning the "divergence at block 29" signal is actually about
/// STATE ACCUMULATED INTO block 29's input, not block 29's own computation). If (A) diverges but
/// (B) [FP32] closes most of the gap, that's decisive evidence for FP16 weight-precision as (at
/// least a major component of) the real cause. If (B) does NOT close the gap, FP16 weights are
/// ruled out and the next candidate (attention kernel reduction order, per Phase 3) becomes the
/// leading hypothesis.
/// </summary>
public sealed class QwenImageBlock29Fp32IsolationTests
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

    private static double CosineAndMaxDiff(float[] a, float[] b, out double maxDiff)
    {
        double dot = 0, na = 0, nb = 0;
        maxDiff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs(a[i] - b[i]);
            if (diff > maxDiff) maxDiff = diff;
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
    }

    [Fact]
    public void Block29_Fp16VsFp32VsCpu_Isolated()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        if (modelPath is null)
        {
            Console.WriteLine("[Block29Isolation] Checkpoint missing, skipping.");
            return;
        }

        // Same real-scale realistic input the whole-model bisection used (unit-Gaussian latent,
        // mid-trajectory timestep=500) -- matches the run that found onset at block 29.
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

        // Step 1: run the CPU trajectory once, freezing [img,txt] entering block 29 (= block 28's
        // output) and also capturing block 29's own real CPU output as the reference.
        float[]? imgIntoBlock29 = null, txtIntoBlock29 = null;
        float[]? cpuBlock29ImgOut = null, cpuBlock29TxtOut = null;

        using var cpuWeights = GgufWeightLoader.Open(modelPath);
        using var cpuModel = new QwenImageModel(cpuWeights);
        cpuModel.OnBlockStateCpu = (b, img, txt) =>
        {
            if (b == BlockIndex - 1)
            {
                imgIntoBlock29 = (float[])img.Clone();
                txtIntoBlock29 = (float[])txt.Clone();
            }
            else if (b == BlockIndex)
            {
                cpuBlock29ImgOut = (float[])img.Clone();
                cpuBlock29TxtOut = (float[])txt.Clone();
            }
        };

        Console.WriteLine("[Block29Isolation] Running full CPU forward to capture block 28->29 state...");
        cpuModel.Forward(latent, timestep, textContext, latH, latW);

        Assert.NotNull(imgIntoBlock29);
        Assert.NotNull(txtIntoBlock29);
        Assert.NotNull(cpuBlock29ImgOut);
        Assert.NotNull(cpuBlock29TxtOut);

        int numImgTokens = imgIntoBlock29!.Length / QwenImageModel.HiddenDim;
        int numTxtTokens = txtIntoBlock29!.Length / QwenImageModel.HiddenDim;

        // Step 2 (control): re-run block 29 on CPU alone from the frozen input, via the isolated
        // single-block entry point, to confirm it reproduces the trajectory run's own real output
        // bit-for-bit (sanity check that isolation itself introduces no discrepancy before drawing
        // any GPU conclusion).
        //
        // REAL BUG FOUND AND FIXED HERE, 2026-09-20 (docs/094 Phase 2 follow-up): this call used to
        // pass imgIntoBlock29/txtIntoBlock29 DIRECTLY. QwenImageModel.TransformerBlock mutates its
        // img/txt arguments IN PLACE (ApplyGatedResidual writes the residual directly into the
        // passed array) and returns those SAME references -- so this "control" call was silently
        // corrupting imgIntoBlock29/txtIntoBlock29 into block 29's OWN OUTPUT before the GPU lanes
        // below ever saw them as input. Every GPU lane then computed block 29 on an
        // already-post-block-29 state, producing a large, consistent (both lanes equally wrong, so
        // FP32-vs-FP16 looked identical) but entirely spurious divergence that had nothing to do
        // with the GPU. Caught via QwenImageBlock29ReproDiagnosticTests, which reproduced the SAME
        // computation without this ordering bug and got a dramatically different (small-divergence)
        // result. Fixed by passing CLONES to this CPU-only control call, leaving the originals
        // intact for the GPU lanes.
        var (cpuIsoImg, cpuIsoTxt) = cpuModel.RunSingleBlockCpuForTest(BlockIndex, (float[])imgIntoBlock29.Clone(), (float[])txtIntoBlock29.Clone(), timestep, patchH, patchW);
        double cpuSelfCos = CosineAndMaxDiff(cpuBlock29ImgOut!, cpuIsoImg, out double cpuSelfMaxDiff);
        Console.WriteLine($"[Block29Isolation] CPU isolation self-check: cosine={cpuSelfCos:F6} maxDiff={cpuSelfMaxDiff:E4} (should be ~exact)");

        // Step 3: GPU lanes, fed the IDENTICAL frozen [img,txt] state.
        using var gpuWeights = GgufWeightLoader.Open(modelPath);
        using var vulkan = new VulkanBackend();
        using var gpuModelForTest = new QwenImageModel(gpuWeights, backend: vulkan);

        Console.WriteLine("[Block29Isolation] Lane A: GPU block 29, FP16 weights (production precision)...");
        var (gpuFp16Img, gpuFp16Txt) = gpuModelForTest.RunSingleBlockGpuForTest(
            BlockIndex, imgIntoBlock29, txtIntoBlock29, timestep, numImgTokens, numTxtTokens, patchH, patchW,
            forceFp32ForThisBlock: false);

        Console.WriteLine("[Block29Isolation] Lane B: GPU block 29, FP32 weights (this block only)...");
        var (gpuFp32Img, gpuFp32Txt) = gpuModelForTest.RunSingleBlockGpuForTest(
            BlockIndex, imgIntoBlock29, txtIntoBlock29, timestep, numImgTokens, numTxtTokens, patchH, patchW,
            forceFp32ForThisBlock: true);

        double cosFp16Img = CosineAndMaxDiff(cpuBlock29ImgOut!, gpuFp16Img, out double maxDiffFp16Img);
        double cosFp16Txt = CosineAndMaxDiff(cpuBlock29TxtOut!, gpuFp16Txt, out double maxDiffFp16Txt);
        double cosFp32Img = CosineAndMaxDiff(cpuBlock29ImgOut!, gpuFp32Img, out double maxDiffFp32Img);
        double cosFp32Txt = CosineAndMaxDiff(cpuBlock29TxtOut!, gpuFp32Txt, out double maxDiffFp32Txt);

        Console.WriteLine($"[Block29Isolation] Lane A (FP16) vs CPU: img cosine={cosFp16Img:F6} maxDiff={maxDiffFp16Img:E4} | txt cosine={cosFp16Txt:F6} maxDiff={maxDiffFp16Txt:E4}");
        Console.WriteLine($"[Block29Isolation] Lane B (FP32) vs CPU: img cosine={cosFp32Img:F6} maxDiff={maxDiffFp32Img:E4} | txt cosine={cosFp32Txt:F6} maxDiff={maxDiffFp32Txt:E4}");

        double imgImprovementRatio = (1.0 - cosFp16Img) / Math.Max(1e-12, 1.0 - cosFp32Img);
        Console.WriteLine($"[Block29Isolation] Img cosine-gap improvement ratio (FP16 gap / FP32 gap) = {imgImprovementRatio:F2}x -- >>1 means FP32 closes most of the gap (weight precision is the cause); ~1 means FP32 changes nothing (weight precision is NOT the cause).");
    }
}
