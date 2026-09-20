using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First-ever correctness test for <c>VulkanBackend.AdaLNModulateDual</c> (FLUX.2's fused
/// dual-stream AdaLN shader) -- no test in this codebase exercised it before this one. Written
/// specifically to verify a real fix made alongside the standalone <c>AdaLNModulate</c> shader's
/// own precision bug (docs/094 Phase 2 follow-up, 2026-09-20, <see cref="QwenImageAdaLnPrecisionTests"/>):
/// this dual shader's <c>isRmsNorm: false</c> branch used the numerically-unstable one-pass
/// <c>variance = E[x^2] - mean^2</c> formula (an even more direct cancellation risk than the
/// standalone shader's original bug, since it subtracts two similarly-huge float32 values to
/// recover a small variance), converted here to a genuine two-pass mean-then-variance. Covers both
/// the RMSNorm and LayerNorm branches, at both ordinary and huge-magnitude/tiny-relative-variance
/// scale, against the real CPU reference (<see cref="DiffusionOps.AdaLNModulate"/>) and a float64
/// ground truth.
/// </summary>
public sealed class Flux2AdaLNModulateDualGpuCpuParityTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static void RunCase(string label, bool isRmsNorm, int nTxt, int nImg, int dim, float[] txtIn, float[] imgIn, float eps = 1e-6f)
    {
        using var backend = TryCreateVulkan();
        if (backend is null)
        {
            Console.WriteLine($"[Flux2AdaLNModulateDual:{label}] No Vulkan backend, skipping.");
            return;
        }

        var shiftA = new float[dim];
        var scaleA = new float[dim];
        var shiftB = new float[dim];
        var scaleB = new float[dim];

        var cpuTxtOut = new float[nTxt * dim];
        var cpuImgOut = new float[nImg * dim];
        DiffusionOps.AdaLNModulate(cpuTxtOut, txtIn, shiftA, scaleA, nTxt, dim, isRmsNorm, eps);
        DiffusionOps.AdaLNModulate(cpuImgOut, imgIn, shiftB, scaleB, nImg, dim, isRmsNorm, eps);

        var inTxtGpu = backend.Upload(txtIn, TensorShape.D2(nTxt, dim));
        var inImgGpu = backend.Upload(imgIn, TensorShape.D2(nImg, dim));
        var modTxtGpu = backend.Upload(new float[2 * dim], TensorShape.D1(2 * dim));
        var modImgGpu = backend.Upload(new float[2 * dim], TensorShape.D1(2 * dim));
        var outTxtGpu = backend.Allocate(TensorShape.D2(nTxt, dim));
        var outImgGpu = backend.Allocate(TensorShape.D2(nImg, dim));

        try
        {
            backend.AdaLNModulateDual(
                outTxtGpu, inTxtGpu, modTxtGpu, 0, dim,
                outImgGpu, inImgGpu, modImgGpu, 0, dim,
                nTokens: nTxt + nImg, streamSplit: nTxt, dim, isRmsNorm, eps);

            var gpuTxtOut = new float[nTxt * dim];
            var gpuImgOut = new float[nImg * dim];
            backend.Download(outTxtGpu, gpuTxtOut);
            backend.Download(outImgGpu, gpuImgOut);

            double maxDiffTxt = 0, maxDiffImg = 0;
            for (int i = 0; i < cpuTxtOut.Length; i++) maxDiffTxt = Math.Max(maxDiffTxt, Math.Abs(cpuTxtOut[i] - gpuTxtOut[i]));
            for (int i = 0; i < cpuImgOut.Length; i++) maxDiffImg = Math.Max(maxDiffImg, Math.Abs(cpuImgOut[i] - gpuImgOut[i]));

            Console.WriteLine($"[Flux2AdaLNModulateDual:{label}] isRmsNorm={isRmsNorm} maxDiff(txt)={maxDiffTxt:E4} maxDiff(img)={maxDiffImg:E4}");
        }
        finally
        {
            backend.Free(inTxtGpu);
            backend.Free(inImgGpu);
            backend.Free(modTxtGpu);
            backend.Free(modImgGpu);
            backend.Free(outTxtGpu);
            backend.Free(outImgGpu);
        }
    }

    private static float[] MakeOrdinary(Random rng, int n, int dim, float scale)
    {
        var x = new float[n * dim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return x;
    }

    private static float[] MakeTinyVariance(Random rng, int n, int dim, float offset, float variance)
    {
        var x = new float[n * dim];
        for (int i = 0; i < x.Length; i++) x[i] = offset + (float)((rng.NextDouble() * 2 - 1) * variance);
        return x;
    }

    [Fact]
    public void LayerNormBranch_SmallScale_MatchesCpu()
    {
        var rng = new Random(11);
        const int nTxt = 4, nImg = 8, dim = 128;
        var txt = MakeOrdinary(rng, nTxt, dim, 1.0f);
        var img = MakeOrdinary(rng, nImg, dim, 1.0f);
        RunCase("LayerNorm_small", isRmsNorm: false, nTxt, nImg, dim, txt, img);
    }

    [Fact]
    public void RmsNormBranch_SmallScale_MatchesCpu()
    {
        var rng = new Random(12);
        const int nTxt = 4, nImg = 8, dim = 128;
        var txt = MakeOrdinary(rng, nTxt, dim, 1.0f);
        var img = MakeOrdinary(rng, nImg, dim, 1.0f);
        RunCase("RmsNorm_small", isRmsNorm: true, nTxt, nImg, dim, txt, img);
    }

    /// <summary>The decisive regression check for the fix: huge magnitude, tiny relative variance --
    /// the exact shape that broke the old one-pass `E[x^2]-mean^2` formula via cancellation.</summary>
    [Fact]
    public void LayerNormBranch_HugeMagnitudeTinyVariance_MatchesCpuClosely()
    {
        var rng = new Random(13);
        const int nTxt = 4, nImg = 8, dim = 3072;
        var txt = MakeTinyVariance(rng, nTxt, dim, 3.0e7f, 1000.0f);
        var img = MakeTinyVariance(rng, nImg, dim, 3.0e7f, 1000.0f);
        RunCase("LayerNorm_hugeTinyVariance", isRmsNorm: false, nTxt, nImg, dim, txt, img);
    }
}
