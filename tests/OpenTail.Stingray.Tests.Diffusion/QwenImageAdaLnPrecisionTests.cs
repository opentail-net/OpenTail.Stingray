using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Follow-up to a second-opinion review of docs/094 Phase 2's Qwen Image GPU bug (2026-09-20):
/// isolates ONE specific candidate the existing bisection never tested directly -- whether the CPU
/// (<see cref="DiffusionOps.LayerNormNoAffine"/>, SIMD-vectorized <c>TensorPrimitives.Sum</c>/
/// <c>SumOfSquares</c> reduction) and GPU (<see cref="Shaders.AdaLNModulate"/>'s <c>isRmsNorm:
/// false</c> branch, a plain sequential per-thread accumulation loop) affine-free-LayerNorm
/// implementations agree at Qwen Image's REAL scale: dim=3072 (not the 128/32 dims
/// <c>Flux2AdaLNModulateLayerNormGpuTests</c> already covers) and, critically, at the HUGE
/// activation magnitudes the block-by-block bisection actually measured (30 million growing past 1
/// billion by block 59) -- not the small [-1,1]-range inputs every existing AdaLN parity test uses.
///
/// The mechanism under test: LayerNorm subtracts the row mean from every element. If a row's values
/// are huge in absolute magnitude but have comparatively SMALL variance (plausible for a
/// heavily-grown residual stream, where each new block's residual contribution is much smaller than
/// the accumulated total), then `x - mean` is a subtraction of two nearly-equal huge floats --
/// catastrophic cancellation. Float32's absolute precision at magnitude 1e8 is roughly 8, at 1e9
/// roughly 64 -- if two backends compute `mean` via different summation orders (CPU's SIMD
/// tree-reduction vs GPU's sequential scalar loop) and round differently by even one ULP, the
/// RESULTING cancellation error could be large relative to the row's actual (small) variance, even
/// though the two `mean` values themselves agree almost exactly. This is a concrete, checkable
/// mechanism connecting the bisection's "huge activation magnitude" observation with the "CPU/GPU
/// disagree" observation, which the original investigation flagged as two separate facts without
/// testing whether they're causally linked.
/// </summary>
public sealed class QwenImageAdaLnPrecisionTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    /// <summary>Double-precision ground truth for one row of affine-free LayerNorm -- the
    /// arbiter for which of CPU (float32 SIMD reduction) or GPU (float32 sequential reduction) is
    /// actually closer to correct when they disagree, rather than just establishing THAT they
    /// disagree.</summary>
    private static void LayerNormNoAffineRowDouble(ReadOnlySpan<float> row, Span<float> outRow, double eps)
    {
        double sum = 0;
        for (int i = 0; i < row.Length; i++) sum += row[i];
        double mean = sum / row.Length;
        double sumSq = 0;
        for (int i = 0; i < row.Length; i++)
        {
            double d = row[i] - mean;
            sumSq += d * d;
        }
        double invStd = 1.0 / Math.Sqrt(sumSq / row.Length + eps);
        for (int i = 0; i < row.Length; i++)
            outRow[i] = (float)((row[i] - mean) * invStd);
    }

    private static void RunCase(string label, int nTokens, int dim, float[] input, float eps = 1e-6f)
    {
        using var backend = TryCreateVulkan();
        if (backend is null)
        {
            Console.WriteLine($"[QwenImageAdaLnPrecision:{label}] No Vulkan backend, skipping.");
            return;
        }

        var shift = new float[dim]; // zero shift/scale isolates the pure normalize step
        var scale = new float[dim];

        var cpuOut = new float[nTokens * dim];
        DiffusionOps.AdaLNModulate(cpuOut, input, shift, scale, nTokens, dim, isRmsNorm: false, eps: eps);

        var refOut = new float[nTokens * dim];
        for (int t = 0; t < nTokens; t++)
            LayerNormNoAffineRowDouble(input.AsSpan(t * dim, dim), refOut.AsSpan(t * dim, dim), eps);

        var inGpu = backend.Upload(input, TensorShape.D2(nTokens, dim));
        var shiftGpu = backend.Upload(shift, TensorShape.D1(dim));
        var scaleGpu = backend.Upload(scale, TensorShape.D1(dim));
        var outGpu = backend.Allocate(TensorShape.D2(nTokens, dim));

        try
        {
            backend.AdaLNModulate(outGpu, inGpu, shiftGpu, scaleGpu, nTokens, dim, isRmsNorm: false, eps: eps);

            var gpuOut = new float[nTokens * dim];
            backend.Download(outGpu, gpuOut);

            double maxDiff = 0, sumAbsCpu = 0;
            double dot = 0, normA = 0, normB = 0;
            double maxDiffCpuRef = 0, maxDiffGpuRef = 0;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                double diff = Math.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > maxDiff) maxDiff = diff;
                sumAbsCpu += Math.Abs(cpuOut[i]);
                dot += (double)cpuOut[i] * gpuOut[i];
                normA += (double)cpuOut[i] * cpuOut[i];
                normB += (double)gpuOut[i] * gpuOut[i];
                maxDiffCpuRef = Math.Max(maxDiffCpuRef, Math.Abs(cpuOut[i] - refOut[i]));
                maxDiffGpuRef = Math.Max(maxDiffGpuRef, Math.Abs(gpuOut[i] - refOut[i]));
            }
            double cosSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-30);
            double meanAbsCpu = sumAbsCpu / cpuOut.Length;

            Console.WriteLine($"[QwenImageAdaLnPrecision:{label}] nTokens={nTokens} dim={dim} maxDiff(cpu,gpu)={maxDiff:E4} meanAbs(cpuOut)={meanAbsCpu:E4} cosine={cosSim:F6} maxDiff(cpu,doubleRef)={maxDiffCpuRef:E4} maxDiff(gpu,doubleRef)={maxDiffGpuRef:E4}");
        }
        finally
        {
            backend.Free(inGpu);
            backend.Free(shiftGpu);
            backend.Free(scaleGpu);
            backend.Free(outGpu);
        }
    }

    [Fact]
    public void AtQwenImageRealDim_SmallScaleInput_CpuGpuAgree()
    {
        const int nTokens = 16, dim = 3072;
        var rng = new Random(1);
        var input = new float[nTokens * dim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        RunCase("dim3072_small[-1,1]", nTokens, dim, input);
    }

    /// <summary>Reproduces the bisection's own observed magnitude range (tens of millions) with
    /// ORDINARY (non-degenerate) per-row variance -- i.e. huge values that vary a lot relative to
    /// their own magnitude, not the pathological near-constant-row case tested below.</summary>
    [Fact]
    public void AtQwenImageRealDim_HugeMagnitudeOrdinaryVariance_CpuGpuAgree()
    {
        const int nTokens = 16, dim = 3072;
        var rng = new Random(2);
        var input = new float[nTokens * dim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)((rng.NextDouble() * 2 - 1) * 3.0e7);
        RunCase("dim3072_huge_ordinaryVariance", nTokens, dim, input);
    }

    /// <summary>The decisive case: huge absolute magnitude (~3e7, matching the bisection's block-0
    /// reading) but SMALL variance relative to that magnitude (values cluster tightly around a huge
    /// common offset) -- the exact shape that produces catastrophic cancellation in `x - mean` if
    /// the two backends' summation order disagrees even slightly. If CPU and GPU still agree here,
    /// the reduction-order hypothesis is refuted at this scale; if they diverge sharply here but not
    /// in the "ordinary variance" case above, that is decisive evidence for the cancellation
    /// mechanism.</summary>
    [Fact]
    public void AtQwenImageRealDim_HugeMagnitudeTinyVariance_CpuGpuComparison()
    {
        const int nTokens = 16, dim = 3072;
        var rng = new Random(3);
        var input = new float[nTokens * dim];
        const float bigOffset = 3.0e7f;
        for (int i = 0; i < input.Length; i++)
        {
            // Small variance (+-1000, i.e. ~0.003% of the offset) riding on top of a huge common
            // per-row offset -- matches "residual stream dominated by an old huge accumulated value
            // plus a comparatively tiny new contribution", the plausible shape of a heavily-grown
            // pre-norm transformer residual after many blocks.
            input[i] = bigOffset + (float)((rng.NextDouble() * 2 - 1) * 1000.0);
        }
        RunCase("dim3072_huge_tinyVariance", nTokens, dim, input);
    }

    /// <summary>Same tiny-variance-on-huge-offset shape as above but at the block-59-scale magnitude
    /// (~1e9) the second bisection run actually measured, to check whether the effect (if any)
    /// worsens as the residual stream's real, observed magnitude grows across the 60-layer
    /// trajectory.</summary>
    [Fact]
    public void AtQwenImageRealDim_ExtremeMagnitudeTinyVariance_CpuGpuComparison()
    {
        const int nTokens = 16, dim = 3072;
        var rng = new Random(4);
        var input = new float[nTokens * dim];
        const float bigOffset = 1.0e9f;
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = bigOffset + (float)((rng.NextDouble() * 2 - 1) * 30000.0);
        }
        RunCase("dim3072_extreme_tinyVariance", nTokens, dim, input);
    }
}
