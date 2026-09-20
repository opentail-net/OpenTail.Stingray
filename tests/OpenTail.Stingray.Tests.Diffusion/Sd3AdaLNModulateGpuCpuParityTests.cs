using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Direct GPU-vs-CPU unit test for the <c>AdaLNModulate</c> primitive itself (docs/094 Phase 1,
/// 2026-09-20 bisection follow-up): the block-level bisection found SD3.5's CPU/GPU forward-pass
/// divergence is introduced entirely within block 0, with the FIRST divergent stage being the
/// modulation output (`ws.ImgMod`/`ws.NormedImg`). Five candidates were already ruled out
/// (FP16 attention, FP16 weights, dual-attention, Q5_K/Q4_K dequant) -- none of which touch the
/// `AdaLNModulate` shader's own LayerNorm math. This test isolates JUST that primitive: feeds the
/// GPU shader and the CPU reference formula the exact SAME hand-constructed x/shift/scale inputs
/// (no Sgemm, no weight upload, no real checkpoint involved at all), so any mismatch can only come
/// from the LayerNorm/affine math itself, not anything upstream.
/// </summary>
public sealed class Sd3AdaLNModulateGpuCpuParityTests
{
    private readonly ITestOutputHelper _output;

    public Sd3AdaLNModulateGpuCpuParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>Exact copy of `MMDiTModel.ModulateNorm`'s CPU formula (that method is private),
    /// used here as the independent reference to compare the GPU shader against.</summary>
    private static void CpuReference(ReadOnlySpan<float> tokens, Span<float> normed, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale, int numTokens, int dim, float eps)
    {
        for (int i = 0; i < numTokens; i++)
        {
            var inRow = tokens.Slice(i * dim, dim);
            var outRow = normed.Slice(i * dim, dim);

            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += inRow[d];
            mean /= dim;

            float sumSq = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = inRow[d] - mean;
                sumSq += diff * diff;
            }
            float invStd = 1f / MathF.Sqrt(sumSq / dim + eps);

            for (int d = 0; d < dim; d++)
            {
                float n = (inRow[d] - mean) * invStd;
                outRow[d] = n * (1f + scale[d]) + shift[d];
            }
        }
    }

    [Fact]
    public void AdaLNModulate_GpuMatchesCpu_DirectInputs()
    {
        using var backend = new VulkanBackend();

        const int numTokens = 333; // same total token count as SD3.5-medium's real 256x256/77-text case
        const int dim = 1536;      // SD3.5-medium's real HiddenSize
        const float eps = 1e-6f;

        var rng = new Random(42);
        var x = new float[numTokens * dim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 20.0 - 10.0); // wide range, matching real activation scale seen in the bisection (values up to ~30)

        // mod buffer: shift at [0,dim), scale at [dim,2*dim) -- same layout AdaLNModulate's
        // (shiftOffset=0, scaleOffset=dim) overload uses in MMDiTModel.ForwardGpu.
        var mod = new float[2 * dim];
        for (int i = 0; i < dim; i++)
        {
            mod[i] = (float)(rng.NextDouble() * 2.0 - 1.0);        // shift
            mod[dim + i] = (float)(rng.NextDouble() * 2.0 - 1.0);  // scale
        }

        var cpuOut = new float[numTokens * dim];
        CpuReference(x, cpuOut, mod.AsSpan(0, dim), mod.AsSpan(dim, dim), numTokens, dim, eps);

        var xGpu = backend.Upload(x, TensorShape.D2(numTokens, dim), exact: true);
        var modGpu = backend.Upload(mod, TensorShape.D1(2 * dim), exact: true);
        var outGpu = backend.Allocate(TensorShape.D2(numTokens, dim));

        backend.AdaLNModulate(outGpu, xGpu, modGpu, numTokens, dim, shiftOffset: 0, scaleOffset: dim, isRmsNorm: false, eps: eps);

        var gpuOut = new float[numTokens * dim];
        backend.Download(outGpu, gpuOut);

        backend.Free(xGpu);
        backend.Free(modGpu);
        backend.Free(outGpu);

        double dot = 0, na = 0, nb = 0;
        float maxDiff = 0;
        int worstIdx = 0;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
            dot += cpuOut[i] * gpuOut[i];
            na += cpuOut[i] * cpuOut[i];
            nb += gpuOut[i] * gpuOut[i];
        }
        double cosSim = dot / (Math.Sqrt(na) * Math.Sqrt(nb));

        string msg = $"[Sd3AdaLNParity] cosine={cosSim:F6} maxDiff={maxDiff:E6} worst: cpu={cpuOut[worstIdx]:F6} gpu={gpuOut[worstIdx]:F6} (token={worstIdx / dim}, dim={worstIdx % dim})";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.True(maxDiff < 1e-3f, $"AdaLNModulate GPU shader diverges from the CPU reference formula by {maxDiff:E6} on IDENTICAL hand-fed inputs (no Sgemm/weight-upload involved) -- a real bug in the shader's own LayerNorm/affine math, not precision or dequant.");
    }
}
