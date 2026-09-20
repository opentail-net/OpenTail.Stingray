using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.SD3;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Direct GPU-vs-CPU unit test for `x_block.adaLN_modulation.1`'s Linear layer using the REAL
/// checkpoint weight and a REAL tVecSilu input (docs/094 Phase 1, 2026-09-20 bisection follow-up
/// #3). Two prior isolated tests this pass proved BOTH the `AdaLNModulate` shader AND the M=1/
/// K=1536/N=13824 Sgemm shape are individually exact (maxDiff ~1e-6) on SYNTHETIC random data --
/// yet the full pipeline's block-0 bisection shows the SAME Sgemm call, with the REAL checkpoint
/// weight and REAL tVec, diverges by maxDiff~1-2.4. This test closes that gap: same real weight
/// tensor, same real Linear call, but isolated from the rest of the 24-block pipeline, to find out
/// whether the real tensor's specific values (not exercised by random synthetic data or the 8
/// one-hot-column dequant samples) are the actual cause.
/// </summary>
public sealed class Sd3RealModulationLinearGpuCpuParityTests
{
    private readonly ITestOutputHelper _output;

    public Sd3RealModulationLinearGpuCpuParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray\models\_models", Path.GetFileName(relativePath)),
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    [Fact]
    public void RealAdaLNModulationLinear_GpuMatchesCpu()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        if (!File.Exists(ditPath))
        {
            _output.WriteLine("[Sd3RealModLinear] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(ditPath);
        // budgetBytes: 0 forces GetOrCreateRepackedQ4Kx8 to always return null, bypassing the
        // repacked Q4Kx8 SIMD fast path (SimdKernels.TryMatMulBatchedQ4Kx8) and falling back to
        // SimdKernels.MatMulBatched's raw-quantized decode instead -- lets this test isolate
        // whether the fast repacked path specifically is where the divergence originates.
        bool bypassRepack = Environment.GetEnvironmentVariable("STINGRAY_SD3_BYPASS_Q4KX8_REPACK") == "1";
        using var cache = new QuantizedWeightCache(weights, "", budgetBytes: bypassRepack ? 0 : null);

        const string weightName = "joint_blocks.0.x_block.adaLN_modulation.1.weight";
        const string biasName = "joint_blocks.0.x_block.adaLN_modulation.1.bias";
        const int inDim = 1536;
        const int outDim = 13824;

        Assert.True(weights.Contains(weightName), $"Expected tensor '{weightName}'");
        Assert.True(weights.Contains(biasName), $"Expected tensor '{biasName}'");

        var bias = weights.ReadF32(biasName);
        Assert.Equal(outDim, bias.Length);

        // REAL tVecSilu: the actual timestep+pooled-Y embedding this checkpoint's own MMDiTModel
        // computes for timestep=1000f (same value used by TestSd35GpuVsCpuParity/the block-0
        // bisection), SiLU'd exactly as ForwardGpu/Forward do before feeding it into this Linear.
        // A synthetic random x showed only maxDiff~4e-6 (correct); the REAL weight with a synthetic
        // x showed maxDiff~9.4e-3 (small, real, but nowhere near the ~1-2.4 seen in the real
        // pipeline) -- this checks whether it's specifically the REAL tVecSilu's actual magnitudes/
        // values (not exercised by random x) that trigger the full-size divergence.
        var mmditForTVec = new MMDiTModel(weights, prefix: "");
        var pooledY = new float[2048];
        var rngPooled = new Random(42);
        for (int i = 0; i < pooledY.Length; i++) pooledY[i] = (float)(rngPooled.NextDouble() * 0.1);
        var tVec = mmditForTVec.ComputeTimeAndPooledEmbedding(1000f, pooledY);
        var x = (float[])tVec.Clone();
        OpenTail.Stingray.Diffusion.DiffusionOps.SiluInPlace(x);
        mmditForTVec.Dispose();

        // CPU path: EXACT same call MMDiTModel.Lin() makes for this tensor when _backend is null.
        var cpuOut = new float[outDim];
        cache.Linear(weightName, x, bias, cpuOut, n: 1, inDim, outDim);

        // GPU path: EXACT same dequant (ReadF32) + upload + Sgemm + bias-add MMDiTGpuWeights/
        // ForwardGpu perform, but standalone (no MMDiTGpuWeights/workspace/24-block loop).
        var realWeightF32 = weights.ReadF32(weightName);
        Assert.Equal((long)outDim * inDim, realWeightF32.Length);

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(inDim), exact: true);
        var wGpu = backend.Upload(realWeightF32, TensorShape.D2(outDim, inDim), exact: true);
        var outGpu = backend.Allocate(TensorShape.D1(outDim));
        backend.Sgemm(outGpu, xGpu, wGpu, M: 1, K: inDim, N: outDim);
        var gpuOutNoBias = new float[outDim];
        backend.Download(outGpu, gpuOutNoBias);
        var gpuOut = new float[outDim];
        for (int i = 0; i < outDim; i++) gpuOut[i] = gpuOutNoBias[i] + bias[i];

        backend.Free(xGpu);
        backend.Free(wGpu);
        backend.Free(outGpu);

        double dot = 0, na = 0, nb = 0;
        float maxDiff = 0;
        int worstIdx = 0;
        for (int i = 0; i < outDim; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
            dot += cpuOut[i] * gpuOut[i];
            na += cpuOut[i] * cpuOut[i];
            nb += gpuOut[i] * gpuOut[i];
        }
        double cosSim = dot / (Math.Sqrt(na) * Math.Sqrt(nb));

        string msg = $"[Sd3RealModLinear] REAL weight, {outDim} outputs: cosine={cosSim:F6} maxDiff={maxDiff:E6} worst: cpu={cpuOut[worstIdx]:F6} gpu={gpuOut[worstIdx]:F6} gpuNoBias={gpuOutNoBias[worstIdx]:F6} bias={bias[worstIdx]:F6} (n={worstIdx})";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        // Cancellation-signature check on the worst row: if sum(|term|) >> |net sum|, this is
        // catastrophic cancellation (many large-magnitude terms nearly cancelling), which explains
        // why different summation ORDERS (naive vs SIMD-paired vs BLAS-blocked vs GPU-tiled) can
        // legitimately diverge by a large absolute amount even with zero "bug" in any one of them.
        double sumAbsTerms = 0, sumSignedTerms = 0;
        int rowOff = worstIdx * inDim;
        for (int k = 0; k < inDim; k++)
        {
            double term = (double)x[k] * realWeightF32[rowOff + k];
            sumAbsTerms += Math.Abs(term);
            sumSignedTerms += term;
        }
        string cancelMsg = $"[Sd3RealModLinear] worst row cancellation check: sum(|term|)={sumAbsTerms:F4} vs |netSum|={Math.Abs(sumSignedTerms):F4} (ratio={sumAbsTerms / Math.Max(Math.Abs(sumSignedTerms), 1e-9):F1}x) -- large ratio = severe cancellation";
        _output.WriteLine(cancelMsg);
        Console.WriteLine(cancelMsg);

        Assert.True(maxDiff < 1e-2f, $"REAL adaLN_modulation.1 Linear diverges between CPU fused-kernel and GPU Sgemm+bias by {maxDiff:E6} using the SAME real checkpoint weight bytes -- points to a real bug in either ReadF32's full-row Q4_K decode or QuantizedWeightCache.Linear's full-row decode (not caught by the 8-sample one-hot column check).");
    }
}
