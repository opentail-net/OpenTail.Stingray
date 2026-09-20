using OpenTail.Stingray.Diffusion.SD3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Follow-up to the Q4_K int8-activation-quantization fix (docs/094 Phase 1, 2026-09-20): fixing
/// that closed block 0's divergence (maxDiff 10.2 -> 0.04), but the coarse bisection re-run showed
/// block 1's QKV stage (attn.qkv, real Q5_K weight -- which never uses the int8 activation path at
/// all, confirmed by reading `DotQ5K`'s source) newly diverges by maxDiff~1.0-1.24, previously
/// masked by block 0's larger error. This isolates `QuantizedWeightCache.Linear`'s real Q5_K
/// `attn.qkv.weight` tensor against a wide-dynamic-range synthetic input (mimicking a real
/// post-LayerNorm, affine-modulated activation's magnitude, which the earlier Q5_K one-hot-column
/// dequant check could not exercise -- same structural blind spot as the original Q4_K check) to
/// determine whether Q5_K has its own real, separate large-magnitude precision bug.
/// </summary>
public sealed class Sd3Q5KLargeMagnitudeParityTests
{
    private readonly ITestOutputHelper _output;

    public Sd3Q5KLargeMagnitudeParityTests(ITestOutputHelper output)
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
    public void RealQ5K_QkvWeight_WideDynamicRangeInput_ReadF32MatchesFusedKernel()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        if (!File.Exists(ditPath))
        {
            _output.WriteLine("[Sd3Q5KLargeMag] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(ditPath);
        using var cache = new QuantizedWeightCache(weights, "");

        const string weightName = "joint_blocks.1.x_block.attn.qkv.weight";
        const int inDim = 1536;
        const int outDim = 4608;

        Assert.True(weights.Contains(weightName), $"Expected tensor '{weightName}'");

        var dequantFull = weights.ReadF32(weightName);
        Assert.Equal((long)outDim * inDim, dequantFull.Length);

        // Wide-dynamic-range synthetic input, mimicking a real post-LayerNorm/affine-modulated
        // activation's magnitude (the block-1 bisection's worst attn-output element reached
        // cpu=34.7) -- NOT the earlier narrow [-1,1] uniform range that the original Q5_K one-hot
        // check used, which structurally cannot expose a full-dot-product precision issue.
        var rng = new Random(99);
        var x = new float[inDim];
        for (int i = 0; i < inDim; i++) x[i] = (float)(rng.NextDouble() * 60.0 - 30.0);

        var cpuOut = new float[outDim];
        cache.Linear(weightName, x, ReadOnlySpan<float>.Empty, cpuOut, n: 1, inDim, outDim, allowQ8: false);

        var refOut = new float[outDim];
        for (int row = 0; row < outDim; row++)
        {
            double acc = 0;
            int rowOff = row * inDim;
            for (int k = 0; k < inDim; k++) acc += (double)x[k] * dequantFull[rowOff + k];
            refOut[row] = (float)acc;
        }

        double dot = 0, na = 0, nb = 0;
        float maxDiff = 0;
        int worstIdx = 0;
        for (int i = 0; i < outDim; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - refOut[i]);
            if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
            dot += cpuOut[i] * refOut[i];
            na += cpuOut[i] * cpuOut[i];
            nb += refOut[i] * refOut[i];
        }
        double cosSim = dot / (Math.Sqrt(na) * Math.Sqrt(nb));

        string msg = $"[Sd3Q5KLargeMag] REAL Q5_K qkv weight, wide-range input: cosine={cosSim:F6} maxDiff={maxDiff:E6} worst: fusedKernel={cpuOut[worstIdx]:F6} naiveRef(double)={refOut[worstIdx]:F6} (n={worstIdx})";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.True(maxDiff < 1e-2f, $"Real Q5_K attn.qkv.weight diverges between QuantizedWeightCache.Linear's fused kernel (DotQ5K) and a naive double-precision reference dot product by {maxDiff:E6} on a wide-dynamic-range input -- a real Q5_K precision bug, separate from the already-fixed Q4_K int8-activation-quantization bug.");
    }
}
