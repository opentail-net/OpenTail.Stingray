using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic for docs/094 Phase 2's Qwen-Image GPU-port investigation: the CPU path's real
/// matmuls go through <see cref="QuantizedWeightCache"/>'s fused Q3_K SIMD dot-product kernels
/// (<c>SimdKernels</c>'s <c>DotQ3K_Q8KS_*In</c> family), while <see cref="QwenImageGpuWeights"/>
/// (the new GPU port) reads the same tensors via the plain <see cref="IWeightLoader.ReadF32"/> path
/// (<c>Dequantize.DequantQ3K</c>) before uploading to VRAM -- two independently-implemented Q3_K
/// decoders. If they disagree, the GPU port would silently use wrong weights while every existing
/// CPU test (which never exercises <c>DequantQ3K</c> for large Q3_K matrices, only the fused
/// kernels) stays green -- exactly the "runs fine, wrong output" signature found in
/// <see cref="QwenImageGpuEndToEndSmokeTests"/>. This test computes the SAME single-row linear
/// projection two ways (once via <c>ReadF32</c>+naive dot product, once via
/// <c>QuantizedWeightCache.Linear</c>'s real fused path) and diffs them directly.
/// </summary>
public sealed class QwenImageQ3KDequantCrossCheckTests
{
    private readonly ITestOutputHelper _output;

    public QwenImageQ3KDequantCrossCheckTests(ITestOutputHelper output)
    {
        _output = output;
    }

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

    [Fact]
    public void ReadF32DequantMatchesFusedKernelDequant_ForOneRealQ3KTensor()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        if (modelPath is null)
        {
            _output.WriteLine("[QwenImageQ3KDequantCrossCheckTests] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(modelPath);
        using var cache = new QuantizedWeightCache(weights, "");

        // REAL CHECKPOINT INVENTORY CORRECTION (found via `stingray list-tensors`, 2026-09-19):
        // img_in.weight is actually BFloat16, NOT Q3_K -- only the per-block transformer_blocks.*
        // weights are Q4_K (the "Q3_K_S" filename names the overall llama.cpp quant PROFILE, not
        // every individual tensor's dtype; img_in/txt_in/norm_out/proj_out are BF16, per-block
        // attn/mlp/mod weights are Q4_K). Cross-check a REAL Q4_K tensor instead --
        // transformer_blocks.0.attn.to_q.weight [3072,3072] -- since that's what actually governs
        // all 60 transformer blocks' compute, the bulk of this model.
        const string tensorName = "transformer_blocks.0.attn.to_q.weight";
        Assert.True(weights.Contains(tensorName), $"Expected tensor '{tensorName}' in checkpoint");

        var dequantFull = weights.ReadF32(tensorName); // [3072*3072] via Dequantize's Q4_K path

        // Run the SAME weight through the fused-kernel path via a real Linear() call with a
        // one-hot input vector: Linear(oneHot_k, W) = W[:, k] (the k-th COLUMN of the [out,in]
        // weight matrix), letting us extract individual dequantized-and-consumed values from the
        // fused kernel's own decode without needing a second full dequant buffer.
        const int inDim = QwenImageModel.HiddenDim;   // 3072
        const int outDim = QwenImageModel.HiddenDim;  // 3072

        double maxDiff = 0;
        double sumAbsFused = 0;
        int sampledCols = 0;
        var rng = new Random(1234);
        // Sample a handful of columns (not all 64) to keep this fast while still covering
        // multiple Q3_K blocks (each block covers 256 contiguous flat elements = row-major
        // [outDim, inDim], so different columns land in different blocks' scale groups).
        for (int trial = 0; trial < 8; trial++)
        {
            int k = rng.Next(inDim);
            var oneHot = new float[inDim];
            oneHot[k] = 1.0f;
            var fusedCol = new float[outDim];
            cache.Linear(tensorName, oneHot, ReadOnlySpan<float>.Empty, fusedCol, n: 1, inDim, outDim);

            for (int row = 0; row < outDim; row++)
            {
                float expected = dequantFull[row * inDim + k];
                float actual = fusedCol[row];
                double diff = Math.Abs(expected - actual);
                if (diff > maxDiff) maxDiff = diff;
                sumAbsFused += Math.Abs(actual);
            }
            sampledCols++;
        }

        double meanAbsFused = sumAbsFused / (sampledCols * outDim);
        string msg = $"[Q3KCrossCheck] Sampled {sampledCols} columns x {outDim} rows. MaxDiff(ReadF32 vs fused kernel) = {maxDiff:E6}, mean|fused| = {meanAbsFused:E6}";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        // A real, independent Q3_K decode disagreement would show up as a diff comparable to the
        // weight magnitude itself (not float rounding noise at ~1e-6/1e-7 relative to mean|fused|).
        Assert.True(maxDiff < meanAbsFused * 0.01 + 1e-4,
            $"ReadF32's Q3_K dequant disagrees with the fused-kernel Q3_K decode by {maxDiff:E6} (mean weight magnitude {meanAbsFused:E6}) -- likely a real Q3_K dequant bug in one of the two independent decoders, not float noise.");
    }
}
