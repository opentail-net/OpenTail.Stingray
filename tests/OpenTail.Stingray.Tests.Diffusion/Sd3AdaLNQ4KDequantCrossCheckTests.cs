using OpenTail.Stingray.Diffusion.SD3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic for docs/094 Phase 1's SD3.5 GPU-bug bisection. The 2026-09-20 block-level bisection
/// (<see cref="Sd3BaselineTests.BisectGpuCpuDivergence_Sd35_WithinBlock0"/>) found CPU/GPU state
/// already matches almost exactly (maxDiff 0.000351) through x_embedder/pos_embed/context_embedder/
/// t_embedder/y_embedder/final_layer -- ALL of which are real Float16 tensors, not quantized -- but
/// diverges immediately (maxDiff ~1-2) at block 0's very first step, the `adaLN_modulation.1`
/// linear. `list-tensors` shows that tensor is real **Q4_K** (`joint_blocks.0.x_block.
/// adaLN_modulation.1.weight`, [1536, 13824]) -- the FIRST quantized weight touched anywhere in
/// the forward pass. Same cross-decoder-parity technique already used to rule out this exact bug
/// class for `attn.qkv.weight`'s Q5_K (<see cref="Sd3Q5KDequantCrossCheckTests"/>, ruled out) and
/// for Qwen Image's Q3_K -- but that Q5_K check covered a DIFFERENT tensor/quant-type combination
/// than this one, so it does not by itself clear Q4_K/adaLN_modulation specifically.
/// </summary>
public sealed class Sd3AdaLNQ4KDequantCrossCheckTests
{
    private readonly ITestOutputHelper _output;

    public Sd3AdaLNQ4KDequantCrossCheckTests(ITestOutputHelper output)
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
    public void ReadF32DequantMatchesFusedKernelDequant_ForRealQ4KAdaLNTensor()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        if (!File.Exists(ditPath))
        {
            _output.WriteLine("[Sd3AdaLNQ4KDequantCrossCheckTests] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(ditPath);
        using var cache = new QuantizedWeightCache(weights, "");

        // joint_blocks.0.x_block.adaLN_modulation.1.weight: [inDim=1536, outDim=13824] per
        // `list-tensors`, real Q4_K.
        const string tensorName = "joint_blocks.0.x_block.adaLN_modulation.1.weight";
        Assert.True(weights.Contains(tensorName), $"Expected tensor '{tensorName}' in checkpoint");

        var dequantFull = weights.ReadF32(tensorName); // via Dequantize.DequantQ4_K

        const int inDim = 1536;
        const int outDim = 13824;

        double maxDiff = 0;
        double sumAbsFused = 0;
        int sampledCols = 0;
        var rng = new Random(4242);
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
        string msg = $"[Sd3AdaLNQ4KCrossCheck] Sampled {sampledCols} columns x {outDim} rows. MaxDiff(ReadF32 vs fused kernel) = {maxDiff:E6}, mean|fused| = {meanAbsFused:E6}";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.True(maxDiff < meanAbsFused * 0.01 + 1e-4,
            $"ReadF32's Q4_K dequant disagrees with the fused-kernel Q4_K decode by {maxDiff:E6} (mean weight magnitude {meanAbsFused:E6}) -- likely a real Q4_K dequant bug in one of the two independent decoders, for THIS tensor shape/quant combination specifically.");
    }
}
