using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic for docs/094 Phase 7's Z-Image GPU-bug bisection (divergence starts at block 0 of
/// 30, real weights, real scale). `ZImageGpuWeights` reads weights via `ZImageDiT.W` ->
/// `IWeightLoader.ReadF32` (`Dequantize.ToFloat32`'s generic per-format decoder), while the CPU
/// path's real matmuls go through `MatQ` -> `_st.TryGetRaw` + `SimdKernels.MatMulBatched`'s own
/// fused quantized kernels -- two independently-implemented decoders for the SAME real Q4_0
/// checkpoint, the exact bug class already tested (and ruled out) for Qwen Image's Q4_K/BF16 and
/// SD3.5's Q5_K tensors this session, but never checked for Q4_0 specifically.
/// </summary>
public sealed class ZImageQ4_0DequantCrossCheckTests
{
    private readonly ITestOutputHelper _output;

    public ZImageQ4_0DequantCrossCheckTests(ITestOutputHelper output)
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
    public void ReadF32DequantMatchesFusedKernelDequant_ForRealQ4_0Tensor()
    {
        string? modelPath = FindModelPath("z_image_turbo-Q4_0.gguf");
        if (modelPath is null)
        {
            _output.WriteLine("[ZImageQ4_0DequantCrossCheckTests] Checkpoint missing, skipping."); Console.WriteLine("[ZImageQ4_0DequantCrossCheckTests] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(modelPath);
        using var cache = new QuantizedWeightCache(weights);

        // layers.0.attention.qkv.weight: real Q4_0 tensor, [inDim=3840, outDim=11520] per
        // `list-tensors`.
        const string tensorName = "layers.0.attention.qkv.weight";
        Assert.True(weights.Contains(tensorName), $"Expected tensor '{tensorName}' in checkpoint");

        var dequantFull = weights.ReadF32(tensorName); // via Dequantize.ToFloat32 (Q4_0 case)

        const int inDim = 3840;
        const int outDim = 11520;

        double maxDiff = 0;
        double sumAbsFused = 0;
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
        }

        double meanAbsFused = sumAbsFused / (8 * outDim);
        string msg = $"[ZImageQ4_0CrossCheck] MaxDiff(ReadF32 vs fused kernel) = {maxDiff:E6}, mean|fused| = {meanAbsFused:E6}";
        _output.WriteLine(msg); Console.WriteLine(msg);

        Assert.True(maxDiff < meanAbsFused * 0.01 + 1e-4,
            $"ReadF32's Q4_0 dequant disagrees with the fused-kernel Q4_0 decode by {maxDiff:E6} (mean weight magnitude {meanAbsFused:E6}) -- likely a real Q4_0 dequant bug in one of the two independent decoders.");
    }
}
