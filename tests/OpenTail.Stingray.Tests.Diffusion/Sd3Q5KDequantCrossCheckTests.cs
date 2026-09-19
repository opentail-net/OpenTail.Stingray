using OpenTail.Stingray.Diffusion.SD3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic for docs/094 Phase 1's SD3.5 GPU-bug bisection (maxDiff 0.118 vs CPU, cause not yet
/// found). Same cross-decoder-parity technique that ruled out this exact bug class for Qwen Image
/// (<see cref="QwenImageQ3KDequantCrossCheckTests"/>): `stingray list-tensors` shows
/// `joint_blocks.0.x_block.attn.qkv.weight` is **Q5_K** (not Q4_K like most of this checkpoint's
/// other block weights) -- a quant format never before cross-checked in this codebase for
/// ReadF32-vs-fused-kernel agreement. `MMDiTGpuWeights` reads this exact tensor via `GetWeight`
/// (-> `CachedWeightReader.Get` -> `IWeightLoader.ReadF32` -> `Dequantize.DequantQ5_K`) for GPU
/// upload, while the CPU path's real matmul goes through `QuantizedWeightCache.Linear`'s separate
/// fused Q5_K SIMD kernel. If these two disagree, MMDiT's GPU port would silently use wrong QKV
/// weights in every joint-attention block while every existing CPU test (which never exercises
/// `DequantQ5_K` for this large matrix, only the fused kernel) stays green.
/// </summary>
public sealed class Sd3Q5KDequantCrossCheckTests
{
    private readonly ITestOutputHelper _output;

    public Sd3Q5KDequantCrossCheckTests(ITestOutputHelper output)
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
    public void ReadF32DequantMatchesFusedKernelDequant_ForRealQ5KTensor()
    {
        string ditPath = FindModelPath(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        if (!File.Exists(ditPath))
        {
            _output.WriteLine("[Sd3Q5KDequantCrossCheckTests] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(ditPath);
        using var cache = new QuantizedWeightCache(weights, "");

        // joint_blocks.0.x_block.attn.qkv.weight: [inDim=1536, outDim=4608] per `list-tensors`,
        // real Q5_K per the real checkpoint inventory (not Q4_K like most other block weights).
        const string tensorName = "joint_blocks.0.x_block.attn.qkv.weight";
        Assert.True(weights.Contains(tensorName), $"Expected tensor '{tensorName}' in checkpoint");

        var dequantFull = weights.ReadF32(tensorName); // via Dequantize.DequantQ5_K

        const int inDim = 1536;
        const int outDim = 4608;

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
        string msg = $"[Sd3Q5KCrossCheck] Sampled {sampledCols} columns x {outDim} rows. MaxDiff(ReadF32 vs fused kernel) = {maxDiff:E6}, mean|fused| = {meanAbsFused:E6}";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.True(maxDiff < meanAbsFused * 0.01 + 1e-4,
            $"ReadF32's Q5_K dequant disagrees with the fused-kernel Q5_K decode by {maxDiff:E6} (mean weight magnitude {meanAbsFused:E6}) -- likely a real Q5_K dequant bug in one of the two independent decoders.");
    }
}
