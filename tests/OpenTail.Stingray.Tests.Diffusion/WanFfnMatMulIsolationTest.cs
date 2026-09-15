using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Isolation test (docs/081, 2026-09-14): SimdKernels.MatMulBatchedF32 at Wan's real FFN
/// dimensions (rows=8960 [ffn_dim], cols=1536 [dim], batchSize=64 [tokens]) compared against a
/// dead-simple triple-loop reference, after a cross-reference dump comparison against the real
/// C++ reference (stable-diffusion.cpp) found the C# port's block0 FFN up-projection output
/// diverges sharply (cosine=0.80, ~2x too-large norm) specifically at this scale -- while smaller
/// or square-shaped Linear calls elsewhere in the same model (self-attention q/k/v/o at 1536x1536,
/// patch_embedding at 64-in/1536-out) were independently verified byte-identical to the reference.
/// </summary>
public sealed class WanFfnMatMulIsolationTest
{
    [Fact]
    public unsafe void MatMulBatchedF32_MatchesNaiveReference_AtRealWanFfnDimensions()
    {
        const int rows = 8960;   // ffn_dim (outDim)
        const int cols = 1536;   // dim (inDim)
        const int batchSize = 64; // tokens

        var rng = new Random(123);
        var weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        var bias = new float[rows];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        var input = new float[batchSize * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var output = new float[batchSize * rows];
        fixed (float* pOut = output, pW = weights, pIn = input, pB = bias)
        {
            SimdKernels.MatMulBatchedF32(pOut, pW, pIn, batchSize, rows, cols, pB);
        }

        // Naive reference: output[p, r] = bias[r] + sum_c weights[r, c] * input[p, c]
        var reference = new float[batchSize * rows];
        for (int p = 0; p < batchSize; p++)
        {
            for (int r = 0; r < rows; r++)
            {
                double sum = bias[r];
                for (int c = 0; c < cols; c++)
                {
                    sum += (double)weights[r * cols + c] * input[p * cols + c];
                }
                reference[p * rows + r] = (float)sum;
            }
        }

        float maxAbsDiff = 0f;
        double sumSqDiff = 0, sumSqRef = 0;
        int worstIdx = -1;
        for (int i = 0; i < output.Length; i++)
        {
            float diff = MathF.Abs(output[i] - reference[i]);
            if (diff > maxAbsDiff) { maxAbsDiff = diff; worstIdx = i; }
            sumSqDiff += (double)diff * diff;
            sumSqRef += (double)reference[i] * reference[i];
        }
        float relErr = (float)Math.Sqrt(sumSqDiff / Math.Max(1e-12, sumSqRef));
        Console.WriteLine($"[WanFfnMatMulIsolation] maxAbsDiff={maxAbsDiff:F6} relErr={relErr:F6} worstIdx={worstIdx} " +
            $"kernel={output[Math.Max(0, worstIdx)]:F6} ref={reference[Math.Max(0, worstIdx)]:F6}");

        Assert.True(relErr < 0.01f, $"MatMulBatchedF32 diverges from naive reference at real Wan FFN dims: relErr={relErr}, maxAbsDiff={maxAbsDiff}");
    }
}
