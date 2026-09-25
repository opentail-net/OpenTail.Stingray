
namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared per-timestep dense math kernels used by every Transformer/Conformer-family pipeline
/// in this codebase (Chatterbox's S3Gen Conformer encoder, Parakeet's FastConformer encoder,
/// and any future rel-pos-attention port): SIMD Linear/LinearNoBias, LayerNorm, softmax, and
/// SiLU/Swish activation. Extracted after Parakeet's `ParakeetConformerEncoder.cs` and
/// Chatterbox's `ChatterboxFlowEncoder.cs` were found to each hand-roll near-identical private
/// copies of these five functions -- keep this the single source of truth going forward instead
/// of a third copy appearing in the next Conformer-style port.
/// </summary>
public static class DenseKernels
{
    /// <summary>y = W@x + b (bias optional). weight is row-major [outDim, inDim] (torch nn.Linear convention).</summary>
    public static unsafe float[] Linear(float[] input, float[] weight, float[]? bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, yp = output)
        {
            SimdKernels.MatVecF32(yp, wp, xp, outDim, inDim);
        }
        if (bias != null)
            TensorPrimitives.Add((ReadOnlySpan<float>)output, bias, output);
        return output;
    }

    /// <summary>y = W@x, no bias.</summary>
    public static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, yp = output)
        {
            SimdKernels.MatVecF32(yp, wp, xp, outDim, inDim);
        }
        return output;
    }

    /// <summary>
    /// Batched Y[m, outDim] = X[m, inDim] @ W^T (no bias), weight row-major [outDim, inDim]. Use this
    /// instead of calling <see cref="LinearNoBias(float[], float[], int, int)"/> once per token:
    /// a per-token mat-vec re-streams the whole weight for every row, while this packs W once
    /// (cached per weight array, 64-byte-aligned native memory freed with the array) and runs the
    /// blocked <see cref="PackedSgemmF32"/> GEMM. Results match the mat-vec to FP32 rounding.
    /// </summary>
    public static unsafe void LinearBatchedNoBias(ReadOnlySpan<float> input, float[] weight, Span<float> output, int m, int inDim, int outDim)
    {
        if (m <= 0) return;
        if (input.Length < m * inDim || output.Length < m * outDim)
            throw new ArgumentException("LinearBatchedNoBias: input/output span too small.");
        fixed (float* xp = input, yp = output)
        {
            if (!PackedSgemmF32.IsSupported || m < 2)
            {
                fixed (float* wp = weight)
                    for (int r = 0; r < m; r++)
                        SimdKernels.MatVecF32(yp + (long)r * outDim, wp, xp + (long)r * inDim, outDim, inDim);
                return;
            }
        }
        int rows = outDim, cols = inDim;
        var packed = s_packedWeights.GetValue(weight, w => new PackedLinearF32(w, null, rows, cols));
        if (packed.OutDim != outDim || packed.InDim != inDim)
            throw new ArgumentException($"LinearBatchedNoBias: weight packed as [{packed.OutDim}x{packed.InDim}], called as [{outDim}x{inDim}].");
        packed.Forward(input, output, m);
    }

    /// <summary>Row-array convenience over <see cref="LinearBatchedNoBias"/>: y[r] = W@x[r] + b for all
    /// rows in one batched GEMM (weight row-major [outDim, inDim]), for callers that hold per-frame
    /// row arrays and previously ran one <see cref="Linear"/> per row.</summary>
    public static float[][] LinearBatchedRows(float[][] rows, float[] weight, float[]? bias, int inDim, int outDim)
    {
        int m = rows.Length;
        var x = new float[m * inDim];
        for (int r = 0; r < m; r++) rows[r].AsSpan(0, inDim).CopyTo(x.AsSpan(r * inDim, inDim));
        var y = new float[m * outDim];
        LinearBatchedNoBias(x, weight, y, m, inDim, outDim);
        var output = new float[m][];
        Parallel.For(0, m, r =>
        {
            var row = y.AsSpan(r * outDim, outDim).ToArray();
            if (bias != null) TensorPrimitives.Add((ReadOnlySpan<float>)row, bias, row);
            output[r] = row;
        });
        return output;
    }

    // Packed panel copy of each weight array (shape fixed by its first use), freed with the array.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<float[], PackedLinearF32> s_packedWeights = new();

    /// <summary>y = W@x, no bias (span input).</summary>
    public static unsafe float[] LinearNoBias(ReadOnlySpan<float> input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, yp = output)
        {
            SimdKernels.MatVecF32(yp, wp, xp, outDim, inDim);
        }
        return output;
    }

    public static unsafe float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[x.Length];
        fixed (float* op = output, xp = x, wp = weight, bp = bias)
        {
            SimdKernels.LayerNorm(op, xp, wp, bp, x.Length, eps);
        }
        return output;
    }

    /// <summary>In-place SiLU/Swish: x *= sigmoid(x).</summary>
    public static void SiluInPlace(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-v));
        }
    }

    /// <summary>In-place softmax on Span, float-accumulated. Explicit max-subtraction, not
    /// <c>TensorPrimitives.SoftMax(x, x)</c> -- confirmed root cause (2026-09-05) of Chatterbox's
    /// S3Gen CFM decoder producing NaN mid-stack (reproducible, always in the real/non-zero
    /// conditioning branch, always around mid-stage 7-9's transformer blocks): bisected by
    /// swapping this exact call back to the explicit stable formula with everything else held
    /// fixed, which eliminated the NaN. Do not swap back to <c>TensorPrimitives.SoftMax</c> here
    /// without re-verifying against a real Chatterbox generation, not just a golden/unit test.</summary>
    public static void SoftmaxInPlace(Span<float> scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    /// <summary>In-place softmax, float-accumulated (no double promotion -- matches the numerical path every other softmax in this codebase uses).</summary>
    public static void SoftmaxInPlace(float[] scores) => SoftmaxInPlace(scores.AsSpan());
}
