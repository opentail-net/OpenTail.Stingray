
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
    public static void SiluInPlace(float[] x)
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
