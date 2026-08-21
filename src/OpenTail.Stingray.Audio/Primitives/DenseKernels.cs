using System;
using OpenTail.Stingray.Cpu;

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
            for (int o = 0; o < outDim; o++) output[o] += bias[o];
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

    public static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f)
    {
        int n = x.Length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
        var /= n;
        float invStd = (float)(1.0 / Math.Sqrt(var + eps));

        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (float)((x[i] - mean) * invStd) * weight[i] + bias[i];
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

    /// <summary>In-place softmax, float-accumulated (no double promotion -- matches the numerical path every other softmax in this codebase uses).</summary>
    public static void SoftmaxInPlace(float[] scores)
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
}
