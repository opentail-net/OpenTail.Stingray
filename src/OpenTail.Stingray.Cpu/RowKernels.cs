using System.Numerics.Tensors;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Row-wise LayerNorm / RMSNorm and an exact max-subtracted softmax over spans (TensorPrimitives), shared by the
/// HF-checkpoint model ports (T5, Chronos-2, Wav2Vec2, CLIP). Kept separate from <see cref="SimdKernels"/>'s pointer
/// kernels on purpose: <c>SimdKernels.SoftmaxInPlace</c> follows ggml's exp details for llama.cpp parity, and these
/// ports were each verified against their ONNX exports with exactly this arithmetic.
/// </summary>
public static class RowKernels
{
    /// <summary><c>y[r] = (x[r] - mean) / sqrt(var + eps) · w + b</c> for each of <paramref name="rows"/> rows of width <paramref name="d"/>.</summary>
    public static void LayerNormRows(float[] x, float[] y, int rows, int d, float[] w, float[] b, float eps)
    {
        Parallel.For(0, rows, r =>
        {
            var xr = x.AsSpan(r * d, d);
            var yr = y.AsSpan(r * d, d);
            float mean = TensorPrimitives.Sum(xr) / d;
            TensorPrimitives.Subtract(xr, mean, yr);
            float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(yr) / d + eps);
            TensorPrimitives.Multiply(yr, inv, yr);
            TensorPrimitives.Multiply(yr, w, yr);
            TensorPrimitives.Add(yr, b, yr);
        });
    }

    /// <summary>T5-style RMSNorm (no mean, no bias): <c>y[r] = x[r] / sqrt(mean(x²) + eps) · w</c>.</summary>
    public static void RmsNormRows(float[] x, float[] y, int rows, int d, float[] w, float eps)
    {
        Parallel.For(0, rows, r =>
        {
            var xr = x.AsSpan(r * d, d);
            var yr = y.AsSpan(r * d, d);
            float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(xr) / d + eps);
            TensorPrimitives.Multiply(xr, inv, yr);
            TensorPrimitives.Multiply(yr, w, yr);
        });
    }

    /// <summary>Softmax with the maximum subtracted first (safe for unscaled attention scores; -inf entries become 0).</summary>
    public static void SoftmaxInPlace(Span<float> s)
    {
        TensorPrimitives.Subtract(s, TensorPrimitives.Max(s), s);
        TensorPrimitives.Exp(s, s);
        TensorPrimitives.Divide(s, TensorPrimitives.Sum(s), s);
    }
}
