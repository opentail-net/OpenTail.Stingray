using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// An F32 linear layer (<c>y = x @ W^T + b</c>, torch <c>nn.Linear</c> layout: W row-major
/// [outDim, inDim]) whose weight is packed once into <see cref="PackedSgemmF32"/>'s panel layout, so
/// every call runs the blocked GEMM over all <c>m</c> rows at once. The packed copy replaces the
/// source array (only the native panels are kept). Without AVX2+FMA it keeps the plain weight and
/// falls back to one mat-vec per row.
/// </summary>
public sealed unsafe class PackedLinearF32 : IDisposable
{
    private float* _packed;
    private readonly float[]? _plain;

    public int OutDim { get; }
    public int InDim { get; }
    public float[]? Bias { get; }

    public PackedLinearF32(float[] weight, float[]? bias, int outDim, int inDim)
    {
        if (weight.Length != (long)outDim * inDim)
            throw new ArgumentException($"PackedLinearF32: weight has {weight.Length} floats, expected {outDim}x{inDim}.");
        if (bias is not null && bias.Length != outDim)
            throw new ArgumentException($"PackedLinearF32: bias has {bias.Length} floats, expected {outDim}.");
        (OutDim, InDim, Bias) = (outDim, inDim, bias);
        if (PackedSgemmF32.IsSupported)
            fixed (float* wp = weight) _packed = PackedSgemmF32.PackWeights(wp, outDim, inDim);
        else
            _plain = weight;
    }

    /// <summary>output[m, OutDim] = input[m, InDim] @ W^T (+ bias).</summary>
    public void Forward(ReadOnlySpan<float> input, Span<float> output, int m)
    {
        if (m <= 0) return;
        if (input.Length < (long)m * InDim || output.Length < (long)m * OutDim)
            throw new ArgumentException("PackedLinearF32.Forward: input/output span too small.");
        ObjectDisposedException.ThrowIf(_packed == null && _plain is null, this);
        fixed (float* xp = input, yp = output, bp = Bias)
        {
            if (_packed != null)
            {
                PackedSgemmF32.Gemm(yp, xp, _packed, bp, m, OutDim, InDim);
                return;
            }
            fixed (float* wp = _plain)
                for (int r = 0; r < m; r++)
                    SimdKernels.MatVecF32(yp + (long)r * OutDim, wp, xp + (long)r * InDim, OutDim, InDim);
        }
        if (Bias is not null)
            for (int r = 0; r < m; r++)
            {
                var row = output.Slice(r * OutDim, OutDim);
                TensorPrimitives.Add(row, Bias, row);
            }
    }

    public void Dispose()
    {
        if (_packed != null)
        {
            NativeMemory.AlignedFree(_packed);
            _packed = null;
        }
        GC.SuppressFinalize(this);
    }

    ~PackedLinearF32()
    {
        if (_packed != null) NativeMemory.AlignedFree(_packed);
    }
}
