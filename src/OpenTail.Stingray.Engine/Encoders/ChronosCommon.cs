using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>Chronos <c>ResidualBlock</c> (Bolt and Chronos-2): <c>output_layer(relu(hidden_layer(x))) + residual_layer(x)</c>.</summary>
internal sealed class ChronosResidualBlock(PackedLinearF32 hidden, PackedLinearF32 output, PackedLinearF32 residual) : IDisposable
{
    public int InDim => hidden.InDim;
    public int OutDim => output.OutDim;

    public float[] Forward(float[] x, int rows)
    {
        var h = new float[rows * hidden.OutDim];
        hidden.Forward(x, h, rows);
        TensorPrimitives.Max(h, 0f, h);
        var y = new float[rows * output.OutDim];
        output.Forward(h, y, rows);
        var r = new float[rows * residual.OutDim];
        residual.Forward(x, r, rows);
        TensorPrimitives.Add(y, r, y);
        return y;
    }

    public void Dispose() { hidden.Dispose(); output.Dispose(); residual.Dispose(); }

    public static ChronosResidualBlock Load(SafetensorsLoader st, string name, int inDim, int hDim, int outDim) => new(
        new PackedLinearF32(st.ReadF32(name + ".hidden_layer.weight"), st.ReadF32(name + ".hidden_layer.bias"), hDim, inDim),
        new PackedLinearF32(st.ReadF32(name + ".output_layer.weight"), st.ReadF32(name + ".output_layer.bias"), outDim, hDim),
        new PackedLinearF32(st.ReadF32(name + ".residual_layer.weight"), st.ReadF32(name + ".residual_layer.bias"), outDim, inDim));
}

/// <summary>Chronos <c>InstanceNorm</c>: nan-aware mean / population std (NaN → 0 / 1, zero scale → 1e-5), optional arcsinh.</summary>
internal readonly record struct ChronosInstanceNorm(float Loc, float Scale, bool Arcsinh)
{
    public static ChronosInstanceNorm Fit(ReadOnlySpan<float> x, bool arcsinh)
    {
        double sum = 0;
        int count = 0;
        foreach (float v in x) if (!float.IsNaN(v)) { sum += v; count++; }
        float loc = count == 0 ? 0f : (float)(sum / count);
        double sq = 0;
        foreach (float v in x) if (!float.IsNaN(v)) sq += (double)(v - loc) * (v - loc);
        float scale = count == 0 ? 1f : MathF.Sqrt((float)(sq / count));
        if (scale == 0f) scale = 1e-5f;
        return new ChronosInstanceNorm(loc, scale, arcsinh);
    }

    public float Apply(float v)
    {
        float s = (v - Loc) / Scale;
        return Arcsinh ? MathF.Asinh(s) : s;
    }

    public void InverseInPlace(Span<float> x)
    {
        if (Arcsinh) for (int i = 0; i < x.Length; i++) x[i] = MathF.Sinh(x[i]);
        TensorPrimitives.Multiply(x, Scale, x);
        TensorPrimitives.Add(x, Loc, x);
    }
}
