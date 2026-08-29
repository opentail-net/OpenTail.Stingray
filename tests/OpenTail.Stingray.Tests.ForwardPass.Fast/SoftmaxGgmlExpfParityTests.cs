
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// docs/bugstofix.md (ModelCompatibility.cs:461, deepseek2 investigation): <see cref="SimdKernels.SoftmaxInPlace"/>
/// now accumulates in double and uses a port of ggml's actual vectorized exp (<c>ggml_v_expf</c>)
/// instead of an independently-derived polynomial. This pins that the result still IS a correct
/// softmax (sums to 1, matches a plain double-precision reference closely) -- proving the swap
/// didn't introduce a correctness bug, not proving bit-parity with ggml (which would need ggml
/// itself as the oracle, out of scope for a fast unit test).
/// </summary>
public sealed unsafe class SoftmaxGgmlExpfParityTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(64)]
    [InlineData(65)] // exercises the AVX2 loop plus a scalar remainder
    public void SoftmaxInPlace_MatchesDoublePrecisionReference(int size)
    {
        var rng = new Random(size);
        float[] x = new float[size];
        for (int i = 0; i < size; i++) x[i] = (float)(rng.NextDouble() * 20 - 10);

        double[] reference = new double[size];
        double max = double.NegativeInfinity;
        foreach (var v in x) max = Math.Max(max, v);
        double sum = 0;
        for (int i = 0; i < size; i++)
        {
            reference[i] = Math.Exp((double)x[i] - max);
            sum += reference[i];
        }
        for (int i = 0; i < size; i++) reference[i] /= sum;

        fixed (float* px = x)
        {
            SimdKernels.SoftmaxInPlace(px, size);
        }

        float total = 0;
        for (int i = 0; i < size; i++)
        {
            total += x[i];
            Assert.True(Math.Abs(x[i] - reference[i]) < 1e-6,
                $"index {i}: got {x[i]:G9}, expected {reference[i]:G9} (size={size})");
        }
        Assert.True(Math.Abs(total - 1.0f) < 1e-5f, $"sum={total:G9} (size={size})");
    }
}
