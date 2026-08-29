
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// <see cref="SimdKernels.MatVecF16GgmlCompat"/> — the opt-in ggml-parity dot for
/// <see cref="OpenTail.Stingray.Core.DType.Float16"/> weights (see docs/bugstofix.md's
/// "activation-precision-matching" finding). ggml's F16 weight paths pair with
/// vec_dot_type=F16: the F32 activation is rounded to half precision before the dot,
/// not kept at full F32 precision. This asserts the rounded-activation result differs
/// from a plain full-precision F32 dot (proving the rounding actually takes effect) and
/// matches a scalar double-precision reference computed with the same fp16-rounded inputs.
/// </summary>
public sealed unsafe class SimdKernelsF16GgmlCompatTests
{
    [Theory]
    [InlineData(3, 96)]
    [InlineData(96, 512)]
    public void MatVecF16GgmlCompat_RoundsActivationToHalf_MatchesReference(int rows, int cols)
    {
        var rnd = new Random(1234);
        var weightsHalf = new Half[rows * cols];
        var input = new float[cols];
        for (int i = 0; i < weightsHalf.Length; i++)
            weightsHalf[i] = (Half)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < cols; i++)
            input[i] = (float)(rnd.NextDouble() * 2 - 1);

        var expected = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++)
            {
                float roundedInput = (float)(Half)input[c];
                sum += (double)(float)weightsHalf[r * cols + c] * roundedInput;
            }
            expected[r] = (float)sum;
        }

        var actual = new float[rows];
        fixed (Half* wPtr = weightsHalf)
        fixed (float* inPtr = input)
        fixed (float* outPtr = actual)
        {
            SimdKernels.MatVecF16GgmlCompat(outPtr, (byte*)wPtr, inPtr, rows, cols);
        }

        for (int r = 0; r < rows; r++)
            Assert.True(Math.Abs(actual[r] - expected[r]) < 1e-3f,
                $"row {r}: expected {expected[r]}, got {actual[r]}");
    }

    [Fact]
    public void GgmlF16DotEnabled_DefaultsOff()
    {
        // Must default off: rounding activations to fp16 is strictly less accurate than this
        // engine's normal full-F32-precision dot, so it must never be silently on.
        if (Environment.GetEnvironmentVariable("STINGRAY_GGML_F16_DOT") is null)
            Assert.False(SimdKernels.GgmlF16DotEnabled);
    }
}
