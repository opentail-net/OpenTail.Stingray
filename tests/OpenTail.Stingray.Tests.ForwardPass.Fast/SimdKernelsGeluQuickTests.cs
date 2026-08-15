using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Scalar-reference tests for <see cref="SimdKernels.GeluQuickInPlace"/> -- the "quick" GELU
/// (<c>x * sigmoid(1.702*x)</c>), NOT the tanh-approximation <see cref="SimdKernels.GeluInPlace"/>
/// already used elsewhere. Added for the Gemma 4 E4B <c>gemma4v</c> ViT encoder
/// (<see cref="OpenTail.Stingray.Vision.Gemma4VVisionEncoder"/>), whose FFN activation is
/// <c>FFN_GELU_QUICK</c> per the real reference (llama.cpp's <c>tools/mtmd/clip.cpp</c>) -- see
/// docs/03-gemma4-e4b-vision-plan.md for the derivation. Reusing the existing tanh-GELU kernel
/// would have been silently wrong for this activation variant, which is exactly why this gets its
/// own dedicated test rather than only exercising it inside the encoder's own structural check.
/// </summary>
public sealed unsafe class SimdKernelsGeluQuickTests
{
    private static float ScalarReference(float x) => x / (1f + MathF.Exp(-1.702f * x));

    [Fact]
    public void GeluQuickInPlace_MatchesScalarReference()
    {
        var rng = new Random(unchecked((int)0xC0FFEE));
        var x = new float[257];
        var expected = new float[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            x[i] = (float)(rng.NextDouble() * 20 - 10); // [-10, 10]
            expected[i] = ScalarReference(x[i]);
        }

        fixed (float* p = x)
            SimdKernels.GeluQuickInPlace(p, x.Length);

        for (var i = 0; i < x.Length; i++)
            Assert.Equal(expected[i], x[i], 5);
    }

    [Fact]
    public void GeluQuickInPlace_ZeroMapsToZero()
    {
        var x = new float[] { 0f };
        fixed (float* p = x)
            SimdKernels.GeluQuickInPlace(p, 1);
        Assert.Equal(0f, x[0]);
    }

    [Fact]
    public void GeluQuickInPlace_LargePositive_ApproachesIdentity()
    {
        // sigmoid(1.702*x) -> 1 for large positive x, so gelu_quick(x) -> x.
        var x = new float[] { 50f };
        fixed (float* p = x)
            SimdKernels.GeluQuickInPlace(p, 1);
        Assert.Equal(50f, x[0], 3);
    }

    [Fact]
    public void GeluQuickInPlace_LargeNegative_ApproachesZero()
    {
        // sigmoid(1.702*x) -> 0 for large negative x, so gelu_quick(x) -> 0.
        var x = new float[] { -50f };
        fixed (float* p = x)
            SimdKernels.GeluQuickInPlace(p, 1);
        Assert.Equal(0f, x[0], 3);
    }

    [Fact]
    public void GeluQuickInPlace_DiffersFromTanhGelu()
    {
        // Sanity guard against accidentally aliasing the two kernels: they must NOT agree away
        // from x=0, or a future edit could silently swap one implementation for the other.
        var quick = new float[] { 2f };
        var tanh = new float[] { 2f };
        fixed (float* pq = quick, pt = tanh)
        {
            SimdKernels.GeluQuickInPlace(pq, 1);
            SimdKernels.GeluInPlace(pt, 1);
        }
        Assert.True(MathF.Abs(quick[0] - tanh[0]) > 0.01f,
            $"quick-GELU ({quick[0]}) and tanh-GELU ({tanh[0]}) should diverge at x=2, not agree.");
    }
}
