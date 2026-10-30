using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The vectorised Q4_K scale/min unpack (ggml's kmask trick) must give exactly the sixteen values the scalar
/// GetScaleMinK4 decoding gives, for every bit pattern of the 12 packed bytes that matters.
/// </summary>
public sealed unsafe class Q4KScaleUnpackTests
{
    private static (byte Scale, byte Min) Scalar(int j, byte* q) => j < 4
        ? ((byte)(q[j] & 63), (byte)(q[j + 4] & 63))
        : ((byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)));

    [Fact]
    public void LoadQ4KScalesMins_MatchesScalarDecode_OnRandomAndExtremeBlocks()
    {
        Assert.SkipUnless(Avx2.IsSupported, "AVX2 only");
        var rng = new Random(7);
        var sc = new byte[12];
        for (int trial = 0; trial < 20000; trial++)
        {
            if (trial == 0) Array.Fill(sc, (byte)0xFF);
            else if (trial == 1) Array.Fill(sc, (byte)0x00);
            else rng.NextBytes(sc);
            fixed (byte* p = sc)
            {
                SimdKernels.LoadQ4KScalesMins(p, out var scales, out var mins);
                for (int j = 0; j < 8; j++)
                {
                    var (s, m) = Scalar(j, p);
                    Assert.Equal(s, scales.GetElement(j));
                    Assert.Equal(m, mins.GetElement(j));
                }
            }
        }
    }
}

/// <summary>The decode two-row Q4_K kernel must stay bit-identical to two single-row dots (and so to _4In/_8In).</summary>
public sealed unsafe class Q4KTwoRowDotTests
{
    [Theory]
    [InlineData(256, 1)]
    [InlineData(1024, 2)]
    [InlineData(4096, 3)]
    public void TwoRow_BitIdenticalToTwoSingleRowDots(int cols, int seed)
    {
        var rng = new Random(seed);
        int bpr = cols / 256 * 144;
        var w = new byte[2 * bpr];
        rng.NextBytes(w);
        for (int off = 0; off + 144 <= w.Length; off += 144)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.01f + rng.NextSingle() * 0.02f));
            ushort dmin = BitConverter.HalfToUInt16Bits((Half)(0.002f + rng.NextSingle() * 0.01f));
            w[off] = (byte)d; w[off + 1] = (byte)(d >> 8); w[off + 2] = (byte)dmin; w[off + 3] = (byte)(dmin >> 8);
        }
        var x = new float[cols];
        for (int i = 0; i < cols; i++) x[i] = (rng.NextSingle() * 2 - 1) * (i % 97 == 0 ? 8f : 1f);
        var scratch = new byte[SimdKernels.Q8KSScratchBytes(cols)];
        fixed (byte* pw = w)
        fixed (float* px = x)
        fixed (byte* ps = scratch)
        {
            SimdKernels.QuantizeRowToQ8KS(px, cols, ps);
            SimdKernels.DotQ4K_Q8KS_2Row(pw, pw + bpr, ps, cols, out float a, out float b);
            Assert.Equal(SimdKernels.DotQ4K_Q8KS(pw, ps, cols), a);
            Assert.Equal(SimdKernels.DotQ4K_Q8KS(pw + bpr, ps, cols), b);
        }
    }
}
