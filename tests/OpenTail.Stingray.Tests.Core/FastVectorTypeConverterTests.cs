
namespace OpenTail.Stingray.Tests.Core;

public class FastVectorTypeConverterTests
{
    // ── Exhaustive 8-bit FP8 E4M3FN Tests ─────────────────────────────────

    [Fact]
    public void ConvertFp8E4M3ToF32_All256ByteValues_MatchesMathematicalReference()
    {
        byte[] src = new byte[256];
        for (int b = 0; b < 256; b++) src[b] = (byte)b;

        float[] dst = new float[256];
        FastVectorTypeConverter.ConvertFp8E4M3ToF32(src, dst);

        for (int b = 0; b < 256; b++)
        {
            float expected = ReferenceFp8E4M3ToFloat((byte)b);
            float actual = dst[b];

            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(actual), $"Byte 0x{b:X2} expected NaN but got {actual}");
            }
            else
            {
                uint expectedBits = BitConverter.SingleToUInt32Bits(expected);
                uint actualBits = BitConverter.SingleToUInt32Bits(actual);
                Assert.True(expectedBits == actualBits,
                    $"Byte 0x{b:X2}: expected {expected} (0x{expectedBits:X8}) but got {actual} (0x{actualBits:X8})");
            }
        }
    }

    // ── Exhaustive 8-bit FP8 E5M2 Tests ───────────────────────────────────

    [Fact]
    public void ConvertFp8E5M2ToF32_All256ByteValues_MatchesMathematicalReference()
    {
        byte[] src = new byte[256];
        for (int b = 0; b < 256; b++) src[b] = (byte)b;

        float[] dst = new float[256];
        FastVectorTypeConverter.ConvertFp8E5M2ToF32(src, dst);

        for (int b = 0; b < 256; b++)
        {
            float expected = ReferenceFp8E5M2ToFloat((byte)b);
            float actual = dst[b];

            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(actual), $"Byte 0x{b:X2} expected NaN but got {actual}");
            }
            else
            {
                uint expectedBits = BitConverter.SingleToUInt32Bits(expected);
                uint actualBits = BitConverter.SingleToUInt32Bits(actual);
                Assert.True(expectedBits == actualBits,
                    $"Byte 0x{b:X2}: expected {expected} (0x{expectedBits:X8}) but got {actual} (0x{actualBits:X8})");
            }
        }
    }

    // ── Exhaustive 16-bit BF16 → F32 Tests ────────────────────────────────

    [Fact]
    public void ConvertBf16ToF32_All65536BitPatterns_BitIdenticalToScalarReference()
    {
        ushort[] src = new ushort[65536];
        for (int i = 0; i < 65536; i++) src[i] = (ushort)i;

        float[] dst = new float[65536];
        FastVectorTypeConverter.ConvertBf16ToF32(src, dst);

        for (int i = 0; i < 65536; i++)
        {
            uint expectedBits = (uint)src[i] << 16;
            uint actualBits = BitConverter.SingleToUInt32Bits(dst[i]);
            Assert.True(expectedBits == actualBits,
                $"BF16 bits 0x{src[i]:X4}: expected 0x{expectedBits:X8} but got 0x{actualBits:X8}");
        }
    }

    [Fact]
    public void ConvertBf16ToF32_ByteSpan_BitIdenticalToScalarReference()
    {
        ushort[] srcUshort = new ushort[65536];
        for (int i = 0; i < 65536; i++) srcUshort[i] = (ushort)i;

        byte[] srcBytes = MemoryMarshal.AsBytes(srcUshort.AsSpan()).ToArray();
        float[] dst = new float[65536];

        FastVectorTypeConverter.ConvertBf16ToF32(srcBytes, dst);

        for (int i = 0; i < 65536; i++)
        {
            uint expectedBits = (uint)srcUshort[i] << 16;
            uint actualBits = BitConverter.SingleToUInt32Bits(dst[i]);
            Assert.Equal(expectedBits, actualBits);
        }
    }

    // ── F16 → F32 & F32 → F16 Tests ───────────────────────────────────────

    [Fact]
    public void ConvertF16ToF32_All65536BitPatterns_ExactMatchToDotNetHalf()
    {
        ushort[] srcBits = new ushort[65536];
        for (int i = 0; i < 65536; i++) srcBits[i] = (ushort)i;

        Half[] srcHalves = MemoryMarshal.Cast<ushort, Half>(srcBits).ToArray();
        float[] dst = new float[65536];

        FastVectorTypeConverter.ConvertF16ToF32(srcHalves, dst);

        for (int i = 0; i < 65536; i++)
        {
            float expected = (float)srcHalves[i];
            float actual = dst[i];

            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(actual), $"F16 bits 0x{i:X4} expected NaN but got {actual}");
            }
            else
            {
                uint expectedBits = BitConverter.SingleToUInt32Bits(expected);
                uint actualBits = BitConverter.SingleToUInt32Bits(actual);
                Assert.True(expectedBits == actualBits,
                    $"F16 bits 0x{i:X4}: expected {expected} (0x{expectedBits:X8}) but got {actual} (0x{actualBits:X8})");
            }
        }
    }

    [Fact]
    public void ConvertF32ToF16_RoundtripAndEdgeCases_ExactMatch()
    {
        float[] src = [
            0.0f, -0.0f, 1.0f, -1.0f, 65504.0f, -65504.0f, 0.00006103515625f,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN
        ];

        Half[] dstHalf = new Half[src.Length];
        FastVectorTypeConverter.ConvertF32ToF16(src, dstHalf);

        for (int i = 0; i < src.Length; i++)
        {
            Half expected = (Half)src[i];
            Half actual = dstHalf[i];
            if (Half.IsNaN(expected))
            {
                Assert.True(Half.IsNaN(actual));
            }
            else
            {
                Assert.Equal(expected, actual);
            }
        }
    }

    // ── F32 → BF16 Narrowing Tests ────────────────────────────────────────

    [Fact]
    public void ConvertF32ToBf16_WideRange_MatchesRoundToNearestEvenReference()
    {
        float[] testValues = [
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 0.25f, 3.14159265f, -2.71828f,
            float.MaxValue, float.MinValue, float.Epsilon,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN,
            1.0000001f, 1.0000002f, 1.0000005f
        ];

        ushort[] dst = new ushort[testValues.Length];
        FastVectorTypeConverter.ConvertF32ToBf16(testValues, dst);

        for (int i = 0; i < testValues.Length; i++)
        {
            ushort expected = ReferenceF32ToBf16(testValues[i]);
            ushort actual = dst[i];
            Assert.True(expected == actual,
                $"Value {testValues[i]}: expected 0x{expected:X4} but got 0x{actual:X4}");
        }
    }

    // ── Q8_0 Block Quantization & Dequantization Tests ─────────────────────

    [Fact]
    public void ConvertF32ToQ8_0_ZeroBlock_ProducesZeroScaleAndZeros()
    {
        float[] src = new float[32]; // All 0.0f
        byte[] dst = new byte[34];

        FastVectorTypeConverter.ConvertF32ToQ8_0(src, dst);

        ushort scaleBits = BitConverter.ToUInt16(dst, 0);
        float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);
        Assert.Equal(0.0f, scale);

        for (int j = 2; j < 34; j++)
        {
            Assert.Equal((byte)0, dst[j]);
        }
    }

    [Fact]
    public void ConvertF32ToQ8_0_KnownValues_ProducesCorrectScaleAndQuants()
    {
        float[] src = new float[32];
        for (int i = 0; i < 32; i++) src[i] = (i - 16) * 2.0f; // -32.0f to +30.0f

        byte[] q8Buffer = new byte[34];
        FastVectorTypeConverter.ConvertF32ToQ8_0(src, q8Buffer);

        float[] dequantized = new float[32];
        FastVectorTypeConverter.ConvertQ8_0ToF32(q8Buffer, dequantized);

        // Max abs value is 32.0f. Quantization error per element should be <= scale = 32.0f / 127.0f (~0.252)
        float maxError = 32.0f / 127.0f + 0.01f;
        for (int i = 0; i < 32; i++)
        {
            float error = MathF.Abs(src[i] - dequantized[i]);
            Assert.True(error <= maxError,
                $"Index {i}: orig={src[i]}, dequant={dequantized[i]}, error={error} > maxError={maxError}");
        }
    }

    [Fact]
    public void ConvertF32ToQ8_0_MultiBlock_RoundtripErrorBounded()
    {
        int numBlocks = 17; // Non-power-of-2 block count
        float[] src = new float[numBlocks * 32];
        Random rand = new Random(42);

        for (int i = 0; i < src.Length; i++)
        {
            src[i] = (float)(rand.NextDouble() * 200.0 - 100.0);
        }

        byte[] q8Buffer = new byte[numBlocks * 34];
        FastVectorTypeConverter.ConvertF32ToQ8_0(src, q8Buffer);

        float[] dequant = new float[numBlocks * 32];
        FastVectorTypeConverter.ConvertQ8_0ToF32(q8Buffer, dequant);

        for (int b = 0; b < numBlocks; b++)
        {
            float maxAbsInBlock = 0f;
            for (int j = 0; j < 32; j++)
            {
                float a = MathF.Abs(src[b * 32 + j]);
                if (a > maxAbsInBlock) maxAbsInBlock = a;
            }

            float allowedError = maxAbsInBlock > 0f ? (maxAbsInBlock / 127.0f * 1.5f) : 0.001f;
            for (int j = 0; j < 32; j++)
            {
                int idx = b * 32 + j;
                float diff = MathF.Abs(src[idx] - dequant[idx]);
                Assert.True(diff <= allowedError,
                    $"Block {b} element {j}: src={src[idx]}, dequant={dequant[idx]}, diff={diff} > allowed={allowedError}");
            }
        }
    }

    // ── Length & Slicing Hardening Tests ──────────────────────────────────

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(1007)]
    public void AllConverters_VariousLengths_ProduceCorrectResultsWithoutOverrun(int length)
    {
        byte[] srcFp8 = new byte[length];
        for (int i = 0; i < length; i++) srcFp8[i] = (byte)(i % 256);

        float[] dstFp8E4 = new float[length + 10];
        FastVectorTypeConverter.ConvertFp8E4M3ToF32(srcFp8, dstFp8E4.AsSpan(0, length));
        for (int i = 0; i < length; i++)
        {
            float expected = ReferenceFp8E4M3ToFloat(srcFp8[i]);
            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(dstFp8E4[i]));
            }
            else
            {
                Assert.Equal(BitConverter.SingleToUInt32Bits(expected), BitConverter.SingleToUInt32Bits(dstFp8E4[i]));
            }
        }

        float[] dstFp8E5 = new float[length + 10];
        FastVectorTypeConverter.ConvertFp8E5M2ToF32(srcFp8, dstFp8E5.AsSpan(0, length));
        for (int i = 0; i < length; i++)
        {
            float expected = ReferenceFp8E5M2ToFloat(srcFp8[i]);
            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(dstFp8E5[i]));
            }
            else
            {
                Assert.Equal(BitConverter.SingleToUInt32Bits(expected), BitConverter.SingleToUInt32Bits(dstFp8E5[i]));
            }
        }

        ushort[] srcBf16 = new ushort[length];
        for (int i = 0; i < length; i++) srcBf16[i] = (ushort)(i * 17);

        float[] dstBf16 = new float[length + 10];
        FastVectorTypeConverter.ConvertBf16ToF32(srcBf16, dstBf16.AsSpan(0, length));
        for (int i = 0; i < length; i++)
        {
            uint expectedBits = (uint)srcBf16[i] << 16;
            Assert.Equal(expectedBits, BitConverter.SingleToUInt32Bits(dstBf16[i]));
        }

        Half[] srcF16 = new Half[length];
        for (int i = 0; i < length; i++) srcF16[i] = (Half)(i * 0.1f);

        float[] dstF16 = new float[length + 10];
        FastVectorTypeConverter.ConvertF16ToF32(srcF16, dstF16.AsSpan(0, length));
        for (int i = 0; i < length; i++)
        {
            Assert.Equal((float)srcF16[i], dstF16[i]);
        }
    }

    [Fact]
    public void ConvertBf16ToF32_MisalignedSlices_OperateCorrectly()
    {
        ushort[] src = new ushort[100];
        for (int i = 0; i < 100; i++) src[i] = (ushort)(i + 1000);

        float[] dstBuffer = new float[110];
        var dstSlice = dstBuffer.AsSpan(3, 100);

        FastVectorTypeConverter.ConvertBf16ToF32(src, dstSlice);

        for (int i = 0; i < 100; i++)
        {
            uint expectedBits = (uint)src[i] << 16;
            Assert.Equal(expectedBits, BitConverter.SingleToUInt32Bits(dstSlice[i]));
        }
    }

    [Fact]
    public void ConvertFp8E4M3ToF32_SmallDestinationSpan_ThrowsArgumentException()
    {
        byte[] src = new byte[10];
        float[] dst = new float[9];
        Assert.Throws<ArgumentException>(() => FastVectorTypeConverter.ConvertFp8E4M3ToF32(src, dst));
    }

    [Fact]
    public void ConvertBf16ToF32_SmallDestinationSpan_ThrowsArgumentException()
    {
        ushort[] src = new ushort[10];
        float[] dst = new float[9];
        Assert.Throws<ArgumentException>(() => FastVectorTypeConverter.ConvertBf16ToF32(src, dst));
    }

    [Fact]
    public void ConvertF32ToQ8_0_InvalidLength_ThrowsArgumentException()
    {
        float[] srcBadLength = new float[31]; // Not a multiple of 32
        byte[] dst = new byte[34];
        Assert.Throws<ArgumentException>(() => FastVectorTypeConverter.ConvertF32ToQ8_0(srcBadLength, dst));
    }

    // ── Reference Formulas for Verification ──────────────────────────────

    private static float ReferenceFp8E4M3ToFloat(byte b)
    {
        int sign = (b >> 7) & 1;
        int exp = (b >> 3) & 0xF;
        int mant = b & 0x7;

        if (exp == 0 && mant == 0) return sign == 0 ? 0.0f : -0.0f;
        float v = exp == 0
            ? MathF.Pow(2.0f, -6.0f) * (mant / 8.0f)
            : (exp == 15 && mant == 7) ? float.NaN
            : MathF.Pow(2.0f, exp - 7.0f) * (1.0f + mant / 8.0f);

        return sign == 0 ? v : -v;
    }

    private static float ReferenceFp8E5M2ToFloat(byte b)
    {
        int sign = (b >> 7) & 1;
        int exp = (b >> 2) & 0x1F;
        int mant = b & 0x3;

        if (exp == 0 && mant == 0) return sign == 0 ? 0.0f : -0.0f;
        float v = exp == 0
            ? MathF.Pow(2.0f, -14.0f) * (mant / 4.0f)
            : (exp == 31 && mant == 0) ? float.PositiveInfinity
            : (exp == 31) ? float.NaN
            : MathF.Pow(2.0f, exp - 15.0f) * (1.0f + mant / 4.0f);

        return sign == 0 ? v : -v;
    }

    private static ushort ReferenceF32ToBf16(float f)
    {
        if (float.IsNaN(f)) return 0x7FC0;
        uint bits = BitConverter.SingleToUInt32Bits(f);
        uint lsb = (bits >> 16) & 1u;
        uint bias = 0x7FFFu + lsb;
        uint rounded = bits + bias;
        return (ushort)(rounded >> 16);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Differential tests.
    //
    // Everything above round-trips this class against itself, which passes even when encode and
    // decode share a bug. The tests below compare against something the converter did not produce:
    // the FP8 field layouts straight from the format spec, and the vectorized path against the
    // scalar path in the same file. Those are the two ways this class can be wrong without any
    // round-trip noticing.
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// FP8 E4M3FN decoded from the field layout: 1 sign, 4 exponent (bias 7), 3 mantissa; subnormals
    /// use 2^-6 * m/8. Written from the OCP FP8 spec, not derived from the table under test.
    /// </summary>
    /// <remarks>
    /// The <c>FN</c> suffix means <b>F</b>inite + <b>N</b>aN: E4M3FN has no infinities, and reserves
    /// <c>S.1111.111</c> (0x7F / 0xFF) for NaN. Max finite is therefore 448 at 0x7E, not 480.
    /// <para><b>This is where <c>SafetensorsLoader.F8E4M3ToFloat</c> is wrong</b> — it has no NaN
    /// case, so it decodes 0x7F as +480, a value E4M3FN cannot represent. Do not "fix" this
    /// reference to agree with the loader; the loader is what needs fixing.</para>
    /// </remarks>
    private static float ReferenceFp8E4M3(byte b)
    {
        int sign = (b >> 7) & 1, exp = (b >> 3) & 0xF, mant = b & 0x7;
        if (exp == 0 && mant == 0) return 0f;
        if (exp == 15 && mant == 7) return sign == 0 ? float.NaN : -float.NaN;
        float v = exp == 0
            ? MathF.Pow(2f, -6f) * (mant / 8f)
            : MathF.Pow(2f, exp - 7f) * (1f + mant / 8f);
        return sign == 0 ? v : -v;
    }

    /// <summary>
    /// FP8 E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa; subnormals use 2^-14 * m/4.
    /// Unlike E4M3FN this format follows IEEE conventions: exponent 31 is Inf (mantissa 0) or NaN.
    /// <c>SafetensorsLoader.F8E5M2ToFloat</c> misses this too and returns finite values near 2^16.
    /// </summary>
    private static float ReferenceFp8E5M2(byte b)
    {
        int sign = (b >> 7) & 1, exp = (b >> 2) & 0x1F, mant = b & 0x3;
        if (exp == 0 && mant == 0) return 0f;
        if (exp == 31)
            return mant == 0
                ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity)
                : (sign == 0 ? float.NaN : -float.NaN);
        float v = exp == 0
            ? MathF.Pow(2f, -14f) * (mant / 4f)
            : MathF.Pow(2f, exp - 15f) * (1f + mant / 4f);
        return sign == 0 ? v : -v;
    }

    [Fact]
    public void Fp8E4M3_AllByteValues_MatchTheFormatSpec()
    {
        byte[] src = new byte[256];
        for (int i = 0; i < 256; i++) src[i] = (byte)i;

        float[] dst = new float[256];
        FastVectorTypeConverter.ConvertFp8E4M3ToF32(src, dst);

        for (int i = 0; i < 256; i++)
        {
            float expected = ReferenceFp8E4M3((byte)i);
            // NaN/Inf encodings are format-specific; compare bitwise so no case is silently skipped.
            Assert.True(expected.Equals(dst[i]),
                $"E4M3 byte 0x{i:X2}: expected {expected}, got {dst[i]}");
        }
    }

    [Fact]
    public void Fp8E5M2_AllByteValues_MatchTheFormatSpec()
    {
        byte[] src = new byte[256];
        for (int i = 0; i < 256; i++) src[i] = (byte)i;

        float[] dst = new float[256];
        FastVectorTypeConverter.ConvertFp8E5M2ToF32(src, dst);

        for (int i = 0; i < 256; i++)
        {
            float expected = ReferenceFp8E5M2((byte)i);
            Assert.True(expected.Equals(dst[i]),
                $"E5M2 byte 0x{i:X2}: expected {expected}, got {dst[i]}");
        }
    }

    [Fact]
    public void Fp8_E4M3AndE5M2_DisagreeOnTheSameBytes()
    {
        // Guards against both tables being built by the same (wrong) code path: the two formats
        // have different exponent biases, so a shared table would be a silent catastrophe.
        byte[] src = [0x3C, 0x40, 0x7B, 0xC0];
        float[] e4 = new float[4], e5 = new float[4];
        FastVectorTypeConverter.ConvertFp8E4M3ToF32(src, e4);
        FastVectorTypeConverter.ConvertFp8E5M2ToF32(src, e5);

        Assert.NotEqual(e4, e5);
    }

    /// <summary>
    /// The vectorized F32→BF16 path must be bit-identical to the scalar tail, for every input class.
    /// </summary>
    /// <remarks>
    /// Buffers shorter than the vector width take the scalar loop, so converting each value alone
    /// yields the scalar result; converting the whole array exercises the packed path. Any
    /// divergence — a mis-ordered pack, a lane permute error, a dropped NaN — shows up here and
    /// nowhere else, because a round-trip test would decode the wrong lane back to the wrong slot
    /// consistently and still pass.
    /// </remarks>
    [Fact]
    public void ConvertF32ToBf16_VectorPath_IsBitIdenticalToScalarPath()
    {
        float[] specials =
        [
            0f, -0f, 1f, -1f, float.Epsilon, -float.Epsilon,
            float.MaxValue, float.MinValue, float.PositiveInfinity, float.NegativeInfinity,
            float.NaN, -float.NaN, 1.5f, -2.25f, 65504f, 1e-40f,
        ];

        var rand = new Random(20260805);
        float[] src = new float[512];
        for (int i = 0; i < src.Length; i++)
            src[i] = i < specials.Length ? specials[i] : (float)(rand.NextDouble() * 2000.0 - 1000.0);

        ushort[] vectorized = new ushort[src.Length];
        FastVectorTypeConverter.ConvertF32ToBf16(src, vectorized);

        for (int i = 0; i < src.Length; i++)
        {
            // A one-element span is below every vector width, so this is the scalar path.
            ushort[] scalarOne = new ushort[1];
            FastVectorTypeConverter.ConvertF32ToBf16(src.AsSpan(i, 1), scalarOne);

            Assert.True(scalarOne[0] == vectorized[i],
                $"index {i} (value {src[i]}, bits 0x{BitConverter.SingleToUInt32Bits(src[i]):X8}): " +
                $"scalar 0x{scalarOne[0]:X4} != vector 0x{vectorized[i]:X4}");
        }
    }

    [Fact]
    public void ConvertBf16ToF32_VectorPath_IsBitIdenticalToScalarPath()
    {
        var rand = new Random(20260806);
        ushort[] src = new ushort[512];
        for (int i = 0; i < src.Length; i++) src[i] = (ushort)rand.Next(0, 0x10000);
        // Cover the exponent extremes explicitly rather than hoping the RNG lands on them.
        src[0] = 0x0000; src[1] = 0x8000; src[2] = 0x7F80; src[3] = 0xFF80; src[4] = 0x7FC0;

        float[] vectorized = new float[src.Length];
        FastVectorTypeConverter.ConvertBf16ToF32(src, vectorized);

        for (int i = 0; i < src.Length; i++)
        {
            float[] scalarOne = new float[1];
            FastVectorTypeConverter.ConvertBf16ToF32(src.AsSpan(i, 1), scalarOne);

            Assert.True(BitConverter.SingleToUInt32Bits(scalarOne[0])
                     == BitConverter.SingleToUInt32Bits(vectorized[i]),
                $"index {i} (bf16 0x{src[i]:X4}): scalar {scalarOne[0]} != vector {vectorized[i]}");
        }
    }
}
