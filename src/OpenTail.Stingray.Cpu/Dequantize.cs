using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Scalar dequantization routines for GGML quantized formats.
/// Matches the reference implementation in ggml-quants.c exactly.
/// </summary>
public static class Dequantize
{
    /// <summary>
    /// Dequantize a tensor from any supported quantized format to Float32.
    /// </summary>
    public static void ToFloat32(ReadOnlySpan<byte> src, Span<float> dst, DType dtype, long elementCount)
    {
        switch (dtype)
        {
            case DType.Float32:
                MemoryMarshal.Cast<byte, float>(src).Slice(0, (int)elementCount).CopyTo(dst);
                break;
            case DType.Float16:
                DequantF16(src, dst, elementCount);
                break;
            case DType.BFloat16:
                DequantBF16(src, dst, elementCount);
                break;
            case DType.Q8_0:
                DequantQ8_0(src, dst, elementCount);
                break;
            case DType.Q8_1:
                DequantQ8_1(src, dst, elementCount);
                break;
            case DType.Q4_0:
                DequantQ4_0(src, dst, elementCount);
                break;
            case DType.Q4_1:
                DequantQ4_1(src, dst, elementCount);
                break;
            case DType.Q5_0:
                DequantQ5_0(src, dst, elementCount);
                break;
            case DType.Q5_1:
                DequantQ5_1(src, dst, elementCount);
                break;
            case DType.Q4_K:
                DequantQ4K(src, dst, elementCount);
                break;
            case DType.Q6_K:
                DequantQ6K(src, dst, elementCount);
                break;
            case DType.Q5_K:
                DequantQ5K(src, dst, elementCount);
                break;
            case DType.Q2_K:
                DequantQ2K(src, dst, elementCount);
                break;
            case DType.Q3_K:
                DequantQ3K(src, dst, elementCount);
                break;
            case DType.MXFP4:
                DequantMxfp4(src, dst, elementCount);
                break;
            case DType.NVFP4:
                DequantNvfp4(src, dst, elementCount);
                break;
            case DType.Q1_0:
                DequantQ1_0(src, dst, elementCount);
                break;
            case DType.Q2_0:
                DequantQ2_0(src, dst, elementCount);
                break;
            case DType.IQ4_NL:
                DequantIq4Nl(src, dst, elementCount);
                break;
            case DType.IQ2_S:
                DequantIq2S(src, dst, elementCount);
                break;
            case DType.IQ2_XS:
                DequantIq2Xs(src, dst, elementCount);
                break;
            case DType.IQ2_XXS:
                DequantIq2Xxs(src, dst, elementCount);
                break;
            case DType.IQ3_XXS:
                DequantIq3Xxs(src, dst, elementCount);
                break;
            case DType.IQ3_S:
                DequantIq3S(src, dst, elementCount);
                break;
            case DType.IQ4_XS:
                DequantIq4Xs(src, dst, elementCount);
                break;
            case DType.IQ1_S:
                DequantIq1S(src, dst, elementCount);
                break;
            case DType.IQ1_M:
                DequantIq1M(src, dst, elementCount);
                break;
            default:
                throw new NotSupportedException($"Dequantization not implemented for {dtype}");
        }
    }

    // MXFP4 decoding below is derived from TensorSharp's ManagedQuantizedOps.
    // Copyright (c) 2026 Zhongkai Fu. BSD-3-Clause; see THIRD_PARTY_NOTICES.md.
    // The block layout is one E8M0 scale byte followed by sixteen packed 4-bit values.
    private static void DequantMxfp4(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 32;
        const int bytesPerBlock = 17;

        if (elementCount % elementsPerBlock != 0)
            throw new ArgumentException($"MXFP4 requires {elementsPerBlock}-element alignment, got {elementCount}.", nameof(elementCount));

        long blockCount = elementCount / elementsPerBlock;
        for (long block = 0; block < blockCount; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float scale = E8M0ToSingle(x[0]);
            ReadOnlySpan<byte> quants = x.Slice(1);

            for (int i = 0; i < quants.Length; i++)
            {
                byte packed = quants[i];
                y[i] = scale * Mxfp4Value(packed & 0x0F);
                y[i + quants.Length] = scale * Mxfp4Value(packed >> 4);
            }
        }
    }

    private static float Mxfp4Value(int value) => value switch
    {
        0 => 0f, 1 => 1f, 2 => 2f, 3 => 3f, 4 => 4f, 5 => 6f, 6 => 8f, 7 => 12f,
        8 => 0f, 9 => -1f, 10 => -2f, 11 => -3f, 12 => -4f, 13 => -6f, 14 => -8f, 15 => -12f,
        _ => throw new ArgumentOutOfRangeException(nameof(value)),
    };

    private static float E8M0ToSingle(byte value)
    {
        uint bits = value < 2 ? 0x00200000u << value : ((uint)value - 1u) << 23;
        return BitConverter.Int32BitsToSingle((int)bits);
    }

    // NVFP4, Q1_0 and Q2_0 layouts below follow the llama.cpp / ggml reference
    // definitions (MIT; see THIRD_PARTY_NOTICES.md). They deliberately start as
    // scalar decoders: SimdKernels' existing dequant-to-F32 fallback makes them
    // executable on the portable CPU path before specialised AVX/GPU kernels exist.
    private static void DequantNvfp4(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 64;
        const int elementsPerSubBlock = 16;
        const int bytesPerBlock = 36; // four UE4M3 scales followed by 32 packed E2M1 values
        if (elementCount % elementsPerBlock != 0)
            throw new ArgumentException($"NVFP4 requires {elementsPerBlock}-element alignment, got {elementCount}.", nameof(elementCount));

        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            for (int sub = 0; sub < 4; sub++)
            {
                float scale = Ue4M3ToSingle(x[sub]);
                ReadOnlySpan<byte> quants = x.Slice(4 + sub * 8, 8);
                int offset = sub * elementsPerSubBlock;
                for (int i = 0; i < quants.Length; i++)
                {
                    byte packed = quants[i];
                    y[offset + i] = scale * Mxfp4Value(packed & 0x0F);
                    y[offset + i + quants.Length] = scale * Mxfp4Value(packed >> 4);
                }
            }
        }
    }

    private static void DequantQ1_0(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 128;
        const int bytesPerBlock = 18;
        if (elementCount % elementsPerBlock != 0)
            throw new ArgumentException($"Q1_0 requires {elementsPerBlock}-element alignment, got {elementCount}.", nameof(elementCount));

        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float scale = HalfToFloat(x[0], x[1]);
            for (int i = 0; i < elementsPerBlock; i++)
                y[i] = (x[2 + i / 8] & (1 << (i % 8))) != 0 ? scale : -scale;
        }
    }

    private static void DequantQ2_0(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 64;
        const int bytesPerBlock = 18;
        if (elementCount % elementsPerBlock != 0)
            throw new ArgumentException($"Q2_0 requires {elementsPerBlock}-element alignment, got {elementCount}.", nameof(elementCount));

        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float scale = HalfToFloat(x[0], x[1]);
            for (int i = 0; i < elementsPerBlock; i++)
                y[i] = (((x[2 + i / 4] >> ((i % 4) * 2)) & 0x03) - 1) * scale;
        }
    }

    // Unsigned E4M3 scale, with ggml's required half-scale convention because
    // the shared E2M1 lookup table stores values doubled. 0x7f is the reserved NaN.
    private static float Ue4M3ToSingle(byte value)
    {
        if (value is 0 or 0x7F) return 0f;
        int exponent = (value >> 3) & 0x0F;
        int mantissa = value & 0x07;
        float raw = exponent == 0
            ? MathF.ScaleB(mantissa, -9)
            : MathF.ScaleB(1f + mantissa / 8f, exponent - 7);
        return raw * 0.5f;
    }

    /// <summary>
    /// Q4_K dequantization. Block size = 256, type size = 144 bytes.
    /// Layout per block (block_q4_K in ggml):
    ///   - 2 bytes: FP16 d (super-block scale)
    ///   - 2 bytes: FP16 dmin (super-block min)
    ///   - 12 bytes: packed 6-bit scales and mins
    ///   - 128 bytes: 4-bit quantized values
    ///
    /// Reference: dequantize_row_q4_K in ggml-quants.c
    /// </summary>
    private static void DequantQ4K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 144;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            var y = dst.Slice((int)(block * QK_K), QK_K);

            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);

            var scales = x.Slice(4, 12);  // K_SCALE_SIZE = 12
            var qs = x.Slice(16, 128);    // QK_K/2 = 128

            int qIdx = 0;
            int scaleIdx = 0;

            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(scaleIdx, scales, out byte sc1, out byte m1);
                float d1 = d * sc1;
                float dm1 = dmin * m1;
                GetScaleMinK4(scaleIdx + 1, scales, out byte sc2, out byte m2);
                float d2 = d * sc2;
                float dm2 = dmin * m2;

                for (int l = 0; l < 32; l++)
                {
                    y[j + l] = d1 * (qs[qIdx + l] & 0xF) - dm1;
                    y[j + l + 32] = d2 * (qs[qIdx + l] >> 4) - dm2;
                }
                qIdx += 32;
                scaleIdx += 2;
            }
        }
    }

    /// <summary>
    /// Decode one 6-bit scale and min from the packed 12-byte scale/min buffer.
    /// Matches get_scale_min_k4 in ggml-quants.c.
    /// </summary>
    private static void GetScaleMinK4(int j, ReadOnlySpan<byte> q, out byte scale, out byte min)
    {
        if (j < 4)
        {
            scale = (byte)(q[j] & 63);
            min = (byte)(q[j + 4] & 63);
        }
        else
        {
            scale = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            min = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    /// <summary>
    /// Q6_K dequantization. Block size = 256, type size = 210 bytes.
    /// Layout per block (block_q6_K in ggml):
    ///   - 128 bytes: ql — lower 4 bits of 6-bit quants
    ///   - 64 bytes: qh — upper 2 bits of 6-bit quants
    ///   - 16 bytes: int8 scales (one per 16-element sub-block)
    ///   - 2 bytes: FP16 d (super-block scale)
    ///
    /// Reference: dequantize_row_q6_K in ggml-quants.c
    /// </summary>
    private static void DequantQ6K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 210;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            var y = dst.Slice((int)(block * QK_K), QK_K);

            float d = HalfToFloat(x[208], x[209]);

            int qlOff = 0;   // into ql (bytes 0..127)
            int qhOff = 128; // into qh (bytes 128..191)
            int scOff = 192; // into scales (bytes 192..207)
            int yOff = 0;

            int scBase = 0;
            for (int n = 0; n < QK_K; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int isc = l / 16; // 0 for l<16, 1 for l>=16

                    int q1 = ((x[qlOff + l] & 0xF) | (((x[qhOff + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((x[qlOff + l + 32] & 0xF) | (((x[qhOff + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((x[qlOff + l] >> 4) | (((x[qhOff + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((x[qlOff + l + 32] >> 4) | (((x[qhOff + l] >> 6) & 3) << 4)) - 32;

                    y[yOff + l] = d * (sbyte)x[scOff + scBase + isc] * q1;
                    y[yOff + l + 32] = d * (sbyte)x[scOff + scBase + isc + 2] * q2;
                    y[yOff + l + 64] = d * (sbyte)x[scOff + scBase + isc + 4] * q3;
                    y[yOff + l + 96] = d * (sbyte)x[scOff + scBase + isc + 6] * q4;
                }
                yOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }
        }
    }

    /// <summary>
    /// Q5_K dequantization. Block size = 256, type size = 176 bytes.
    /// Layout per block (block_q5_K in ggml):
    ///   - 2 bytes: FP16 d (super-block scale)
    ///   - 2 bytes: FP16 dmin (super-block min)
    ///   - 12 bytes: packed 6-bit scales and mins (same as Q4_K)
    ///   - 32 bytes: qh — high bits (one bit per element, packed)
    ///   - 128 bytes: ql — lower 4 bits (two elements per byte)
    ///
    /// Reference: dequantize_row_q5_K in ggml-quants.c
    /// </summary>
    private static void DequantQ5K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 176;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            var y = dst.Slice((int)(block * QK_K), QK_K);

            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);

            var scales = x.Slice(4, 12);  // K_SCALE_SIZE = 12
            var qh = x.Slice(16, 32);     // high bits: 256 bits = 32 bytes
            var ql = x.Slice(48, 128);    // QK_K/2 = 128

            int qIdx = 0;
            int scaleIdx = 0;

            // qh bit layout per byte: bits 0,1 for j=0; bits 2,3 for j=64;
            // bits 4,5 for j=128; bits 6,7 for j=192.
            // u1 masks the low-nibble high bit, u2 masks the high-nibble high bit.
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(scaleIdx, scales, out byte sc1, out byte m1);
                float d1 = d * sc1;
                float dm1 = dmin * m1;
                GetScaleMinK4(scaleIdx + 1, scales, out byte sc2, out byte m2);
                float d2 = d * sc2;
                float dm2 = dmin * m2;

                for (int l = 0; l < 32; l++)
                {
                    int hLo = (qh[l] & u1) != 0 ? 16 : 0;
                    int hHi = (qh[l] & u2) != 0 ? 16 : 0;
                    y[j + l] = d1 * ((ql[qIdx + l] & 0xF) + hLo) - dm1;
                    y[j + l + 32] = d2 * ((ql[qIdx + l] >> 4) + hHi) - dm2;
                }
                qIdx += 32;
                scaleIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    /// <summary>
    /// Q2_K dequantization. Block size = 256, type size = 84 bytes.
    /// Layout per block (block_q2_K in ggml):
    ///   - 16 bytes: scales (4 bits each, packed as nibbles)
    ///   - 64 bytes: qs (2-bit quantized values, 4 per byte)
    ///   - 2 bytes: FP16 d (super-block scale)
    ///   - 2 bytes: FP16 dmin (super-block min)
    /// Reference: dequantize_row_q2_K in ggml-quants.c
    /// </summary>
    /// <summary>
    /// Q2_K: matches ggml dequantize_row_q2_K exactly.
    /// Layout: [scales:16][qs:64][d:FP16][dmin:FP16] = 84 bytes / 256 elements.
    /// The 64 qs bytes are read 4 times with shifts 0,2,4,6 per 128-element group.
    /// </summary>
    private static void DequantQ2K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 84;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            int yOff = (int)(block * QK_K);

            float d = HalfToFloat(x[80], x[81]);
            float min = HalfToFloat(x[82], x[83]);

            int qOff = 16; // qs at byte 16
            int isIdx = 0;
            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte sc = x[isIdx++]; // scales at byte 0
                    float dl = d * (sc & 0xF);
                    float ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        dst[yOff++] = dl * ((x[qOff + l] >> shift) & 3) - ml;

                    sc = x[isIdx++];
                    dl = d * (sc & 0xF);
                    ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        dst[yOff++] = dl * ((x[qOff + l + 16] >> shift) & 3) - ml;

                    shift += 2;
                }
                qOff += 32;
            }
        }
    }

    /// <summary>
    /// Q3_K dequantization. Block size = 256, type size = 110 bytes.
    /// Layout per block (block_q3_K in ggml):
    ///   - 32 bytes: hmask (high bit per element)
    ///   - 64 bytes: qs (lower 2 bits, 4 per byte)
    ///   - 12 bytes: packed scales
    ///   - 2 bytes: FP16 d
    /// Reference: dequantize_row_q3_K in ggml-quants.c
    /// </summary>
    /// <summary>
    /// Q3_K: matches ggml dequantize_row_q3_K exactly.
    /// Layout: [hmask:32][qs:64][scales:12][d:FP16] = 110 bytes / 256 elements.
    /// Uses the aux[] uint32 manipulation for scale unpacking.
    /// </summary>
    private static void DequantQ3K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 110;
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        long numBlocks = elementCount / QK_K;

        Span<uint> aux = stackalloc uint[4];

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            int yOff = (int)(block * QK_K);

            float dAll = HalfToFloat(x[108], x[109]);

            // Unpack scales: copy 12 bytes at offset 96 into aux[0..2], then manipulate
            aux[0] = (uint)(x[96] | (x[97] << 8) | (x[98] << 16) | (x[99] << 24));
            aux[1] = (uint)(x[100] | (x[101] << 8) | (x[102] << 16) | (x[103] << 24));
            uint tmp = (uint)(x[104] | (x[105] << 8) | (x[106] << 16) | (x[107] << 24));

            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            // aux now contains 16 signed 6-bit scales as bytes (subtract 32 when used)
            int isIdx = 0;
            int qOff = 32; // qs at byte 32
            byte m = 1;    // hmask bit

            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    // Scale as signed int8 from aux bytes
                    int scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    float dl = dAll * (scByte - 32);
                    isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((x[qOff + l] >> shift) & 3) - ((x[l] & m) != 0 ? 0 : 4);
                        dst[yOff++] = dl * q;
                    }

                    scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    dl = dAll * (scByte - 32);
                    isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((x[qOff + l + 16] >> shift) & 3) - ((x[l + 16] & m) != 0 ? 0 : 4);
                        dst[yOff++] = dl * q;
                    }

                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }
        }
    }

    /// <summary>
    /// Q4_0 dequantization. Block size = 32, bytes per block = 18.
    /// Layout: [d:FP16][qs:16 × uint8] — two 4-bit values per byte, signed nibbles.
    /// Value = (nibble - 8) * d   (range -8..7)
    /// Reference: dequantize_row_q4_0 in ggml-quants.c
    /// </summary>
    private static void DequantQ4_0(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK = 32;
        const int bytesPerBlock = 18; // 2 (FP16 scale) + 16 (32 * 4-bit / 8)
        long numBlocks = elementCount / QK;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            var y = dst.Slice((int)(block * QK), QK);
            for (int j = 0; j < QK / 2; j++)
            {
                y[j]          = ((x[2 + j] & 0xF) - 8) * d;
                y[j + QK / 2] = ((x[2 + j] >>  4) - 8) * d;
            }
        }
    }

    /// <summary>
    /// Q5_0 dequantization. Block size = 32, bytes per block = 22.
    /// Layout: [d:FP16][qh:4 bytes (high bits)][qs:16 × uint8 (low 4 bits)]
    /// Value = ((low4 | highBit&lt;&lt;4) - 16) * d   (range -16..15)
    /// Reference: dequantize_row_q5_0 in ggml-quants.c
    /// </summary>
    private static void DequantQ5_0(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK = 32;
        const int bytesPerBlock = 22; // 2 (FP16) + 4 (qh) + 16 (qs)
        long numBlocks = elementCount / QK;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            uint qh = (uint)(x[2] | (x[3] << 8) | (x[4] << 16) | (x[5] << 24));
            var y = dst.Slice((int)(block * QK), QK);
            for (int j = 0; j < QK / 2; j++)
            {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int x0 = (x[6 + j] & 0xF) | xh0;
                int x1 = (x[6 + j] >>  4) | xh1;
                y[j]          = (x0 - 16) * d;
                y[j + QK / 2] = (x1 - 16) * d;
            }
        }
    }

    /// <summary>Q4_1: [d:FP16][m:FP16][16 packed unsigned nibbles].</summary>
    private static void DequantQ4_1(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 32;
        const int bytesPerBlock = 20;
        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            float m = HalfToFloat(x[2], x[3]);
            for (int i = 0; i < 16; i++)
            {
                y[i] = (x[4 + i] & 0x0F) * d + m;
                y[i + 16] = (x[4 + i] >> 4) * d + m;
            }
        }
    }

    /// <summary>Q5_1: [d:FP16][m:FP16][qh:4][16 packed low nibbles].</summary>
    private static void DequantQ5_1(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 32;
        const int bytesPerBlock = 24;
        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            float m = HalfToFloat(x[2], x[3]);
            uint qh = (uint)(x[4] | (x[5] << 8) | (x[6] << 16) | (x[7] << 24));
            for (int i = 0; i < 16; i++)
            {
                int low = (x[8 + i] & 0x0F) | ((int)((qh >> i) & 1) << 4);
                int high = (x[8 + i] >> 4) | ((int)((qh >> (i + 16)) & 1) << 4);
                y[i] = low * d + m;
                y[i + 16] = high * d + m;
            }
        }
    }

    /// <summary>
    /// Q8_0 dequantization. Block size = 32, bytes per block = 34.
    /// Layout: [d:FP16][qs:32 × int8]
    /// Reference: dequantize_row_q8_0 in ggml-quants.c
    /// </summary>
    private static void DequantQ8_0(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK = 32;
        const int bytesPerBlock = 34; // 2 (FP16 scale) + 32 (int8 values)
        long numBlocks = elementCount / QK;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            var y = dst.Slice((int)(block * QK), QK);
            for (int j = 0; j < QK; j++)
                y[j] = (sbyte)x[2 + j] * d;
        }
    }

    /// <summary>Q8_1: [d:FP16][quantized sum:FP16][32 signed bytes].</summary>
    private static void DequantQ8_1(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int elementsPerBlock = 32;
        const int bytesPerBlock = 36;
        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            for (int i = 0; i < elementsPerBlock; i++)
                y[i] = (sbyte)x[4 + i] * d;
        }
    }

    /// <summary>IQ4_NL: [d:FP16][16 non-linear-codebook nibbles].</summary>
    private static void DequantIq4Nl(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        var codebook = IqCodebooks.Iq4NlCodebook;
        const int elementsPerBlock = 32;
        const int bytesPerBlock = 18;
        for (long block = 0; block < elementCount / elementsPerBlock; block++)
        {
            ReadOnlySpan<byte> x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            Span<float> y = dst.Slice((int)(block * elementsPerBlock), elementsPerBlock);
            float d = HalfToFloat(x[0], x[1]);
            for (int i = 0; i < 16; i++)
            {
                y[i] = d * codebook[x[2 + i] & 0x0F];
                y[i + 16] = d * codebook[x[2 + i] >> 4];
            }
        }
    }

    /// <summary>FP16 (IEEE 754 half-precision) dequantization.</summary>
    private static void DequantF16(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        var halves = MemoryMarshal.Cast<byte, Half>(src);
        for (int i = 0; i < (int)elementCount; i++)
            dst[i] = (float)halves[i];
    }

    /// <summary>BFloat16 dequantization.</summary>
    private static void DequantBF16(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        for (int i = 0; i < (int)elementCount; i++)
        {
            ushort bits = (ushort)(src[i * 2] | (src[i * 2 + 1] << 8));
            dst[i] = BitConverter.Int32BitsToSingle(bits << 16);
        }
    }

    /// <summary>
    /// IQ2_S scalar dequantization decoder (256 elements / 82 bytes per block), matching ggml's
    /// dequantize_row_iq2_s: 8 groups of 32 elements, each sharing one nibble-split scale byte
    /// (same layout as IQ2_XS) and one qh byte contributing 2 extra high bits per grid index
    /// (index = qs-byte | ((qh &lt;&lt; (8-2*l)) &amp; 0x300), a 10-bit index into the 1024-entry
    /// <see cref="IqCodebooks.Iq2SGrid"/>). Unlike IQ2_XS/IQ2_XXS, the sign byte is used
    /// directly as an 8-bit mask (no <see cref="IqCodebooks.KSignsIq2Xs"/> indirection) --
    /// ggml stores the 32 grid-index bytes and 32 sign bytes back-to-back in the same field.
    /// </summary>
    private static void DequantIq2S(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 82;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq2SGrid;
        var kmask = IqCodebooks.KMaskIq2Xs;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            var qsLow = block.Slice(2, 32);
            var signs = block.Slice(34, 32);
            var qh = block.Slice(66, 8);
            var scales = block.Slice(74, 8);
            int y = b * kK;

            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                float db0 = d * (0.5f + (scales[ib32] & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                int baseOff = ib32 * 4;
                for (int l = 0; l < 4; l++)
                {
                    float dl = l < 2 ? db0 : db1;
                    int idx = qsLow[baseOff + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                    ulong gridVal = grid[idx];
                    byte sgn = signs[baseOff + l];
                    for (int j = 0; j < 8; j++)
                        dst[y + j] = dl * (byte)(gridVal >> (8 * j)) * ((sgn & kmask[j]) != 0 ? -1f : 1f);
                    y += 8;
                }
            }
        }
    }

    /// <summary>
    /// IQ2_XS scalar dequantization decoder (256 elements / 74 bytes per block), matching
    /// ggml's dequantize_row_iq2_xs: 8 groups of 32 elements, each group sharing one scale byte
    /// (low nibble scales the group's first two 8-element sub-groups, high nibble the last two)
    /// and four 16-bit qs words, each packing a 9-bit grid index (low 9 bits) plus a 7-bit
    /// sign-select field (top 7 bits) resolved through <see cref="IqCodebooks.KSignsIq2Xs"/>.
    /// Same real-table provenance as <see cref="DequantIq2Xxs"/>; see its remarks.
    /// </summary>
    private static void DequantIq2Xs(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 74;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq2XsGrid;
        var ksigns = IqCodebooks.KSignsIq2Xs;
        var kmask = IqCodebooks.KMaskIq2Xs;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            var qs = block.Slice(2, 64);
            var scales = block.Slice(66, 8);
            int y = b * kK;

            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                float db0 = d * (0.5f + (scales[ib32] & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    int qsOff = (4 * ib32 + l) * 2;
                    int qval = qs[qsOff] | (qs[qsOff + 1] << 8);
                    ulong gridVal = grid[qval & 511];
                    byte signs = ksigns[qval >> 9];
                    float dl = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                        dst[y + j] = dl * (byte)(gridVal >> (8 * j)) * ((signs & kmask[j]) != 0 ? -1f : 1f);
                    y += 8;
                }
            }
        }
    }

    /// <summary>
    /// IQ2_XXS scalar dequantization decoder (256 elements / 66 bytes per block), matching
    /// ggml's dequantize_row_iq2_xxs exactly: 8 groups of 32 elements, each group carrying a
    /// packed 32-bit scale/sign-select word (top 4 bits + 0.5 offset, quarter-scale) alongside
    /// four 8-byte grid lookups, each negated per-element by a 7-bit sign field decoded through
    /// <see cref="IqCodebooks.KSignsIq2Xs"/>. The previous version ignored the per-group scale
    /// and sign bits entirely, treating every byte as a flat codebook index under one global
    /// scale -- see docs/bugstofix.md's IqCodebooks.cs entry.
    /// </summary>
    private static void DequantIq2Xxs(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 66;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq2XxsGrid;
        var ksigns = IqCodebooks.KSignsIq2Xs;
        var kmask = IqCodebooks.KMaskIq2Xs;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            var qs = block.Slice(2, 64);
            int y = b * kK;

            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                int off = ib32 * 8;
                uint aux0 = (uint)(qs[off] | (qs[off + 1] << 8) | (qs[off + 2] << 16) | (qs[off + 3] << 24));
                uint aux1 = (uint)(qs[off + 4] | (qs[off + 5] << 8) | (qs[off + 6] << 16) | (qs[off + 7] << 24));
                float db = d * (0.5f + (aux1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    ulong gridVal = grid[(byte)(aux0 >> (8 * l))];
                    byte signs = ksigns[(int)((aux1 >> (7 * l)) & 127)];
                    for (int j = 0; j < 8; j++)
                        dst[y + j] = db * (byte)(gridVal >> (8 * j)) * ((signs & kmask[j]) != 0 ? -1f : 1f);
                    y += 8;
                }
            }
        }
    }

    /// <summary>
    /// IQ3_XXS scalar dequantization decoder (256 elements / 98 bytes per block), matching
    /// ggml's dequantize_row_iq3_xxs: 64 grid-index bytes (2 per 8-element half-group) followed
    /// by 8 packed scale/sign words (one per 32-element group), each grid lookup is a 4-byte
    /// vector negated per-element by a 7-bit sign field. See DequantIq2Xxs's remarks.
    /// </summary>
    private static void DequantIq3Xxs(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 98;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq3XxsGrid;
        var ksigns = IqCodebooks.KSignsIq2Xs;
        var kmask = IqCodebooks.KMaskIq2Xs;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            var qs = block.Slice(2, 64);
            var scalesAndSigns = block.Slice(2 + 64, 32);
            int y = b * kK;
            int qsOff = 0;

            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                int so = ib32 * 4;
                uint aux32 = (uint)(scalesAndSigns[so] | (scalesAndSigns[so + 1] << 8)
                    | (scalesAndSigns[so + 2] << 16) | (scalesAndSigns[so + 3] << 24));
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++)
                {
                    byte signs = ksigns[(int)((aux32 >> (7 * l)) & 127)];
                    uint grid1 = grid[qs[qsOff + 2 * l]];
                    uint grid2 = grid[qs[qsOff + 2 * l + 1]];
                    for (int j = 0; j < 4; j++)
                    {
                        dst[y + j] = db * (byte)(grid1 >> (8 * j)) * ((signs & kmask[j]) != 0 ? -1f : 1f);
                        dst[y + j + 4] = db * (byte)(grid2 >> (8 * j)) * ((signs & kmask[j + 4]) != 0 ? -1f : 1f);
                    }
                    y += 8;
                }
                qsOff += 8;
            }
        }
    }

    /// <summary>
    /// IQ3_S scalar dequantization decoder (256 elements / 110 bytes per block), matching
    /// ggml's dequantize_row_iq3_s: grid indices carry a 9th bit from the qh side-channel byte
    /// (one per 32-element sub-group pair), each 4-byte grid lookup negated per-element by an
    /// explicit sign byte (not the packed 7-bit field the XXS variants use), and per-32-element
    /// scales taken directly from a 4-bit nibble (linear, not the 0.5-offset quarter-scale the
    /// XXS variants use). See DequantIq2Xxs's remarks.
    /// </summary>
    private static void DequantIq3S(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 110;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq3SGrid;
        var kmask = IqCodebooks.KMaskIq2Xs;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            var qs = block.Slice(2, 64);
            var qh = block.Slice(2 + 64, 8);
            var signs = block.Slice(2 + 64 + 8, 32);
            var scales = block.Slice(2 + 64 + 8 + 32, 4);
            int y = b * kK;
            int qsOff = 0, signsOff = 0;

            for (int ib32 = 0; ib32 < 8; ib32 += 2)
            {
                float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xF));
                float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
                for (int half = 0; half < 2; half++)
                {
                    float db = half == 0 ? db1 : db2;
                    byte qhByte = qh[ib32 + half];
                    for (int l = 0; l < 4; l++)
                    {
                        uint grid1 = grid[qs[qsOff + 2 * l] | ((uint)(qhByte << (8 - 2 * l)) & 256)];
                        uint grid2 = grid[qs[qsOff + 2 * l + 1] | ((uint)(qhByte << (7 - 2 * l)) & 256)];
                        byte s = signs[signsOff + l];
                        for (int j = 0; j < 4; j++)
                        {
                            dst[y + j] = db * (byte)(grid1 >> (8 * j)) * ((s & kmask[j]) != 0 ? -1f : 1f);
                            dst[y + j + 4] = db * (byte)(grid2 >> (8 * j)) * ((s & kmask[j + 4]) != 0 ? -1f : 1f);
                        }
                        y += 8;
                    }
                    qsOff += 8;
                    signsOff += 4;
                }
            }
        }
    }

    /// <summary>
    /// IQ4_XS scalar dequantization decoder (256 elements / 136 bytes per block), matching
    /// ggml's dequantize_row_iq4_xs. Unlike the other three IQ formats, IQ4_XS is NOT a
    /// grid/codebook-vector format at all -- it reuses IQ4_NL's 16-entry non-linear scalar
    /// codebook per-nibble (<see cref="IqCodebooks.Iq4NlCodebook"/>), split into eight
    /// 32-element sub-groups each with its own 6-bit scale (4 bits from scales_l, 2 from
    /// scales_h). The previous version fabricated an unrelated 256-entry ±1 "grid" for this
    /// format; see docs/bugstofix.md's IqCodebooks.cs entry.
    /// </summary>
    private static void DequantIq4Xs(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 136;
        int numBlocks = (int)(elementCount / kK);
        var cb = IqCodebooks.Iq4NlCodebook;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            int scalesH = block[2] | (block[3] << 8);
            var scalesL = block.Slice(4, 4);
            var qs = block.Slice(8, 128);
            int y = b * kK;
            int qsOff = 0;

            for (int ib = 0; ib < 8; ib++)
            {
                int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++)
                {
                    dst[y + j] = dl * cb[qs[qsOff + j] & 0xF];
                    dst[y + j + 16] = dl * cb[qs[qsOff + j] >> 4];
                }
                y += 32;
                qsOff += 16;
            }
        }
    }

    /// <summary>
    /// IQ1_S scalar dequantization decoder (256 elements / 50 bytes per block), matching ggml's
    /// dequantize_row_iq1_s: 8 groups of 32 elements, each carrying a 16-bit qh word packing a
    /// 3-bit scale (bits 12-14), a global-sign bit (bit 15) applied as a ±<see
    /// cref="IqCodebooks.Iq1sDelta"/> offset added to every grid value before scaling (not a
    /// per-element sign flip the way the IQ2/IQ3 formats work), and 3 extra high bits per
    /// 4-element sub-group (bits 0-2, 3-5, 6-8, 9-11) that combine with a qs byte to form a
    /// grid index. Grid entries are already-signed int8 values (see <see
    /// cref="IqCodebooks.Iq1sGrid"/>'s remarks) -- no separate sign table needed.
    /// </summary>
    private static void DequantIq1S(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 50;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq1sGrid;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            float d = HalfToFloat(block[0], block[1]);
            var qs = block.Slice(2, 32);
            var qhBytes = block.Slice(34, 16);
            int y = b * kK;
            int qsOff = 0;

            for (int ib = 0; ib < 8; ib++)
            {
                int qh = qhBytes[ib * 2] | (qhBytes[ib * 2 + 1] << 8);
                float dl = d * (2 * ((qh >> 12) & 7) + 1);
                float delta = (qh & 0x8000) != 0 ? -IqCodebooks.Iq1sDelta : IqCodebooks.Iq1sDelta;
                for (int l = 0; l < 4; l++)
                {
                    int idx = qs[qsOff + l] | (((qh >> (3 * l)) & 7) << 8);
                    ulong gridVal = grid[idx];
                    for (int j = 0; j < 8; j++)
                        dst[y + j] = dl * ((sbyte)(byte)(gridVal >> (8 * j)) + delta);
                    y += 8;
                }
                qsOff += 4;
            }
        }
    }

    /// <summary>
    /// IQ1_M scalar dequantization decoder (256 elements / 56 bytes per block), matching ggml's
    /// dequantize_row_iq1_m. Shares IQ1_S's grid table and delta scheme, but has no dedicated
    /// scale field: the shared block scale is scavenged from the TOP 4 bits of each of the 4
    /// uint16 words the 8-byte `scales` field reinterprets to (ggml's `iq1m_scale_t` union
    /// trick), leaving the LOW 12 bits of each word to hold two 3-bit sub-scales apiece (one
    /// scales word covers 2 consecutive 32-element groups). Each 32-element group also has TWO
    /// grid+delta pairs (not one) -- l=0,1 share `dl1`, l=2,3 share `dl2` -- and each of the 4
    /// grid lookups needs both `qs` and `qh` bytes (2 qh bytes per group, 1 shared between the
    /// first/second and third/fourth lookups' high-bit contribution).
    /// </summary>
    private static void DequantIq1M(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int kK = 256;
        const int bytesPerBlock = 56;
        int numBlocks = (int)(elementCount / kK);
        var grid = IqCodebooks.Iq1sGrid;

        for (int b = 0; b < numBlocks; b++)
        {
            var block = src.Slice(b * bytesPerBlock, bytesPerBlock);
            var qs = block.Slice(0, 32);
            var qh = block.Slice(32, 16);
            var scalesBytes = block.Slice(48, 8);

            int sc0 = scalesBytes[0] | (scalesBytes[1] << 8);
            int sc1 = scalesBytes[2] | (scalesBytes[3] << 8);
            int sc2 = scalesBytes[4] | (scalesBytes[5] << 8);
            int sc3 = scalesBytes[6] | (scalesBytes[7] << 8);
            int scaleBits = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
            float d = HalfToFloat((byte)(scaleBits & 0xFF), (byte)((scaleBits >> 8) & 0xFF));

            int y = b * kK;
            int qsOff = 0, qhOff = 0;

            for (int ib = 0; ib < 8; ib++)
            {
                int scWord = (ib / 2) switch { 0 => sc0, 1 => sc1, 2 => sc2, _ => sc3 };
                int shift = 6 * (ib % 2);
                float dl1 = d * (2 * ((scWord >> (shift + 0)) & 7) + 1);
                float dl2 = d * (2 * ((scWord >> (shift + 3)) & 7) + 1);

                byte qh0 = qh[qhOff];
                byte qh1 = qh[qhOff + 1];
                int idx0 = qs[qsOff] | ((qh0 << 8) & 0x700);
                int idx1 = qs[qsOff + 1] | ((qh0 << 4) & 0x700);
                int idx2 = qs[qsOff + 2] | ((qh1 << 8) & 0x700);
                int idx3 = qs[qsOff + 3] | ((qh1 << 4) & 0x700);
                float delta0 = (qh0 & 0x08) != 0 ? -IqCodebooks.Iq1sDelta : IqCodebooks.Iq1sDelta;
                float delta1 = (qh0 & 0x80) != 0 ? -IqCodebooks.Iq1sDelta : IqCodebooks.Iq1sDelta;
                float delta2 = (qh1 & 0x08) != 0 ? -IqCodebooks.Iq1sDelta : IqCodebooks.Iq1sDelta;
                float delta3 = (qh1 & 0x80) != 0 ? -IqCodebooks.Iq1sDelta : IqCodebooks.Iq1sDelta;

                ulong g0 = grid[idx0], g1 = grid[idx1], g2 = grid[idx2], g3 = grid[idx3];
                for (int j = 0; j < 8; j++) dst[y + j] = dl1 * ((sbyte)(byte)(g0 >> (8 * j)) + delta0);
                y += 8;
                for (int j = 0; j < 8; j++) dst[y + j] = dl1 * ((sbyte)(byte)(g1 >> (8 * j)) + delta1);
                y += 8;
                for (int j = 0; j < 8; j++) dst[y + j] = dl2 * ((sbyte)(byte)(g2 >> (8 * j)) + delta2);
                y += 8;
                for (int j = 0; j < 8; j++) dst[y + j] = dl2 * ((sbyte)(byte)(g3 >> (8 * j)) + delta3);
                y += 8;

                qsOff += 4;
                qhOff += 2;
            }
        }
    }

    /// <summary>Convert two bytes (little-endian) to FP16, then to float.</summary>
    private static float HalfToFloat(byte lo, byte hi)
    {
        ushort bits = (ushort)(lo | (hi << 8));
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }
}
