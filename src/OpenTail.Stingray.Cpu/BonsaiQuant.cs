// Ported from TensorSharp (https://github.com/zhongkaifu/TensorSharp), TensorSharp.GGML.Native/bonsai_quant.cpp,
// Copyright (c) Zhongkai Fu, BSD 3-Clause License (see THIRD_PARTY_NOTICES.md).

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Decoding for the PrismML Bonsai2 publisher encodings <see cref="DType.PQ2_0"/> (GGUF type 142) and
/// <see cref="DType.PTQ1_0"/> (143): ternary weights {-1, 0, +1} (PQ2_0 also allows +2) times one fp16
/// scale per 128-element block.
/// <list type="bullet">
/// <item>PQ2_0 (34 bytes/block): fp16 scale, then 32 bytes of 2-bit codes, lane j at bits 2*(j%4) of
/// byte j/4.</item>
/// <item>PTQ1_0 (28 bytes/block): 26 bytes of base-3 digits, then the fp16 scale. 16 lanes x 5 trits, then
/// 8 lanes x 5, then 2 lanes x 4. Each byte stores ceil(base3 * 256 / 243), so multiplying by successive
/// powers of 3 modulo 256 exposes the most significant trit first.</item>
/// </list>
/// Codes are unsigned (0 = -1, 1 = 0, 2 = +1, 3 = +2). <see cref="TranscodeToQ2_0"/> rewrites a tensor
/// losslessly as upstream ggml <see cref="DType.Q2_0"/> (64-element blocks, the same code meaning), so
/// the existing Q2_0 dequantizer and matvec kernel run these weights unchanged.
/// </summary>
public static unsafe class BonsaiQuant
{
    public const int BlockElements = 128;
    public const int Pq2Bytes = 34, Ptq1Bytes = 28, Q2Bytes = 18;

    public static bool IsBonsaiType(DType dtype) => dtype is DType.PQ2_0 or DType.PTQ1_0;

    /// <summary>Unsigned ternary codes of one 128-element block, and its fp16 scale bits.</summary>
    public static void UnpackBlock(DType dtype, byte* src, byte* codes, out ushort scaleBits)
    {
        if (dtype == DType.PQ2_0)
        {
            scaleBits = (ushort)(src[0] | (src[1] << 8));
            for (int j = 0; j < BlockElements; j++)
                codes[j] = (byte)((src[2 + j / 4] >> (2 * (j % 4))) & 3);
            return;
        }
        if (dtype != DType.PTQ1_0) throw new ArgumentException($"Not a Bonsai2 type: {dtype}", nameof(dtype));

        scaleBits = (ushort)(src[26] | (src[27] << 8));
        ReadOnlySpan<uint> powers = [1, 3, 9, 27, 81];
        int o = 0;
        for (int stage = 0; stage < 2; stage++)
        {
            int lanes = stage == 0 ? 16 : 8;
            int offset = stage == 0 ? 0 : 16;
            foreach (uint power in powers)
                for (int lane = 0; lane < lanes; lane++)
                    codes[o++] = (byte)(((uint)(byte)(src[offset + lane] * power) * 3u) >> 8);
        }
        for (int trit = 0; trit < 4; trit++)
            for (int lane = 0; lane < 2; lane++)
                codes[o++] = (byte)(((uint)(byte)(src[24 + lane] * powers[trit]) * 3u) >> 8);
    }

    /// <summary>Dequantize <paramref name="elementCount"/> values (a multiple of 128) to F32.</summary>
    public static void Dequantize(DType dtype, byte* src, float* dst, long elementCount)
    {
        if (elementCount % BlockElements != 0)
            throw new ArgumentException($"{dtype} needs a multiple of {BlockElements} elements, got {elementCount}.");
        int stride = dtype == DType.PQ2_0 ? Pq2Bytes : Ptq1Bytes;
        byte* codes = stackalloc byte[BlockElements];
        for (long b = 0; b < elementCount / BlockElements; b++)
        {
            UnpackBlock(dtype, src + b * stride, codes, out ushort sb);
            float d = (float)BitConverter.UInt16BitsToHalf(sb);
            float* y = dst + b * BlockElements;
            for (int j = 0; j < BlockElements; j++) y[j] = (codes[j] - 1) * d;
        }
    }

    /// <summary>Bytes of the <see cref="DType.Q2_0"/> transcode of <paramref name="elementCount"/> values.</summary>
    public static long Q2_0Bytes(long elementCount) => elementCount / 64 * Q2Bytes;

    /// <summary>
    /// Losslessly rewrite <paramref name="elementCount"/> values as ggml Q2_0: each 128-element block becomes
    /// two 64-element Q2_0 blocks with the same scale and codes. No floating-point requantization.
    /// <paramref name="dst"/> must not overlap <paramref name="src"/>.
    /// </summary>
    public static void TranscodeToQ2_0(DType dtype, byte* src, long elementCount, byte* dst)
    {
        if (elementCount % BlockElements != 0)
            throw new ArgumentException($"{dtype} needs a multiple of {BlockElements} elements, got {elementCount}.");
        int stride = dtype == DType.PQ2_0 ? Pq2Bytes : Ptq1Bytes;
        byte* codes = stackalloc byte[BlockElements];
        for (long b = 0; b < elementCount / BlockElements; b++)
        {
            byte* s = src + b * stride;
            byte* t = dst + b * 2 * Q2Bytes;
            if (dtype == DType.PQ2_0)
            {
                // Already Q2's sequential 2-bit layout: duplicate the scale for the two halves.
                t[0] = s[0]; t[1] = s[1];
                Buffer.MemoryCopy(s + 2, t + 2, 16, 16);
                t[Q2Bytes] = s[0]; t[Q2Bytes + 1] = s[1];
                Buffer.MemoryCopy(s + 18, t + Q2Bytes + 2, 16, 16);
                continue;
            }
            UnpackBlock(dtype, s, codes, out ushort sb);
            for (int half = 0; half < 2; half++)
            {
                byte* p = t + half * Q2Bytes;
                p[0] = (byte)sb; p[1] = (byte)(sb >> 8);
                for (int j = 0; j < 16; j++)
                {
                    byte* c = codes + half * 64 + j * 4;
                    p[2 + j] = (byte)(c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6));
                }
            }
        }
    }
}
