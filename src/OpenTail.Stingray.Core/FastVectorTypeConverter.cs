using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Core;

/// <summary>
/// High-performance SIMD type converters for float conversions:
/// F16, BF16, FP8 (E4M3FN / E5M2) to Float32, Float32 to BF16/F16 narrowing,
/// and Q8_0 block quantization / dequantization.
/// </summary>
public static class FastVectorTypeConverter
{
    // Pre-computed 256-element lookup tables for 8-bit float formats (O(1) conversion per byte).
    private static readonly float[] s_fp8E4M3Table = BuildFp8E4M3Table();
    private static readonly float[] s_fp8E5M2Table = BuildFp8E5M2Table();

    // ── FP8 E4M3FN → F32 ──────────────────────────────────────────────────

    /// <summary>
    /// Converts FP8 (E4M3FN) bytes to Float32.
    /// </summary>
    public static void ConvertFp8E4M3ToF32(ReadOnlySpan<byte> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than source length ({src.Length}).");

        ref byte sRef = ref MemoryMarshal.GetReference(src);
        ref float dRef = ref MemoryMarshal.GetReference(dst);
        ref float tableRef = ref MemoryMarshal.GetArrayDataReference(s_fp8E4M3Table);

        int i = 0;
        int count = src.Length;

        // Unroll 8x for cache-friendly table lookup
        int count8 = count & ~7;
        for (; i < count8; i += 8)
        {
            Unsafe.Add(ref dRef, i + 0) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 0));
            Unsafe.Add(ref dRef, i + 1) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 1));
            Unsafe.Add(ref dRef, i + 2) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 2));
            Unsafe.Add(ref dRef, i + 3) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 3));
            Unsafe.Add(ref dRef, i + 4) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 4));
            Unsafe.Add(ref dRef, i + 5) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 5));
            Unsafe.Add(ref dRef, i + 6) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 6));
            Unsafe.Add(ref dRef, i + 7) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 7));
        }

        for (; i < count; i++)
        {
            Unsafe.Add(ref dRef, i) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i));
        }
    }

    // ── FP8 E5M2 → F32 ────────────────────────────────────────────────────

    /// <summary>
    /// Converts FP8 (E5M2) bytes to Float32.
    /// </summary>
    public static void ConvertFp8E5M2ToF32(ReadOnlySpan<byte> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than source length ({src.Length}).");

        ref byte sRef = ref MemoryMarshal.GetReference(src);
        ref float dRef = ref MemoryMarshal.GetReference(dst);
        ref float tableRef = ref MemoryMarshal.GetArrayDataReference(s_fp8E5M2Table);

        int i = 0;
        int count = src.Length;

        int count8 = count & ~7;
        for (; i < count8; i += 8)
        {
            Unsafe.Add(ref dRef, i + 0) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 0));
            Unsafe.Add(ref dRef, i + 1) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 1));
            Unsafe.Add(ref dRef, i + 2) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 2));
            Unsafe.Add(ref dRef, i + 3) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 3));
            Unsafe.Add(ref dRef, i + 4) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 4));
            Unsafe.Add(ref dRef, i + 5) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 5));
            Unsafe.Add(ref dRef, i + 6) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 6));
            Unsafe.Add(ref dRef, i + 7) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i + 7));
        }

        for (; i < count; i++)
        {
            Unsafe.Add(ref dRef, i) = Unsafe.Add(ref tableRef, Unsafe.Add(ref sRef, i));
        }
    }

    // ── BF16 → F32 ────────────────────────────────────────────────────────

    /// <summary>
    /// Converts BFloat16 raw little-endian bytes to Float32.
    /// Each element requires 2 bytes in src.
    /// </summary>
    /// <remarks>
    /// <para><b>QUESTIONABLE — do not wire into a decode path without evidence.</b> Widening BF16 to
    /// a scratch buffer is the shape this engine deliberately moved away from. Decode is
    /// memory-bandwidth-bound, so a standalone pass that reads N bytes and writes 2N to scratch
    /// spends its budget on traffic; the win came from fusing the widen into the consumer, which is
    /// what <c>SimdKernels.DotF32Bf16</c> and the BF16 branches in <c>ForwardPass</c> already do.
    /// The KV-narrowing work established that decode and prefill want opposite widening designs —
    /// fused-in-dot versus widen-to-scratch — and this method only serves the second.</para>
    ///
    /// <para>Legitimate uses are bulk, one-shot, off the hot path: widening a whole tensor at load
    /// time, or a test/reference harness. If a caller in the decode loop wants this, the question to
    /// answer first is why it is not fusing instead.</para>
    /// </remarks>
    public static void ConvertBf16ToF32(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int elementCount = src.Length / 2;
        if (dst.Length < elementCount)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than required element count ({elementCount}).");

        ReadOnlySpan<ushort> bf16Span = MemoryMarshal.Cast<byte, ushort>(src);
        ConvertBf16ToF32Internal(bf16Span, dst.Slice(0, elementCount));
    }

    /// <summary>
    /// Converts BFloat16 ushort values to Float32.
    /// </summary>
    /// <remarks>QUESTIONABLE for the same reason as the byte overload above — prefer fusing the
    /// widen into the consumer over materialising an F32 scratch buffer on a hot path.</remarks>
    public static void ConvertBf16ToF32(ReadOnlySpan<ushort> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than source length ({src.Length}).");

        ConvertBf16ToF32Internal(src, dst.Slice(0, src.Length));
    }

    private static void ConvertBf16ToF32Internal(ReadOnlySpan<ushort> src, Span<float> dst)
    {
        int count = src.Length;
        int i = 0;

        // Vector256 path (16 ushorts -> 16 floats)
        if (Vector256.IsHardwareAccelerated && count >= 16)
        {
            int vec256Count = count & ~15;
            for (; i < vec256Count; i += 16)
            {
                Vector256<ushort> v16 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(src), (nuint)i);
                Vector256<uint> v32Lo = Vector256.WidenLower(v16);
                Vector256<uint> v32Hi = Vector256.WidenUpper(v16);

                Vector256<uint> fBitsLo = Vector256.ShiftLeft(v32Lo, 16);
                Vector256<uint> fBitsHi = Vector256.ShiftLeft(v32Hi, 16);

                fBitsLo.AsSingle().StoreUnsafe(ref MemoryMarshal.GetReference(dst), (nuint)i);
                fBitsHi.AsSingle().StoreUnsafe(ref MemoryMarshal.GetReference(dst), (nuint)(i + 8));
            }
        }
        else if (Avx2.IsSupported && count >= 8)
        {
            int vec256Count = count & ~7;
            for (; i < vec256Count; i += 8)
            {
                Vector128<ushort> v16 = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(src), (nuint)i);
                Vector256<int> v32 = Avx2.ConvertToVector256Int32(v16);
                Vector256<int> shifted = Avx2.ShiftLeftLogical(v32, 16);
                shifted.AsSingle().StoreUnsafe(ref MemoryMarshal.GetReference(dst), (nuint)i);
            }
        }

        // Scalar fallback loop
        ref ushort sRef = ref MemoryMarshal.GetReference(src);
        ref float dRef = ref MemoryMarshal.GetReference(dst);
        for (; i < count; i++)
        {
            uint bits = (uint)Unsafe.Add(ref sRef, i) << 16;
            Unsafe.Add(ref dRef, i) = BitConverter.UInt32BitsToSingle(bits);
        }
    }

    // ── F16 → F32 ─────────────────────────────────────────────────────────

    /// <summary>
    /// Converts Half (FP16) raw little-endian bytes to Float32.
    /// Each element requires 2 bytes in src.
    /// </summary>
    /// <remarks>
    /// <para><b>QUESTIONABLE — this is not a vector path.</b> Despite living in a class called
    /// "FastVector", the implementation is an 8x-unrolled scalar <c>(float)Half</c> cast: the JIT
    /// lowers each one to a single <c>vcvtph2ps</c>, and the unrolling adds nothing a plain loop
    /// would not get. It earns its place here only as an API-shape sibling to the other converters,
    /// not as an optimisation, and callers should not choose it expecting one.</para>
    ///
    /// <para>A genuine win would need <c>vcvtph2ps</c> on a whole vector at a time, which .NET does
    /// not expose (there is no F16C intrinsic surface) — the same limitation that shaped the BF16
    /// work. Until that changes, this carries the cost of a public API for no measured benefit.</para>
    /// </remarks>
    public static void ConvertF16ToF32(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int elementCount = src.Length / 2;
        if (dst.Length < elementCount)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than required element count ({elementCount}).");

        ReadOnlySpan<Half> f16Span = MemoryMarshal.Cast<byte, Half>(src);
        ConvertF16ToF32Internal(f16Span, dst.Slice(0, elementCount));
    }

    /// <summary>
    /// Converts Half (FP16) values to Float32.
    /// </summary>
    /// <remarks>QUESTIONABLE for the same reason as the byte overload above — an unrolled scalar
    /// cast, not a vector path.</remarks>
    public static void ConvertF16ToF32(ReadOnlySpan<Half> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than source length ({src.Length}).");

        ConvertF16ToF32Internal(src, dst.Slice(0, src.Length));
    }

    private static void ConvertF16ToF32Internal(ReadOnlySpan<Half> src, Span<float> dst)
    {
        int count = src.Length;
        int i = 0;

        int count8 = count & ~7;
        ref Half sRef = ref MemoryMarshal.GetReference(src);
        ref float dRef = ref MemoryMarshal.GetReference(dst);

        for (; i < count8; i += 8)
        {
            Unsafe.Add(ref dRef, i + 0) = (float)Unsafe.Add(ref sRef, i + 0);
            Unsafe.Add(ref dRef, i + 1) = (float)Unsafe.Add(ref sRef, i + 1);
            Unsafe.Add(ref dRef, i + 2) = (float)Unsafe.Add(ref sRef, i + 2);
            Unsafe.Add(ref dRef, i + 3) = (float)Unsafe.Add(ref sRef, i + 3);
            Unsafe.Add(ref dRef, i + 4) = (float)Unsafe.Add(ref sRef, i + 4);
            Unsafe.Add(ref dRef, i + 5) = (float)Unsafe.Add(ref sRef, i + 5);
            Unsafe.Add(ref dRef, i + 6) = (float)Unsafe.Add(ref sRef, i + 6);
            Unsafe.Add(ref dRef, i + 7) = (float)Unsafe.Add(ref sRef, i + 7);
        }

        for (; i < count; i++)
        {
            Unsafe.Add(ref dRef, i) = (float)Unsafe.Add(ref sRef, i);
        }
    }

    // ── F32 → F16 ─────────────────────────────────────────────────────────

    /// <summary>
    /// Converts Float32 to Half (FP16) raw little-endian bytes.
    /// Destination byte span must be at least 2 * src.Length bytes.
    /// </summary>
    public static void ConvertF32ToF16(ReadOnlySpan<float> src, Span<byte> dst)
    {
        if (dst.Length < src.Length * 2)
            throw new ArgumentException($"Destination byte span length ({dst.Length}) is smaller than required byte size ({src.Length * 2}).");

        Span<Half> f16Dst = MemoryMarshal.Cast<byte, Half>(dst);
        ConvertF32ToF16Internal(src, f16Dst.Slice(0, src.Length));
    }

    /// <summary>
    /// Converts Float32 to Half (FP16) values.
    /// </summary>
    public static void ConvertF32ToF16(ReadOnlySpan<float> src, Span<Half> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than source length ({src.Length}).");

        ConvertF32ToF16Internal(src, dst.Slice(0, src.Length));
    }

    private static void ConvertF32ToF16Internal(ReadOnlySpan<float> src, Span<Half> dst)
    {
        int count = src.Length;
        int i = 0;
        int count8 = count & ~7;
        ref float sRef = ref MemoryMarshal.GetReference(src);
        ref Half dRef = ref MemoryMarshal.GetReference(dst);

        for (; i < count8; i += 8)
        {
            Unsafe.Add(ref dRef, i + 0) = (Half)Unsafe.Add(ref sRef, i + 0);
            Unsafe.Add(ref dRef, i + 1) = (Half)Unsafe.Add(ref sRef, i + 1);
            Unsafe.Add(ref dRef, i + 2) = (Half)Unsafe.Add(ref sRef, i + 2);
            Unsafe.Add(ref dRef, i + 3) = (Half)Unsafe.Add(ref sRef, i + 3);
            Unsafe.Add(ref dRef, i + 4) = (Half)Unsafe.Add(ref sRef, i + 4);
            Unsafe.Add(ref dRef, i + 5) = (Half)Unsafe.Add(ref sRef, i + 5);
            Unsafe.Add(ref dRef, i + 6) = (Half)Unsafe.Add(ref sRef, i + 6);
            Unsafe.Add(ref dRef, i + 7) = (Half)Unsafe.Add(ref sRef, i + 7);
        }

        for (; i < count; i++)
        {
            Unsafe.Add(ref dRef, i) = (Half)Unsafe.Add(ref sRef, i);
        }
    }

    // ── F32 → BF16 ────────────────────────────────────────────────────────

    /// <summary>
    /// Converts Float32 to BFloat16 raw little-endian bytes with round-to-nearest-even.
    /// Destination span must be at least 2 * src.Length bytes.
    /// </summary>
    /// <remarks>
    /// <para>Sound and genuinely vectorized, but <b>currently unwired, and not by oversight.</b> The
    /// only F32→BF16 narrowing in the engine is <c>PagedKvCache.ToBf16Bits</c>, whose NaN handling
    /// differs on purpose: it preserves the sign and payload and forces the quiet bit
    /// (<c>(u &gt;&gt; 16) | 0x0040</c>) so the KV store and its quality scaffold agree bit for bit,
    /// whereas this method canonicalizes every NaN to <c>0x7FC0</c>. Swapping one for the other
    /// would silently change what the KV cache stores.</para>
    ///
    /// <para>It is also the wrong granularity there: the cache narrows element-by-element as it
    /// appends, not in spans. So this waits for a genuine bulk narrowing caller — writing a BF16
    /// tensor to disk, or an export path. Reconciling the NaN rule is a prerequisite, not a detail.</para>
    /// </remarks>
    public static void ConvertF32ToBf16(ReadOnlySpan<float> src, Span<byte> dst)
    {
        if (dst.Length < src.Length * 2)
            throw new ArgumentException($"Destination byte span length ({dst.Length}) is smaller than required byte size ({src.Length * 2}).");

        Span<ushort> bf16Dst = MemoryMarshal.Cast<byte, ushort>(dst);
        ConvertF32ToBf16Internal(src, bf16Dst.Slice(0, src.Length));
    }

    /// <summary>
    /// Converts Float32 to BFloat16 ushort values with round-to-nearest-even.
    /// </summary>
    public static void ConvertF32ToBf16(ReadOnlySpan<float> src, Span<ushort> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than source length ({src.Length}).");

        ConvertF32ToBf16Internal(src, dst.Slice(0, src.Length));
    }

    private static void ConvertF32ToBf16Internal(ReadOnlySpan<float> src, Span<ushort> dst)
    {
        int count = src.Length;
        int i = 0;

        // AVX2 path: 16 floats -> 16 bf16 per iteration, packed and stored as whole vectors.
        //
        // NaN is canonicalized to 0x7FC0 to match the scalar tail below — a raw shift would keep the
        // sign bit and emit 0xFFC0 for -NaN, so the two paths would disagree on an input class no
        // round-trip test can see. The select is done IN the vector: an earlier version canonicalized
        // by extracting each lane with GetElement(j) inside a scalar loop, which paid for the vector
        // work and then threw the vectorization away.
        if (Avx2.IsSupported && count >= 16)
        {
            int vec256Count = count & ~15;
            Vector256<uint> biasBase = Vector256.Create(0x7FFFu);
            Vector256<uint> one = Vector256.Create(1u);
            Vector256<uint> quietNan = Vector256.Create(0x7FC0u);

            for (; i < vec256Count; i += 16)
            {
                Vector256<float> loF = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(src), (nuint)i);
                Vector256<float> hiF = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(src), (nuint)(i + 8));
                Vector256<uint> loBits = loF.AsUInt32();
                Vector256<uint> hiBits = hiF.AsUInt32();

                // RNE: bits += 0x7FFF + ((bits >> 16) & 1), then take the high half.
                Vector256<uint> loRounded = Avx2.ShiftRightLogical(
                    Avx2.Add(loBits, Avx2.Add(biasBase, Avx2.And(Avx2.ShiftRightLogical(loBits, 16), one))), 16);
                Vector256<uint> hiRounded = Avx2.ShiftRightLogical(
                    Avx2.Add(hiBits, Avx2.Add(biasBase, Avx2.And(Avx2.ShiftRightLogical(hiBits, 16), one))), 16);

                // x == x is false only for NaN, so the mask is all-ones on the lanes to keep.
                loRounded = Vector256.ConditionalSelect(Vector256.Equals(loF, loF).AsUInt32(), loRounded, quietNan);
                hiRounded = Vector256.ConditionalSelect(Vector256.Equals(hiF, hiF).AsUInt32(), hiRounded, quietNan);

                // PackUnsignedSaturate works within each 128-bit lane, so it interleaves the halves;
                // the permute restores memory order. Values are <= 0xFFFF, so saturation never bites.
                Vector256<ushort> packed = Avx2.PackUnsignedSaturate(loRounded.AsInt32(), hiRounded.AsInt32());
                packed = Avx2.Permute4x64(packed.AsUInt64(), 0b11_01_10_00).AsUInt16();

                packed.StoreUnsafe(ref MemoryMarshal.GetReference(dst), (nuint)i);
            }
        }

        // Scalar fallback loop with round-to-nearest-even logic
        ref float sRef = ref MemoryMarshal.GetReference(src);
        ref ushort dRef = ref MemoryMarshal.GetReference(dst);
        for (; i < count; i++)
        {
            float f = Unsafe.Add(ref sRef, i);
            if (float.IsNaN(f))
            {
                Unsafe.Add(ref dRef, i) = 0x7FC0;
                continue;
            }

            uint bits = BitConverter.SingleToUInt32Bits(f);
            uint lsb = (bits >> 16) & 1u;
            uint bias = 0x7FFFu + lsb;
            uint rounded = bits + bias;
            Unsafe.Add(ref dRef, i) = (ushort)(rounded >> 16);
        }
    }

    // ── Q8_0 Quantization & Dequantization ─────────────────────────────────

    /// <summary>
    /// Quantizes Float32 values into 34-byte Q8_0 blocks (32 elements per block).
    /// Input length must be a multiple of 32.
    /// Destination length must be at least (src.Length / 32) * 34 bytes.
    /// </summary>
    /// <remarks>
    /// <para><b>QUESTIONABLE — a Q8_0 quantizer was deliberately deleted from this codebase on
    /// 2026-08-05, and this reintroduces one.</b> See <c>docs/safetensors-support-plan.md</c> items
    /// R1, R5 and R10: the previous attempt selected the wrong tensors, was unreachable from any
    /// product code path, and had its numerical accuracy verified against nothing. The resolution
    /// was not "fix it" but "remove it", on the grounds that SafeTensors stays high precision and
    /// GGUF is the quantized deployment route.</para>
    ///
    /// <para>That reasoning has not changed, so before wiring this anywhere: (a) decide what product
    /// need it serves that GGUF does not already serve, and (b) prove the blocks it emits dequantize
    /// to the same values a GGUF Q8_0 build produces. A quantizer nothing calls is dead weight; a
    /// quantizer that runs and is subtly wrong is worse than either.</para>
    ///
    /// <para>The arithmetic itself follows llama.cpp: the reciprocal is taken from the unrounded
    /// scale while the stored scale is rounded to F16, so quantize and dequantize disagree very
    /// slightly — matching the reference rather than being more accurate than it.</para>
    /// </remarks>
    public static void ConvertF32ToQ8_0(ReadOnlySpan<float> src, Span<byte> dst)
    {
        if (src.Length % 32 != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of 32.");

        int numBlocks = src.Length / 32;
        int requiredBytes = numBlocks * 34;
        if (dst.Length < requiredBytes)
            throw new ArgumentException($"Destination byte span length ({dst.Length}) is smaller than required ({requiredBytes}).");

        ref float sRef = ref MemoryMarshal.GetReference(src);
        ref byte dRef = ref MemoryMarshal.GetReference(dst);

        for (int b = 0; b < numBlocks; b++)
        {
            int srcOffset = b * 32;
            int dstOffset = b * 34;

            // 1. Find max absolute value in the 32-element block
            float maxAbs = 0f;
            if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> v0 = Vector256.Abs(Vector256.LoadUnsafe(ref sRef, (nuint)(srcOffset + 0)));
                Vector256<float> v1 = Vector256.Abs(Vector256.LoadUnsafe(ref sRef, (nuint)(srcOffset + 8)));
                Vector256<float> v2 = Vector256.Abs(Vector256.LoadUnsafe(ref sRef, (nuint)(srcOffset + 16)));
                Vector256<float> v3 = Vector256.Abs(Vector256.LoadUnsafe(ref sRef, (nuint)(srcOffset + 24)));

                Vector256<float> m0 = Vector256.Max(v0, v1);
                Vector256<float> m1 = Vector256.Max(v2, v3);
                Vector256<float> maxVec = Vector256.Max(m0, m1);

                for (int j = 0; j < 8; j++)
                {
                    float a = maxVec.GetElement(j);
                    if (a > maxAbs) maxAbs = a;
                }
            }
            else
            {
                for (int j = 0; j < 32; j++)
                {
                    float a = MathF.Abs(Unsafe.Add(ref sRef, srcOffset + j));
                    if (a > maxAbs) maxAbs = a;
                }
            }

            // 2. Compute scale d and reciprocal
            float d = maxAbs / 127.0f;
            float invScale = maxAbs > 0f ? 127.0f / maxAbs : 0f;

            // 3. Store 2-byte FP16 scale
            Half hScale = (Half)d;
            ushort scaleBits = BitConverter.HalfToUInt16Bits(hScale);
            Unsafe.As<byte, ushort>(ref Unsafe.Add(ref dRef, dstOffset)) = scaleBits;

            // 4. Quantize 32 elements into 32 sbytes
            for (int j = 0; j < 32; j++)
            {
                float val = Unsafe.Add(ref sRef, srcOffset + j);
                int q = (int)MathF.Round(val * invScale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                Unsafe.Add(ref dRef, dstOffset + 2 + j) = (byte)(sbyte)q;
            }
        }
    }

    /// <summary>
    /// Dequantizes 34-byte Q8_0 blocks back into Float32 values (32 elements per block).
    /// Input length must be a multiple of 34 bytes.
    /// Destination length must be at least (src.Length / 34) * 32 float elements.
    /// </summary>
    /// <remarks>QUESTIONABLE for the same reasons as <see cref="ConvertF32ToQ8_0"/> — and note that
    /// round-tripping through that method proves nothing, since a symmetric error cancels. The
    /// dequantized values must be checked against a GGUF Q8_0 build, not against this file.</remarks>
    public static void ConvertQ8_0ToF32(ReadOnlySpan<byte> src, Span<float> dst)
    {
        if (src.Length % 34 != 0)
            throw new ArgumentException($"Source byte length ({src.Length}) must be a multiple of 34.");

        int numBlocks = src.Length / 34;
        int requiredElements = numBlocks * 32;
        if (dst.Length < requiredElements)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than required element count ({requiredElements}).");

        ref byte sRef = ref MemoryMarshal.GetReference(src);
        ref float dRef = ref MemoryMarshal.GetReference(dst);

        for (int b = 0; b < numBlocks; b++)
        {
            int srcOffset = b * 34;
            int dstOffset = b * 32;

            // Read FP16 scale d
            ushort scaleBits = Unsafe.As<byte, ushort>(ref Unsafe.Add(ref sRef, srcOffset));
            float d = (float)BitConverter.UInt16BitsToHalf(scaleBits);

            ReadOnlySpan<sbyte> qSpan = MemoryMarshal.Cast<byte, sbyte>(src.Slice(srcOffset + 2, 32));

            // SIMD dequantization (4x 8-element vectors)
            if (Avx2.IsSupported)
            {
                Vector256<float> dVec = Vector256.Create(d);
                for (int j = 0; j < 32; j += 8)
                {
                    Vector128<sbyte> q128 = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(qSpan), (nuint)j);
                    Vector256<int> q32 = Avx2.ConvertToVector256Int32(q128);
                    Vector256<float> f32 = Avx.ConvertToVector256Single(q32);
                    Vector256<float> scaled = Vector256.Multiply(f32, dVec);
                    scaled.StoreUnsafe(ref dRef, (nuint)(dstOffset + j));
                }
            }
            else
            {
                for (int j = 0; j < 32; j++)
                {
                    Unsafe.Add(ref dRef, dstOffset + j) = qSpan[j] * d;
                }
            }
        }
    }

    // ── FP8 Table Builders ────────────────────────────────────────────────

    private static float[] BuildFp8E4M3Table()
    {
        float[] table = new float[256];
        for (int b = 0; b < 256; b++)
        {
            int sign = (b >> 7) & 1;
            int exp = (b >> 3) & 0xF;
            int mant = b & 0x7;

            float val;
            if (exp == 0 && mant == 0)
            {
                val = 0.0f;
            }
            else if (exp == 0)
            {
                // Subnormal: 2^-6 * (mant / 8)
                val = MathF.Pow(2.0f, -6.0f) * (mant / 8.0f);
            }
            else if (exp == 15 && mant == 7)
            {
                // E4M3FN NaN representation (0x7F / 0xFF)
                val = float.NaN;
            }
            else
            {
                // Normal: 2^(exp - 7) * (1 + mant / 8)
                val = MathF.Pow(2.0f, exp - 7.0f) * (1.0f + mant / 8.0f);
            }

            table[b] = sign == 0 ? val : -val;
        }
        return table;
    }

    private static float[] BuildFp8E5M2Table()
    {
        float[] table = new float[256];
        for (int b = 0; b < 256; b++)
        {
            int sign = (b >> 7) & 1;
            int exp = (b >> 2) & 0x1F;
            int mant = b & 0x3;

            float val;
            if (exp == 0 && mant == 0)
            {
                val = 0.0f;
            }
            else if (exp == 0)
            {
                // Subnormal: 2^-14 * (mant / 4)
                val = MathF.Pow(2.0f, -14.0f) * (mant / 4.0f);
            }
            else if (exp == 31 && mant == 0)
            {
                // Infinity
                val = float.PositiveInfinity;
            }
            else if (exp == 31)
            {
                // NaN
                val = float.NaN;
            }
            else
            {
                // Normal: 2^(exp - 15) * (1 + mant / 4)
                val = MathF.Pow(2.0f, exp - 15.0f) * (1.0f + mant / 4.0f);
            }

            table[b] = sign == 0 ? val : -val;
        }
        return table;
    }
}
