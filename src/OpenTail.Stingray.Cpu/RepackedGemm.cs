using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Phase-2 (docs/cpu-prefill-repack-gemm-plan.md) checkpoint 1: the Q4_K row-interleave
/// transform, ported byte-for-byte from llama.cpp's <c>make_block_q4_Kx8</c>
/// (ggml/src/ggml-cpu/repack.cpp) so the repacked layout can be validated against the
/// existing scalar dequantizer before any GEMM kernel is written. No dot product lives here
/// yet — this is the plan's §8 "prove the data-layout transform is correct" checkpoint.
///
/// Block-group layout (mirrors <c>block_q4_Kx8</c> in repack.h): for every 8 source rows and
/// every Q4_K super-block column position, one 1152-byte group:
///   16 bytes  d[8]      (fp16 super-block scale, one per row)
///   16 bytes  dmin[8]   (fp16 super-block min, one per row)
///   96 bytes  scales    (the 8 rows' packed 6-bit scale/min values, re-interleaved)
///   1024 bytes qs       (the 8 rows' 4-bit quants, byte-interleaved 8 at a time)
/// </summary>
internal static class RepackedGemm
{
    internal const int Q4KBytesPerBlock = 144;
    internal const int Q4KGroupBytesPerBlock = 16 + 16 + 96 + 1024;
    internal const int RowsPerGroup = 8;

    /// <summary>
    /// Repacks 8 Q4_K rows (each <paramref name="blocksPerRow"/> blocks of 144 bytes, rows
    /// laid out consecutively in <paramref name="src"/>) into the interleaved block_q4_Kx8
    /// layout. <paramref name="src"/> must be exactly <c>8 * blocksPerRow * 144</c> bytes.
    /// </summary>
    internal static byte[] RepackQ4K8Rows(ReadOnlySpan<byte> src, int blocksPerRow)
    {
        if (src.Length != RowsPerGroup * blocksPerRow * Q4KBytesPerBlock)
            throw new ArgumentException(
                $"expected {RowsPerGroup * blocksPerRow * Q4KBytesPerBlock} bytes for {RowsPerGroup} rows x {blocksPerRow} blocks, got {src.Length}.",
                nameof(src));

        var dst = new byte[blocksPerRow * Q4KGroupBytesPerBlock];
        Span<int> blockOffset = stackalloc int[RowsPerGroup];
        Span<byte> s = stackalloc byte[RowsPerGroup];
        Span<byte> m = stackalloc byte[RowsPerGroup];

        for (int x = 0; x < blocksPerRow; x++)
        {
            var group = dst.AsSpan(x * Q4KGroupBytesPerBlock, Q4KGroupBytesPerBlock);
            var d = group.Slice(0, 16);
            var dmin = group.Slice(16, 16);
            var scales = group.Slice(32, 96);
            var qs = group.Slice(128, 1024);

            for (int r = 0; r < RowsPerGroup; r++)
                blockOffset[r] = (r * blocksPerRow + x) * Q4KBytesPerBlock;

            for (int r = 0; r < RowsPerGroup; r++)
            {
                var block = src.Slice(blockOffset[r], Q4KBytesPerBlock);
                block.Slice(0, 2).CopyTo(d.Slice(r * 2, 2));
                block.Slice(2, 2).CopyTo(dmin.Slice(r * 2, 2));
            }

            // qs: 128 groups of 8 bytes; src_id = i % 8, src_offset = (i/8)*8, dst_offset = i*8.
            for (int i = 0; i < 128; i++)
            {
                int srcId = i % RowsPerGroup;
                int srcOffset = (i / RowsPerGroup) * RowsPerGroup;
                int dstOffset = i * RowsPerGroup;
                var srcQs = src.Slice(blockOffset[srcId] + 16, 128); // qs starts at byte 16 of the 144-byte block
                srcQs.Slice(srcOffset, RowsPerGroup).CopyTo(qs.Slice(dstOffset, RowsPerGroup));
            }

            // scales: raw scales[12] bytes start at offset 4 of the 144-byte block.
            PackScalePass(src, blockOffset, i: 0, dstBase: scales, dstOffset: 0, s, m);
            PackScalePass(src, blockOffset, i: 1, dstBase: scales, dstOffset: 12, s, m);
            PackScalePass(src, blockOffset, i: 2, dstBase: scales, dstOffset: 24, s, m);
            PackScalePass(src, blockOffset, i: 3, dstBase: scales, dstOffset: 36, s, m);
            PackScaleHighPass(src, blockOffset, i: 0, dstBase: scales, dstOffset: 48, s, m);
            PackScaleHighPass(src, blockOffset, i: 1, dstBase: scales, dstOffset: 60, s, m);
            PackScaleHighPass(src, blockOffset, i: 2, dstBase: scales, dstOffset: 72, s, m);
            PackScaleHighPass(src, blockOffset, i: 3, dstBase: scales, dstOffset: 84, s, m);
        }

        return dst;
    }

    private static void PackScalePass(
        ReadOnlySpan<byte> src, ReadOnlySpan<int> blockOffset, int i,
        Span<byte> dstBase, int dstOffset, Span<byte> s, Span<byte> m)
    {
        for (int j = 0; j < RowsPerGroup; j++)
        {
            var raw = src.Slice(blockOffset[j] + 4, 12);
            s[j] = (byte)(raw[i] & 63);
            m[j] = (byte)(raw[i + 4] & 63);
        }
        WritePacked(dstBase, dstOffset, s, m);
    }

    private static void PackScaleHighPass(
        ReadOnlySpan<byte> src, ReadOnlySpan<int> blockOffset, int i,
        Span<byte> dstBase, int dstOffset, Span<byte> s, Span<byte> m)
    {
        for (int j = 0; j < RowsPerGroup; j++)
        {
            var raw = src.Slice(blockOffset[j] + 4, 12);
            s[j] = (byte)(((raw[i] & 192) >> 2) | (raw[i + 8] & 15));
            m[j] = (byte)(((raw[i + 4] & 192) >> 2) | ((raw[i + 8] & 240) >> 4));
        }
        WritePacked(dstBase, dstOffset, s, m);
    }

    private static void WritePacked(Span<byte> dstBase, int dstOffset, ReadOnlySpan<byte> s, ReadOnlySpan<byte> m)
    {
        dstBase[dstOffset + 0] = (byte)((s[0] & 63) + ((s[4] & 48) << 2));
        dstBase[dstOffset + 1] = (byte)((s[1] & 63) + ((s[5] & 48) << 2));
        dstBase[dstOffset + 2] = (byte)((s[2] & 63) + ((s[6] & 48) << 2));
        dstBase[dstOffset + 3] = (byte)((s[3] & 63) + ((s[7] & 48) << 2));
        dstBase[dstOffset + 4] = (byte)((m[0] & 63) + ((m[4] & 48) << 2));
        dstBase[dstOffset + 5] = (byte)((m[1] & 63) + ((m[5] & 48) << 2));
        dstBase[dstOffset + 6] = (byte)((m[2] & 63) + ((m[6] & 48) << 2));
        dstBase[dstOffset + 7] = (byte)((m[3] & 63) + ((m[7] & 48) << 2));
        dstBase[dstOffset + 8] = (byte)((s[4] & 15) + ((m[4] & 15) << 4));
        dstBase[dstOffset + 9] = (byte)((s[5] & 15) + ((m[5] & 15) << 4));
        dstBase[dstOffset + 10] = (byte)((s[6] & 15) + ((m[6] & 15) << 4));
        dstBase[dstOffset + 11] = (byte)((s[7] & 15) + ((m[7] & 15) << 4));
    }

    /// <summary>
    /// Dequantizes one row out of a repacked block_q4_Kx8 buffer directly to float, without
    /// reconstructing the original raw bytes — used only by the round-trip correctness test
    /// (never on a hot path). Must match <c>Dequantize.ToFloat32</c> on the original row
    /// exactly, since the repack is a pure permutation with no numeric transform.
    /// </summary>
    internal static void DequantizeRepackedQ4KRow(ReadOnlySpan<byte> repacked, int blocksPerRow, int row, Span<float> dst)
    {
        if ((uint)row >= RowsPerGroup)
            throw new ArgumentOutOfRangeException(nameof(row));

        const int QK_K = 256;
        Span<byte> scaleMin = stackalloc byte[16]; // [scale0..7, min0..7]

        for (int x = 0; x < blocksPerRow; x++)
        {
            var group = repacked.Slice(x * Q4KGroupBytesPerBlock, Q4KGroupBytesPerBlock);
            float dRow = HalfToFloat(group[row * 2], group[row * 2 + 1]);
            float dminRow = HalfToFloat(group[16 + row * 2], group[16 + row * 2 + 1]);
            var scales = group.Slice(32, 96);
            var qs = group.Slice(128, 1024);

            for (int i = 0; i < 4; i++)
                DecodeScaleMinPair(scales, i * 12, row, out scaleMin[i], out scaleMin[i + 8]);
            for (int i = 0; i < 4; i++)
                DecodeScaleMinPair(scales, i * 12 + 48, row, out scaleMin[i + 4], out scaleMin[i + 12]);

            var y = dst.Slice(x * QK_K, QK_K);
            int qIdx = 0;
            int subBlock = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                float d1 = dRow * scaleMin[subBlock];
                float dm1 = dminRow * scaleMin[subBlock + 8];
                float d2 = dRow * scaleMin[subBlock + 1];
                float dm2 = dminRow * scaleMin[subBlock + 9];

                for (int l = 0; l < 32; l++)
                {
                    byte qByte = ExtractQ(qs, row, qIdx + l);
                    y[j + l] = d1 * (qByte & 0xF) - dm1;
                    y[j + l + 32] = d2 * (qByte >> 4) - dm2;
                }
                qIdx += 32;
                subBlock += 2;
            }
        }
    }

    private static float HalfToFloat(byte lo, byte hi) =>
        (float)BitConverter.UInt16BitsToHalf((ushort)(lo | (hi << 8)));

    // Forward packing copies row `row`'s 8-byte span [group8*8, group8*8+8) as a unit (i = group8*8 + row),
    // so a given source byte index within the row lands at the same within-group offset in the destination.
    private static byte ExtractQ(ReadOnlySpan<byte> qs, int row, int qsByteIndexInRow)
    {
        int group8 = qsByteIndexInRow / 8;
        int withinGroup = qsByteIndexInRow % 8;
        int i = group8 * RowsPerGroup + row;
        return qs[i * RowsPerGroup + withinGroup];
    }

    private static void DecodeScaleMinPair(ReadOnlySpan<byte> scales12, int baseOffset, int row, out byte scale, out byte min)
    {
        if (row < 4)
        {
            scale = (byte)(scales12[baseOffset + row] & 0x3F);
            min = (byte)(scales12[baseOffset + 4 + row] & 0x3F);
        }
        else
        {
            int k = row - 4;
            byte packed = scales12[baseOffset + 8 + k];
            scale = (byte)((packed & 0x0F) | (((scales12[baseOffset + k] >> 6) & 0x03) << 4));
            min = (byte)(((packed >> 4) & 0x0F) | (((scales12[baseOffset + 4 + k] >> 6) & 0x03) << 4));
        }
    }

    /// <summary>
    /// Batched 16-token dispatch wrapper over <see cref="GemvQ4K8x8Q8K"/>: Groups activation rows
    /// in 16-token chunks per <see cref="Parallel.For"/> task to amortize thread dispatch overhead
    /// during multi-token prefill. Note: Intended as a dispatch helper for repacked 8-row weight groups;
    /// for maximum throughput on dense prefill, <see cref="SimdKernels.MatMulBatched"/> (_4In path) remains
    /// the primary production engine.
    /// </summary>
    internal static unsafe void GemmQ4K16x16Q8(
        float* output,           // [batchSize, rows]
        int lda,                 // rows
        byte* repackedGroups,    // [numGroups, blocksPerRow * Q4KGroupBytesPerBlock]
        int numGroups,           // total 8-row groups (rows / 8)
        byte* activationBase,    // [batchSize, Q8KScratchBytes(cols)]
        int batchSize,
        int blocksPerRow)
    {
        long bytesPerActRow = SimdKernels.Q8KScratchBytes(blocksPerRow * 256);
        Parallel.For(0, (batchSize + 15) / 16, mBlock =>
        {
            int mStart = mBlock * 16;
            int mEnd = Math.Min(mStart + 16, batchSize);

            for (int g = 0; g < numGroups; g++)
            {
                byte* groupPtr = repackedGroups + (long)g * blocksPerRow * Q4KGroupBytesPerBlock;
                for (int m = mStart; m < mEnd; m++)
                {
                    byte* actPtr = activationBase + m * bytesPerActRow;
                    float* outPtr = output + m * lda + g * 8;
                    GemvQ4K8x8Q8K(outPtr, groupPtr, actPtr, blocksPerRow);
                }
            }
        });
    }

    /// <summary>
    /// GEMV: one activation row against 8 repacked Q4_K weight rows, producing 8 outputs per
    /// call instead of <c>_4In</c>'s "4 activations against 1 row" batching — the weight-side
    /// axis §2 of the phase-2 plan targets. Direct port of llama.cpp's
    /// <c>ggml_gemv_q4_K_8x8_q8_K_generic</c> (repack.cpp:958), same accumulation order
    /// (per-subblock int accumulate, scale, sum), so any numerical delta from the reference
    /// scalar kernel is attributable to activation-quantization noise, not a reassociation this
    /// port introduced. <paramref name="activation"/> must be one row of
    /// <see cref="SimdKernels.QuantizeRowToQ8K"/> scratch (ggml's <c>block_q8_K</c> layout,
    /// NOT <c>Q8_KS</c> — this kernel's math assumes per-superblock scale + 16-way bsums).
    /// </summary>
    internal static unsafe void GemvQ4K8x8Q8K(float* output8, byte* repackedGroups, byte* activation, int blocksPerRow)
    {
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            GemvQ4K8x8Q8K_Avx2(output8, repackedGroups, activation, blocksPerRow);
            return;
        }
        GemvQ4K8x8Q8K_Scalar(output8, repackedGroups, activation, blocksPerRow);
    }

    /// <summary>
    /// AVX2 fast path: same per-column accumulation as the scalar/ggml-`_generic` reference
    /// (§12 of the phase-2 plan found the ported `_generic` function is llama.cpp's non-SIMD
    /// fallback — ~8x slower than production kernels here). Rather than replicating llama.cpp's
    /// x86-specific cross-column shuffle kernel (`arch/x86/repack.cpp`, register-choreography
    /// heavy), this reuses the codebase's own proven AVX2 idiom from
    /// <see cref="SimdKernels.DotQ4K_Q8KS"/> (nibble unpack + <c>MultiplyAddAdjacent</c>) per
    /// column, substituting Q8_K's single per-superblock scale + bsums for Q8_KS's per-32 scale.
    ///
    /// Scale/min for all 8 columns is decoded ONCE per super-block (the <c>utmp</c> unpack,
    /// same bit-twiddling as <see cref="GemvQ4K8x8Q8K_Scalar"/>) rather than once per column
    /// per chunk — each <c>utmp</c> byte, once unpacked, already holds one value per column
    /// directly indexable by <c>col</c> (`utmp[sb*16+col]` = scale, `utmp[sb*16+8+col]` = min).
    ///
    /// Each 32-byte "chunk" of one column's qs (in original-row order) is 4 non-contiguous
    /// 8-byte pieces in the repacked buffer (one per <c>kk</c> group). §16 of the phase-2 plan
    /// measured copying those through a scratch buffer first as ~32% of the kernel's cost;
    /// assembled directly here via 4 <c>Vector64</c> loads + <c>Vector128</c>/<c>Vector256</c>
    /// combines instead (§17), no scratch buffer, no scalar byte-by-byte copy loop.
    /// </summary>
    internal static unsafe void GemvQ4K8x8Q8K_Avx2(float* output8, byte* repackedGroups, byte* activation, int blocksPerRow)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;

        float* actD = (float*)activation;
        sbyte* actQs = (sbyte*)(activation + blocksPerRow * 4);
        short* actBsums = (short*)(activation + blocksPerRow * 4 + blocksPerRow * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        byte* utmp = stackalloc byte[128];
        uint* utmpWords = (uint*)utmp;

        float* acc = stackalloc float[8];
        for (int col = 0; col < 8; col++) acc[col] = 0f;

        var q8_0 = stackalloc Vector256<sbyte>[4];
        var q8_1 = stackalloc Vector256<sbyte>[4];

        for (int l = 0; l < blocksPerRow; l++)
        {
            byte* group = repackedGroups + l * Q4KGroupBytesPerBlock;
            byte* d16 = group;
            byte* dmin16 = group + 16;
            byte* scales = group + 32;
            byte* qs = group + 128;

            for (int sb = 0; sb < 8; sb++)
            {
                byte* raw12 = scales + sb * 12;
                uint u0 = *(uint*)raw12;
                uint u1 = *(uint*)(raw12 + 4);
                uint u2 = *(uint*)(raw12 + 8);
                uint u3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
                uint uaux0 = u1 & kmask1;
                u1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
                u2 = uaux0;
                u0 &= kmask1;
                utmpWords[sb * 4 + 0] = u0; // scale, cols 0-3
                utmpWords[sb * 4 + 1] = u1; // scale, cols 4-7
                utmpWords[sb * 4 + 2] = u2; // min, cols 0-3
                utmpWords[sb * 4 + 3] = u3; // min, cols 4-7
            }

            sbyte* q8 = actQs + l * 256;
            short* bsums = actBsums + l * 16;
            float dY = actD[l];

            // Activation vectors depend only on (l, chunk), not on the weight column -- load
            // each chunk's two 32-wide q8 vectors once here instead of once per column below
            // (was 8x redundant: phase-2 plan §15).
            for (int chunk = 0; chunk < 4; chunk++)
            {
                q8_0[chunk] = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
                q8_1[chunk] = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
            }

            for (int col = 0; col < 8; col++)
            {
                float dRow = HalfToFloat(d16[col * 2], d16[col * 2 + 1]);
                float dminRow = HalfToFloat(dmin16[col * 2], dmin16[col * 2 + 1]);

                for (int chunk = 0; chunk < 4; chunk++)
                {
                    int sb0 = 2 * chunk, sb1 = 2 * chunk + 1;
                    byte sc1 = utmp[sb0 * 16 + col], m1 = utmp[sb0 * 16 + 8 + col];
                    byte sc2 = utmp[sb1 * 16 + col], m2 = utmp[sb1 * 16 + 8 + col];

                    // The original 32-byte "chunk" (row-order bytes [chunk*32, chunk*32+32))
                    // is physically 4 non-contiguous 8-byte pieces in the repacked buffer, one
                    // per kk = chunk*4 + t (t=0..3), each at qs[kk*64 + col*8 .. +8). Assemble
                    // the Vector256 directly from 4 small vector loads instead of copying
                    // through a scratch buffer first (phase-2 plan §17 — the de-interleave
                    // copy loop measured as ~32% of the kernel's cost in §16).
                    int kkBase = chunk * 4;
                    var v0 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 0) * 64 + col * 8));
                    var v1 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 1) * 64 + col * 8));
                    var v2 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 2) * 64 + col * 8));
                    var v3 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 3) * 64 + col * 8));
                    var qbytes = Vector256.Create(Vector128.Create(v0, v1), Vector128.Create(v2, v3));
                    var lo = Avx2.And(qbytes, m0F);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                    var p16_0 = Avx2.MultiplyAddAdjacent(lo, q8_0[chunk]);
                    int sub0 = HSumI32_256(Avx2.MultiplyAddAdjacent(p16_0, one16));
                    int bsum0 = bsums[2 * sb0] + bsums[2 * sb0 + 1];
                    acc[col] += dY * (dRow * sc1 * sub0 - dminRow * m1 * bsum0);

                    var p16_1 = Avx2.MultiplyAddAdjacent(hi, q8_1[chunk]);
                    int sub1 = HSumI32_256(Avx2.MultiplyAddAdjacent(p16_1, one16));
                    int bsum1 = bsums[2 * sb1] + bsums[2 * sb1 + 1];
                    acc[col] += dY * (dRow * sc2 * sub1 - dminRow * m2 * bsum1);
                }
            }
        }

        for (int col = 0; col < 8; col++)
            output8[col] = acc[col];
    }

    /// <summary>
    /// GEMM: 4 activation rows against 8 repacked Q4_K weight rows per call (32 outputs) —
    /// the genuine 2D tile llama.cpp's real GEMM kernel gets and <see cref="GemvQ4K8x8Q8K"/>
    /// (1 token × 8 columns) doesn't. Rather than building a new interleaved
    /// <c>block_q8_Kx4</c>-style activation format (more new surface to get subtly wrong), this
    /// reuses 4 independent <see cref="SimdKernels.QuantizeRowToQ8K"/> scratch buffers exactly
    /// as <see cref="GemvQ4K8x8Q8K"/> already does, and shares the expensive per-(column,chunk)
    /// weight-nibble decode (<c>qbytes</c>/<c>lo</c>/<c>hi</c>) across all 4 tokens instead of
    /// redoing it once per token — the actual 2D reuse, without new activation-layout risk.
    /// </summary>
    internal static unsafe void GemmQ4K8x8x4Q8K_Avx2(
        float* out0, float* out1, float* out2, float* out3,
        byte* repackedGroups, byte* act0, byte* act1, byte* act2, byte* act3, int blocksPerRow)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        byte* utmp = stackalloc byte[128];
        uint* utmpWords = (uint*)utmp;

        float* acc0 = stackalloc float[8]; float* acc1 = stackalloc float[8];
        float* acc2 = stackalloc float[8]; float* acc3 = stackalloc float[8];
        for (int col = 0; col < 8; col++) { acc0[col] = 0f; acc1[col] = 0f; acc2[col] = 0f; acc3[col] = 0f; }

        byte*[] actPtrs = [act0, act1, act2, act3];
        var q8_0 = stackalloc Vector256<sbyte>[4 * 4]; // [token, chunk]
        var q8_1 = stackalloc Vector256<sbyte>[4 * 4];
        float* dYs = stackalloc float[4];
        short*[] bsumsArr = new short*[4];

        for (int l = 0; l < blocksPerRow; l++)
        {
            byte* group = repackedGroups + l * Q4KGroupBytesPerBlock;
            byte* d16 = group;
            byte* dmin16 = group + 16;
            byte* scales = group + 32;
            byte* qs = group + 128;

            for (int sb = 0; sb < 8; sb++)
            {
                byte* raw12 = scales + sb * 12;
                uint u0 = *(uint*)raw12;
                uint u1 = *(uint*)(raw12 + 4);
                uint u2 = *(uint*)(raw12 + 8);
                uint u3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
                uint uaux0 = u1 & kmask1;
                u1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
                u2 = uaux0;
                u0 &= kmask1;
                utmpWords[sb * 4 + 0] = u0;
                utmpWords[sb * 4 + 1] = u1;
                utmpWords[sb * 4 + 2] = u2;
                utmpWords[sb * 4 + 3] = u3;
            }

            for (int t = 0; t < 4; t++)
            {
                byte* act = actPtrs[t];
                float* actD = (float*)act;
                sbyte* actQs = (sbyte*)(act + blocksPerRow * 4);
                short* actBsums = (short*)(act + blocksPerRow * 4 + blocksPerRow * 256);
                dYs[t] = actD[l];
                bsumsArr[t] = actBsums + l * 16;
                sbyte* q8 = actQs + l * 256;
                for (int chunk = 0; chunk < 4; chunk++)
                {
                    q8_0[t * 4 + chunk] = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
                    q8_1[t * 4 + chunk] = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
                }
            }

            for (int col = 0; col < 8; col++)
            {
                float dRow = HalfToFloat(d16[col * 2], d16[col * 2 + 1]);
                float dminRow = HalfToFloat(dmin16[col * 2], dmin16[col * 2 + 1]);

                for (int chunk = 0; chunk < 4; chunk++)
                {
                    int sb0 = 2 * chunk, sb1 = 2 * chunk + 1;
                    byte sc1 = utmp[sb0 * 16 + col], m1 = utmp[sb0 * 16 + 8 + col];
                    byte sc2 = utmp[sb1 * 16 + col], m2 = utmp[sb1 * 16 + 8 + col];

                    // Weight-nibble decode -- shared across all 4 tokens below (the actual 2D reuse).
                    int kkBase = chunk * 4;
                    var v0 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 0) * 64 + col * 8));
                    var v1 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 1) * 64 + col * 8));
                    var v2 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 2) * 64 + col * 8));
                    var v3 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 3) * 64 + col * 8));
                    var qbytes = Vector256.Create(Vector128.Create(v0, v1), Vector128.Create(v2, v3));
                    var lo = Avx2.And(qbytes, m0F);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                    AccumToken(lo, hi, one16, q8_0[0 * 4 + chunk], q8_1[0 * 4 + chunk],
                        dRow, dminRow, sc1, m1, sc2, m2, bsumsArr[0], sb0, sb1, dYs[0], ref acc0[col]);
                    AccumToken(lo, hi, one16, q8_0[1 * 4 + chunk], q8_1[1 * 4 + chunk],
                        dRow, dminRow, sc1, m1, sc2, m2, bsumsArr[1], sb0, sb1, dYs[1], ref acc1[col]);
                    AccumToken(lo, hi, one16, q8_0[2 * 4 + chunk], q8_1[2 * 4 + chunk],
                        dRow, dminRow, sc1, m1, sc2, m2, bsumsArr[2], sb0, sb1, dYs[2], ref acc2[col]);
                    AccumToken(lo, hi, one16, q8_0[3 * 4 + chunk], q8_1[3 * 4 + chunk],
                        dRow, dminRow, sc1, m1, sc2, m2, bsumsArr[3], sb0, sb1, dYs[3], ref acc3[col]);
                }
            }
        }

        for (int col = 0; col < 8; col++)
        {
            out0[col] = acc0[col]; out1[col] = acc1[col]; out2[col] = acc2[col]; out3[col] = acc3[col];
        }
    }

    /// <summary>
    /// GEMM: 8 activation rows against 8 repacked Q4_K weight rows per call (64 outputs) —
    /// widens <see cref="GemmQ4K8x8x4Q8K_Avx2"/>'s token axis from 4 to 8 to match `_8In`'s
    /// own token-side reuse width (phase-2 plan §23's tried-next item (a)), while keeping the
    /// column-side reuse `_8In` doesn't have. Same design: 8 independent
    /// <see cref="SimdKernels.QuantizeRowToQ8K"/> scratch buffers, no new activation format;
    /// the per-(column,chunk) weight-nibble decode is shared across all 8 tokens instead of 4.
    /// <paramref name="outs"/> and <paramref name="acts"/> are 8-element pointer arrays (one
    /// float[8] output buffer and one Q8_K scratch buffer per token).
    /// </summary>
    internal static unsafe void GemmQ4K8x8x8Q8K_Avx2(
        float** outs, byte* repackedGroups, byte** acts, int blocksPerRow)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        byte* utmp = stackalloc byte[128];
        uint* utmpWords = (uint*)utmp;

        float* acc = stackalloc float[8 * 8]; // [token, col]
        for (int i = 0; i < 64; i++) acc[i] = 0f;

        var q8_0 = stackalloc Vector256<sbyte>[8 * 4]; // [token, chunk]
        var q8_1 = stackalloc Vector256<sbyte>[8 * 4];
        float* dYs = stackalloc float[8];
        short*[] bsumsArr = new short*[8];

        for (int l = 0; l < blocksPerRow; l++)
        {
            byte* group = repackedGroups + l * Q4KGroupBytesPerBlock;
            byte* d16 = group;
            byte* dmin16 = group + 16;
            byte* scales = group + 32;
            byte* qs = group + 128;

            for (int sb = 0; sb < 8; sb++)
            {
                byte* raw12 = scales + sb * 12;
                uint u0 = *(uint*)raw12;
                uint u1 = *(uint*)(raw12 + 4);
                uint u2 = *(uint*)(raw12 + 8);
                uint u3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
                uint uaux0 = u1 & kmask1;
                u1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
                u2 = uaux0;
                u0 &= kmask1;
                utmpWords[sb * 4 + 0] = u0;
                utmpWords[sb * 4 + 1] = u1;
                utmpWords[sb * 4 + 2] = u2;
                utmpWords[sb * 4 + 3] = u3;
            }

            for (int t = 0; t < 8; t++)
            {
                byte* act = acts[t];
                float* actD = (float*)act;
                sbyte* actQs = (sbyte*)(act + blocksPerRow * 4);
                short* actBsums = (short*)(act + blocksPerRow * 4 + blocksPerRow * 256);
                dYs[t] = actD[l];
                bsumsArr[t] = actBsums + l * 16;
                sbyte* q8 = actQs + l * 256;
                for (int chunk = 0; chunk < 4; chunk++)
                {
                    q8_0[t * 4 + chunk] = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
                    q8_1[t * 4 + chunk] = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
                }
            }

            for (int col = 0; col < 8; col++)
            {
                float dRow = HalfToFloat(d16[col * 2], d16[col * 2 + 1]);
                float dminRow = HalfToFloat(dmin16[col * 2], dmin16[col * 2 + 1]);

                for (int chunk = 0; chunk < 4; chunk++)
                {
                    int sb0 = 2 * chunk, sb1 = 2 * chunk + 1;
                    byte sc1 = utmp[sb0 * 16 + col], m1 = utmp[sb0 * 16 + 8 + col];
                    byte sc2 = utmp[sb1 * 16 + col], m2 = utmp[sb1 * 16 + 8 + col];

                    // Weight-nibble decode -- shared across all 8 tokens below (the 2D reuse).
                    int kkBase = chunk * 4;
                    var v0 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 0) * 64 + col * 8));
                    var v1 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 1) * 64 + col * 8));
                    var v2 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 2) * 64 + col * 8));
                    var v3 = Vector64.LoadUnsafe(ref *(qs + (kkBase + 3) * 64 + col * 8));
                    var qbytes = Vector256.Create(Vector128.Create(v0, v1), Vector128.Create(v2, v3));
                    var lo = Avx2.And(qbytes, m0F);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                    for (int t = 0; t < 8; t++)
                    {
                        AccumToken(lo, hi, one16, q8_0[t * 4 + chunk], q8_1[t * 4 + chunk],
                            dRow, dminRow, sc1, m1, sc2, m2, bsumsArr[t], sb0, sb1, dYs[t],
                            ref acc[t * 8 + col]);
                    }
                }
            }
        }

        for (int t = 0; t < 8; t++)
            for (int col = 0; col < 8; col++)
                outs[t][col] = acc[t * 8 + col];
    }

    private static unsafe void AccumToken(
        Vector256<byte> lo, Vector256<byte> hi, Vector256<short> one16,
        Vector256<sbyte> q8_0, Vector256<sbyte> q8_1,
        float dRow, float dminRow, byte sc1, byte m1, byte sc2, byte m2,
        short* bsums, int sb0, int sb1, float dY, ref float acc)
    {
        var p16_0 = Avx2.MultiplyAddAdjacent(lo, q8_0);
        int sub0 = HSumI32_256(Avx2.MultiplyAddAdjacent(p16_0, one16));
        int bsum0 = bsums[2 * sb0] + bsums[2 * sb0 + 1];
        acc += dY * (dRow * sc1 * sub0 - dminRow * m1 * bsum0);

        var p16_1 = Avx2.MultiplyAddAdjacent(hi, q8_1);
        int sub1 = HSumI32_256(Avx2.MultiplyAddAdjacent(p16_1, one16));
        int bsum1 = bsums[2 * sb1] + bsums[2 * sb1 + 1];
        acc += dY * (dRow * sc2 * sub1 - dminRow * m2 * bsum1);
    }

    private static int HSumI32_256(Vector256<int> v)
    {
        var lo = v.GetLower();
        var hi = Avx.ExtractVector128(v, 1);
        var s = Sse2.Add(lo, hi);
        s = Sse2.Add(s, Sse2.Shuffle(s, 0x4E)); // [2,3,0,1]
        s = Sse2.Add(s, Sse2.Shuffle(s, 0xB1)); // [1,0,3,2]
        return s.ToScalar();
    }

    internal static unsafe void GemvQ4K8x8Q8K_Scalar(float* output8, byte* repackedGroups, byte* activation, int blocksPerRow)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;

        float* sumf = stackalloc float[8];
        float* sumMinf = stackalloc float[8];
        byte* utmp = stackalloc byte[128]; // 32 uint32 words, viewed as bytes (matches ggml's uint8_t* utmp casts)
        uint* utmpWords = (uint*)utmp;

        float* actD = (float*)activation;
        sbyte* actQs = (sbyte*)(activation + blocksPerRow * 4);
        short* actBsums = (short*)(activation + blocksPerRow * 4 + blocksPerRow * 256);

        for (int j = 0; j < 8; j++) { sumf[j] = 0f; sumMinf[j] = 0f; }

        for (int l = 0; l < blocksPerRow; l++)
        {
            byte* group = repackedGroups + l * Q4KGroupBytesPerBlock;
            byte* d = group;
            byte* dmin = group + 16;
            byte* scales = group + 32;
            byte* qs = group + 128;

            for (int sb = 0; sb < 8; sb++)
            {
                byte* raw12 = scales + sb * 12;
                uint u0 = *(uint*)raw12;
                uint u1 = *(uint*)(raw12 + 4);
                uint u2 = *(uint*)(raw12 + 8);
                uint u3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
                uint uaux0 = u1 & kmask1;
                u1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
                u2 = uaux0;
                u0 &= kmask1;
                utmpWords[sb * 4 + 0] = u0;
                utmpWords[sb * 4 + 1] = u1;
                utmpWords[sb * 4 + 2] = u2;
                utmpWords[sb * 4 + 3] = u3;
            }

            sbyte* qy = actQs + l * 256;
            short* bsumsY = actBsums + l * 16;
            float dY = actD[l];

            for (int k = 0; k < 16; k++)
            {
                byte* scales0 = utmp + (k / 4) * 32;
                byte* scales1 = utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < 8; j++)
                {
                    int sumi = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        byte qByte = qs[k * 64 + j * 8 + i];
                        int v0 = qByte & 0xF;
                        int v1 = qByte >> 4;
                        int sumi1 = v0 * qy[(k >> 2) * 64 + (k % 4) * 8 + i];
                        int sumi2 = v1 * qy[(k >> 2) * 64 + (k % 4) * 8 + i + 32];
                        sumi1 *= scales0[j];
                        sumi2 *= scales1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * HalfToFloat(d[j * 2], d[j * 2 + 1]) * dY;
                }
            }

            for (int sb = 0; sb < 8; sb++)
            {
                byte* mins = utmp + 8 + sb * 16;
                int bsum = bsumsY[sb * 2] + bsumsY[sb * 2 + 1];
                for (int j = 0; j < 8; j++)
                    sumMinf[j] += mins[j] * bsum * HalfToFloat(dmin[j * 2], dmin[j * 2 + 1]) * dY;
            }
        }

        for (int j = 0; j < 8; j++)
            output8[j] = sumf[j] - sumMinf[j];
    }
}
