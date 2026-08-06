using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Phase-2 attempt at a literal port of llama.cpp's actual plain-AVX2 GEMM kernel
/// (<c>ggml_gemm_q4_K_8x8_q8_K</c>, the 4-tokens×8-columns tiled path in
/// <c>ggml-cpu/arch/x86/repack.cpp</c> lines ~2818-3157 — NOT the <c>_generic</c> scalar
/// fallback ported earlier, and NOT this codebase's own from-scratch AVX2 idiom
/// (<see cref="RepackedGemm.GemmQ4K8x8x4Q8K_Avx2"/>). docs/cpu-prefill-repack-gemm-plan.md
/// §32+ tracks this.
///
/// Given the kernel's size (~340 lines of interdependent shuffle/blend immediates) and that
/// a wrong immediate produces silently-wrong logits rather than a crash, this is built and
/// tested one isolated "seam" at a time against a hand-computed scalar reference, rather than
/// transcribed whole and trusted. Each seam is a pure data transform with no dependency on the
/// ones after it, so a mistake is caught at the seam that introduced it.
/// </summary>
internal static class RealAvx2Gemm
{
    /// <summary>
    /// Seam 1: the "0145/2367 column-pair rearrange" — <c>rhs_raw_mat_0145_0</c> /
    /// <c>rhs_raw_mat_2367_0</c> in the C source. Pure byte permutation, no arithmetic: takes
    /// one repacked <c>kk</c>-block's 64 bytes (columns 0-3's 8-byte chunks followed by
    /// columns 4-7's, exactly the layout <see cref="RepackedGemm.RepackQ4K8Rows"/> already
    /// produces and checkpoint 1 already proved correct) and reorders them into two 32-byte
    /// groups: columns [0,1,4,5] and columns [2,3,6,7], each still 8 bytes per column.
    ///
    /// Traced by hand against the C intrinsics (not guessed): <c>requiredOrder =
    /// {4,5,6,7,0,1,2,3}</c> swaps the low/high 4-lane halves of the cols4-7 vector, so
    /// <c>permutevar8x32(cols4567, requiredOrder)</c> becomes [col6,col7,col4,col5]; blending
    /// that with cols0-3 using mask 0xF0 (high 4 lanes from the permuted operand) yields
    /// [col0,col1,col4,col5] — hence "0145". The mirror computation for cols2367 confirms the
    /// same pattern with the operand roles swapped.
    /// </summary>
    internal static unsafe void RearrangeColumnPairs0145_2367(
        byte* qsGroup, int kk, out Vector256<byte> mat0145, out Vector256<byte> mat2367)
    {
        var cols0123 = Vector256.LoadUnsafe(ref *(qsGroup + kk * 64));
        var cols4567 = Vector256.LoadUnsafe(ref *(qsGroup + kk * 64 + 32));

        var requiredOrder = Vector256.Create(4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u);
        var cols4567Swapped = Avx2.PermuteVar8x32(cols4567.AsUInt32(), requiredOrder).AsByte();
        var cols0123Swapped = Avx2.PermuteVar8x32(cols0123.AsUInt32(), requiredOrder).AsByte();

        // Blend mask 0xF0 = high 4 of 8 int32 lanes come from the second operand.
        mat0145 = Avx2.Blend(cols0123.AsInt32(), cols4567Swapped.AsInt32(), 0xF0).AsByte();
        mat2367 = Avx2.Blend(cols0123Swapped.AsInt32(), cols4567.AsInt32(), 0xF0).AsByte();
    }

    /// <summary>
    /// Seam 2: 4-bit → 8-bit nibble unpack (<c>rhs_mat_0145_00</c>/<c>rhs_mat_0145_10</c> etc.
    /// in the C source) — same mask/shift pattern already used correctly elsewhere in this
    /// codebase (e.g. <c>DotQ4K</c>), applied to seam 1's rearranged output instead of a raw
    /// row. Low nibble = "sub-block 0" (<paramref name="lo"/>), high nibble = "sub-block 1"
    /// (<paramref name="hi"/>), unsigned 0-15 (no centering — matches every other Q4_K kernel
    /// in this file, the min-correction supplies the offset separately).
    /// </summary>
    internal static void UnpackNibbles(Vector256<byte> packed, out Vector256<byte> lo, out Vector256<byte> hi)
    {
        var m4b = Vector256.Create((byte)0x0F);
        lo = Avx2.And(packed, m4b);
        hi = Avx2.And(Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), m4b);
    }

    /// <summary>
    /// Seam 3: the weight-side "sp1/sp2" duplicate-shuffle (<c>rhs_mat_*_sp1</c>/<c>_sp2</c> in
    /// the C source, <c>_mm256_shuffle_epi32</c> with immediates 136/221). Input is one of
    /// seam 2's nibble-unpacked vectors: per 128-bit lane, 4 dwords = [colA bytes0-3, colA
    /// bytes4-7, colB bytes0-3, colB bytes4-7] (a nibble-unpacked column is 8 bytes = 2
    /// dwords). <c>vpshufd</c> applies the same 4-element index pattern independently to each
    /// 128-bit lane.
    ///
    /// Traced by hand: imm 136 = 0b10_00_10_00 = per-lane index pattern [0,2,0,2] → picks
    /// dword0 (colA 0-3) and dword2 (colB 0-3), each written twice → "B_A(0-3) B_B(0-3)
    /// B_A(0-3) B_B(0-3)" per lane, matching the source comments exactly. imm 221 =
    /// 0b11_01_11_01 = pattern [1,3,1,3] → the (4-7) byte range instead. This duplication is
    /// what lets one <c>maddubs_epi16</c> call cover two output columns' partial sums from a
    /// single load — the actual weight-side reuse mechanism.
    /// </summary>
    internal static void DuplicateShuffleWeightPattern(
        Vector256<byte> unpacked, out Vector256<byte> sp1, out Vector256<byte> sp2)
    {
        sp1 = Avx2.Shuffle(unpacked.AsInt32(), 0b10_00_10_00).AsByte(); // 136
        sp2 = Avx2.Shuffle(unpacked.AsInt32(), 0b11_01_11_01).AsByte(); // 221
    }

    /// <summary>
    /// Seam 4: the activation-side load + cross-128-lane broadcast (<c>lhs_mat_01_00</c> /
    /// <c>lhs_mat_23_00</c> in the C source, <c>_mm256_permute2f128_si256(x, x, imm)</c>).
    /// One 32-byte load of a token's Q8_K quants (4 tokens' worth interleaved in
    /// <c>block_q8_Kx4</c>, but this seam only needs one already-loaded 256-bit vector, not
    /// the interleaved format itself) is broadcast two ways: its low 128 bits repeated across
    /// both halves ("01" — tokens 0,1's data), and its high 128 bits repeated across both
    /// halves ("23" — tokens 2,3's data).
    ///
    /// Traced by hand: <c>_mm256_permute2f128_si256(a, b, imm)</c> selects the output's low
    /// 128 bits from one of {a_low, a_high, b_low, b_high} via <c>imm</c> bits[1:0], and the
    /// high 128 bits via bits[5:4]. With <c>a == b == x</c>: imm=0 (bits 00_00) selects
    /// a_low/a_low → broadcast-low. imm=17=0b00010001 (bits 01_01) selects a_high/a_high →
    /// broadcast-high.
    /// </summary>
    internal static void BroadcastLowHigh128(
        Vector256<byte> loaded, out Vector256<byte> broadcastLow, out Vector256<byte> broadcastHigh)
    {
        broadcastLow = Vector256.Create(loaded.GetLower(), loaded.GetLower());
        broadcastHigh = Vector256.Create(loaded.GetUpper(), loaded.GetUpper());
    }

    /// <summary>
    /// Seam 5: the activation-side "sp1/sp2" duplicate-shuffle (<c>lhs_mat_*_sp1</c>/<c>_sp2</c>
    /// in the C source, <c>_mm256_shuffle_epi32</c> with immediates 160/245). Input is one of
    /// seam 4's broadcast vectors: per 128-bit lane, 4 dwords = [tokenA bytes0-3, tokenA
    /// bytes4-7, tokenB bytes0-3, tokenB bytes4-7].
    ///
    /// Different immediates from seam 3 despite the structurally identical dword layout,
    /// because the target duplication differs: weight wants "column,column,column,column"
    /// alternating (to line up against two different <c>maddubs</c> lanes), activation wants
    /// each token's own bytes duplicated adjacently. Traced by hand: imm 160 = 0b10_10_00_00 →
    /// per-lane pattern [0,0,2,2] → dword0 (tokenA 0-3) twice, dword2 (tokenB 0-3) twice. imm
    /// 245 = 0b11_11_01_01 → pattern [1,1,3,3] → the (4-7) byte range instead, same doubling.
    /// </summary>
    internal static void DuplicateShuffleActivationPattern(
        Vector256<byte> broadcast, out Vector256<byte> sp1, out Vector256<byte> sp2)
    {
        sp1 = Avx2.Shuffle(broadcast.AsInt32(), 0b10_10_00_00).AsByte(); // 160
        sp2 = Avx2.Shuffle(broadcast.AsInt32(), 0b11_11_01_01).AsByte(); // 245
    }

    /// <summary>
    /// Seam 6: the <c>maddubs_epi16</c> dot + <c>add_epi16</c> combine chain
    /// (<c>iacc_mat_00_0_sp1</c> etc. in the C source) — the first seam with real arithmetic,
    /// not pure permutation. Layout-agnostic: takes 4 (weight, activation) byte-vector pairs
    /// (one per 8-byte sub-chunk of a 32-byte super-chunk) and produces one accumulated int16
    /// vector, exactly matching <c>((maddubs(rhs3,lhs3) + maddubs(rhs2,lhs2)) + maddubs(rhs1,
    /// lhs1)) + maddubs(rhs0,lhs0)</c> — same nested order as the source, not reassociated,
    /// so this is safe even though int16 addition isn't associative under overflow.
    /// <c>maddubs_epi16</c> itself is unsigned-weight × signed-activation → signed int16 pairs,
    /// matching every other Q4_K int8 kernel already in this codebase (weight nibbles are
    /// unsigned 0-15, never centered).
    /// </summary>
    internal static Vector256<short> MaddubsAccumulate4(
        Vector256<byte> rhs0, Vector256<sbyte> lhs0,
        Vector256<byte> rhs1, Vector256<sbyte> lhs1,
        Vector256<byte> rhs2, Vector256<sbyte> lhs2,
        Vector256<byte> rhs3, Vector256<sbyte> lhs3)
    {
        var p0 = Avx2.MultiplyAddAdjacent(rhs0, lhs0);
        var p1 = Avx2.MultiplyAddAdjacent(rhs1, lhs1);
        var p2 = Avx2.MultiplyAddAdjacent(rhs2, lhs2);
        var p3 = Avx2.MultiplyAddAdjacent(rhs3, lhs3);
        return Avx2.Add(Avx2.Add(Avx2.Add(p3, p2), p1), p0);
    }

    /// <summary>
    /// Seam 7: the per-column-group scale extraction + <c>madd_epi16</c> reduce
    /// (<c>scale_0145_0</c>/<c>scale_2367_0</c> derivation at C source lines 2983-2984, applied
    /// at lines 3104-3107). <paramref name="rawScales"/> is the 16-lane int16 scale vector this
    /// codebase already builds and tests via the <c>utmp</c> decode in
    /// <see cref="RepackedGemm"/> (not re-derived here) — <c>_mm256_cvtepu8_epi16</c> zero-
    /// extending 16 scale bytes. <c>vpshufd</c> imm 68 = 0b01_00_01_00 → per-128-bit-lane dword
    /// pattern [0,1,0,1] (duplicates the columns-0145 scale pair); imm 238 = 0b11_10_11_10 →
    /// pattern [2,3,2,3] (columns-2367 pair) — same derivation technique as seams 3/5, different
    /// immediates because this duplicates whole dword *pairs*, not one dword at a time.
    /// <c>madd_epi16</c> then pairwise-multiplies-and-sums adjacent int16 lanes into int32 —
    /// the .NET intrinsic already returns <c>Vector256&lt;int&gt;</c> directly, matching
    /// <c>_mm256_madd_epi16</c> exactly.
    /// </summary>
    internal static Vector256<int> ScaleAndReduce0145(Vector256<short> acc, Vector256<short> rawScales)
    {
        var scale0145 = Avx2.Shuffle(rawScales.AsInt32(), 0b01_00_01_00).AsInt16(); // 68
        return Avx2.MultiplyAddAdjacent(acc, scale0145);
    }

    /// <summary>See <see cref="ScaleAndReduce0145"/> — the columns-2367 mirror (imm 238).</summary>
    internal static Vector256<int> ScaleAndReduce2367(Vector256<short> acc, Vector256<short> rawScales)
    {
        var scale2367 = Avx2.Shuffle(rawScales.AsInt32(), 0b11_10_11_10).AsInt16(); // 238
        return Avx2.MultiplyAddAdjacent(acc, scale2367);
    }

    /// <summary>
    /// Seam 8: "straighten out to make 4 row vectors" (C source lines 3114-3118, comment
    /// verbatim: "Straighten out to make 4 row vectors (4 for each sub block which are
    /// accumulated together in the next step)"). Inputs are seam 7's four scaled int32
    /// accumulators — <c>mat00</c>/<c>mat01</c> hold row-pair (0,1)'s partial sums for column
    /// groups 0145/2367 respectively (8 int32 lanes = one value per output column, but rows 0
    /// and 1 interleaved 4+4 within the vector because seam 4 broadcast both tokens' data into
    /// one register); <c>mat10</c>/<c>mat11</c> are the row-pair (2,3) mirror. This seam
    /// untangles that interleaving into one vector per row.
    ///
    /// Traced by hand: <c>vpshufd</c> imm 78 = 0b01_00_11_10 → per-128-bit-lane dword pattern
    /// [2,3,0,1] (swaps each lane's low/high dword pairs). <c>_mm256_blend_epi32</c> imm 204 =
    /// 0b11001100 selects dwords {2,3,6,7} from the second operand, {0,1,4,5} from the first.
    /// Combining: <c>row0 = blend(mat00, shuffle(mat01,78), 204)</c> keeps row 0's own dwords
    /// from <c>mat00</c> (lanes 0,1,4,5) and pulls row 0's dwords out of <c>mat01</c>'s swapped
    /// layout (lanes 2,3,6,7, which after the swap now hold what was originally at 0,1) — i.e.
    /// each row vector picks its own column-045-group values from one input and its own
    /// column-2367-group values from the other, exactly the row/column-group cross the source
    /// comment describes.
    /// </summary>
    internal static void StraightenToRowVectors(
        Vector256<int> mat00, Vector256<int> mat01, Vector256<int> mat10, Vector256<int> mat11,
        out Vector256<int> row0, out Vector256<int> row1, out Vector256<int> row2, out Vector256<int> row3)
    {
        const byte swapImm = 0b01_00_11_10; // 78
        const byte blendImm = 204;

        var mat00Swapped = Avx2.Shuffle(mat00, swapImm);
        var mat01Swapped = Avx2.Shuffle(mat01, swapImm);
        var mat10Swapped = Avx2.Shuffle(mat10, swapImm);
        var mat11Swapped = Avx2.Shuffle(mat11, swapImm);

        row0 = Avx2.Blend(mat00, mat01Swapped, blendImm);
        row1 = Avx2.Blend(mat00Swapped, mat01, blendImm);
        row2 = Avx2.Blend(mat10, mat11Swapped, blendImm);
        row3 = Avx2.Blend(mat10Swapped, mat11, blendImm);
    }

    /// <summary>
    /// Seam 9a: the bsums-based min correction (<c>iacc_row_min_i = madd_epi16(shuffle_epi32
    /// (lhs_bsums_hsum_0123_01, imm), mins_01)</c>, C source lines 3139-3142). Takes the
    /// per-token bsums-hsum vector and the per-sub-block mins vector as already-built inputs
    /// (both constructed earlier in the C function, outside this port's seam set — this seam
    /// only covers the shuffle+madd combine). <c>vpshufd</c> imm ∈ {0, 85, 170, 255} each
    /// broadcast a single source dword to all 4 dword slots per 128-bit lane (0=0b00000000→dword
    /// 0 everywhere, 85=0b01010101→dword 1, 170=0b10101010→dword 2, 255=0b11111111→dword 3) —
    /// selects one token's bsum pair and replicates it across the vector before the same
    /// <c>madd_epi16</c> pairwise-dot-and-sum used in <see cref="ScaleAndReduce0145"/>.
    /// </summary>
    internal static Vector256<int> BsumsMinCorrection(Vector256<short> bsumsHsum, int tokenIndex, Vector256<short> mins01)
    {
        // Shuffle immediate must be a compile-time constant for the JIT to emit vpshufd, hence
        // the switch instead of passing the byte straight through.
        Vector256<int> broadcast = tokenIndex switch
        {
            0 => Avx2.Shuffle(bsumsHsum.AsInt32(), 0),
            1 => Avx2.Shuffle(bsumsHsum.AsInt32(), 85),
            2 => Avx2.Shuffle(bsumsHsum.AsInt32(), 170),
            3 => Avx2.Shuffle(bsumsHsum.AsInt32(), 255),
            _ => throw new ArgumentOutOfRangeException(nameof(tokenIndex)),
        };
        return Avx2.MultiplyAddAdjacent(broadcast.AsInt16(), mins01);
    }

    /// <summary>
    /// Seam 9b: the final per-row FP scale/min-correction combine and store value (C source
    /// lines 3134-3137 + 3144-3147 + 3154, one row's worth). <c>row_scale</c> holds one d-value
    /// per token in 4 lanes, broadcast to 8; <c>vshufps</c> imm ∈ {0,85,170,255} broadcasts a
    /// single token's d-value the same way <see cref="BsumsMinCorrection"/>'s imm broadcasts a
    /// dword (same four immediates, same "pick one of 4, replicate" mechanism, float lanes
    /// instead of int32). <c>colScale</c>/<c>colDmin</c> are the eight per-column d/dmin values
    /// (already-built inputs, loaded once per super-block outside this seam). Combines: scale
    /// term = <c>colScale * rowScaleBroadcast</c>, fmadd into the running <c>accRow</c>
    /// accumulator; mirror for <c>colDmin</c> into <c>accMinRow</c>; final stored value is
    /// <c>accRow - accMinRow</c> (C source line 3154, outside the per-block loop in the real
    /// kernel, included here as a convenience — a seam test can check either the accumulator
    /// state or the finished subtraction).
    /// </summary>
    internal static Vector256<float> ScaleAndAccumulateRow(
        Vector256<int> iaccRow, Vector256<int> iaccRowMin,
        Vector256<float> colScale, Vector256<float> colDmin, Vector256<float> rowScale, int tokenIndex,
        Vector256<float> accRow, Vector256<float> accMinRow,
        out Vector256<float> newAccRow, out Vector256<float> newAccMinRow)
    {
        // Shuffle immediate must be a compile-time constant for the JIT to emit vshufps, hence
        // the switch instead of passing the byte straight through.
        Vector256<float> rowScaleBroadcast = tokenIndex switch
        {
            0 => Avx.Shuffle(rowScale, rowScale, 0),
            1 => Avx.Shuffle(rowScale, rowScale, 85),
            2 => Avx.Shuffle(rowScale, rowScale, 170),
            3 => Avx.Shuffle(rowScale, rowScale, 255),
            _ => throw new ArgumentOutOfRangeException(nameof(tokenIndex)),
        };
        newAccRow = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iaccRow), Avx.Multiply(colScale, rowScaleBroadcast), accRow);
        newAccMinRow = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iaccRowMin), Avx.Multiply(colDmin, rowScaleBroadcast), accMinRow);
        return Avx.Subtract(newAccRow, newAccMinRow);
    }

    /// <summary>
    /// Composes seams 1-9 into the actual 4-token x 8-column GEMM, matching
    /// <see cref="RepackedGemm.GemmQ4K8x8x4Q8K_Avx2"/>'s signature and activation-buffer layout
    /// exactly (4 independent <c>SimdKernels.QuantizeRowToQ8K</c> scratch buffers — the
    /// lower-risk choice flagged in seam 4's doc entry, confirmed viable by the composition
    /// proof test since seams 4-6 only need a 16-byte token-pair built in-register, not
    /// llama.cpp's on-disk <c>block_q8_Kx4</c> interleave). Column order is natural (0-7)
    /// throughout — see <c>docs/real-avx2-gemm-port-plan.md</c>'s "Composition" section for the
    /// hand-derivation plus the empirical proof test that confirmed it
    /// (<c>ComposedSeams1Through8_ProduceNaturalColumnOrderPerToken</c>) before this function was
    /// written.
    ///
    /// One repacked super-block group is <see cref="RepackedGemm.Q4KGroupBytesPerBlock"/> bytes:
    /// 16B d[8] + 16B dmin[8] + 96B scales + 1024B qs. Each super-block has 4 "chunks" (C
    /// source's inner <c>sb</c> loop, <c>QK_K/64</c>) of 256 qs-bytes each; each chunk covers 2
    /// real Q4_K sub-blocks (sb0 = low nibble, sb1 = high nibble of the same bytes) and splits
    /// into 4 "kk" positions of 64 bytes (all 8 columns' interleaved bytes at that position —
    /// exactly seam 1's expected input). Activation-side, a chunk's matching 64 elements/token
    /// split into the same 4 kk positions, 8 bytes/token each, but sb0 and sb1 read from
    /// different 32-byte halves of the qs region (nibble-packing shares bytes; Q8_K activations
    /// don't).
    /// </summary>
    /// <summary>
    /// One (kk, sub-block, token-pair) activation build: loads 8 bytes/token at the given
    /// byte offset, packs them into the 16-byte pair seam 4 expects, and runs seams 4-5.
    /// A plain static method (not a local-function closure) so it allocates nothing.
    /// </summary>
    private static unsafe void BuildActLhsPair(
        sbyte* qA, sbyte* qB, int subByteOffset, int kk, out Vector256<sbyte> sp1, out Vector256<sbyte> sp2)
    {
        var lowHalf = Vector128.Create(
            Vector64.LoadUnsafe(ref *(qA + subByteOffset + kk * 8)).AsByte(),
            Vector64.LoadUnsafe(ref *(qB + subByteOffset + kk * 8)).AsByte());
        var vec = Vector256.Create(lowHalf, lowHalf);
        BroadcastLowHigh128(vec, out var bcast, out _);
        DuplicateShuffleActivationPattern(bcast, out var s1, out var s2);
        sp1 = s1.AsSByte();
        sp2 = s2.AsSByte();
    }

    internal static unsafe void GemmQ4K8x8x4Q8K_RealAvx2(
        float* out0, float* out1, float* out2, float* out3,
        byte* repackedGroups, byte* act0, byte* act1, byte* act2, byte* act3, int blocksPerRow)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;

        byte* utmp = stackalloc byte[128];
        uint* utmpWords = (uint*)utmp;

        byte*[] actPtrs = [act0, act1, act2, act3];
        float* dYs = stackalloc float[4];
        sbyte*[] q8Ptrs = new sbyte*[4];
        short*[] bsumsPtrs = new short*[4];

        var accRow = new Vector256<float>[4];
        var accMinRow = new Vector256<float>[4];
        for (int t = 0; t < 4; t++) { accRow[t] = Vector256<float>.Zero; accMinRow[t] = Vector256<float>.Zero; }

        float* colScaleArr = stackalloc float[8];
        float* colDminArr = stackalloc float[8];
        short* rawScalesBuf = stackalloc short[16];
        short* minsArr = stackalloc short[16];
        short* bsumsHsumArr = stackalloc short[16];

        for (int l = 0; l < blocksPerRow; l++)
        {
            byte* group = repackedGroups + l * RepackedGemm.Q4KGroupBytesPerBlock;
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

            for (int col = 0; col < 8; col++)
            {
                colScaleArr[col] = (float)BitConverter.UInt16BitsToHalf((ushort)(d16[col * 2] | (d16[col * 2 + 1] << 8)));
                colDminArr[col] = (float)BitConverter.UInt16BitsToHalf((ushort)(dmin16[col * 2] | (dmin16[col * 2 + 1] << 8)));
            }
            var colScale = Vector256.Create(colScaleArr[0], colScaleArr[1], colScaleArr[2], colScaleArr[3], colScaleArr[4], colScaleArr[5], colScaleArr[6], colScaleArr[7]);
            var colDmin = Vector256.Create(colDminArr[0], colDminArr[1], colDminArr[2], colDminArr[3], colDminArr[4], colDminArr[5], colDminArr[6], colDminArr[7]);

            for (int t = 0; t < 4; t++)
            {
                byte* act = actPtrs[t];
                float* actD = (float*)act;
                dYs[t] = actD[l];
                q8Ptrs[t] = (sbyte*)(act + blocksPerRow * 4) + l * 256;
                bsumsPtrs[t] = (short*)(act + blocksPerRow * 4 + blocksPerRow * 256) + l * 16;
            }
            var rowScale = Vector256.Create(dYs[0], dYs[1], dYs[2], dYs[3], dYs[0], dYs[1], dYs[2], dYs[3]);

            for (int chunk = 0; chunk < 4; chunk++)
            {
                int sb0 = 2 * chunk, sb1 = 2 * chunk + 1;
                byte* qsChunkBase = qs + chunk * 256;

                // Weight side: unrolled over kk=0..3 (named locals, not arrays -- this loop runs
                // blocksPerRow*4 times per GEMM call, so per-call heap allocations here would
                // dominate any register-reuse win the seams provide).
                RearrangeColumnPairs0145_2367(qsChunkBase, 0, out var m0145_0, out var m2367_0);
                RearrangeColumnPairs0145_2367(qsChunkBase, 1, out var m0145_1, out var m2367_1);
                RearrangeColumnPairs0145_2367(qsChunkBase, 2, out var m0145_2, out var m2367_2);
                RearrangeColumnPairs0145_2367(qsChunkBase, 3, out var m0145_3, out var m2367_3);
                UnpackNibbles(m0145_0, out var lo0145_0, out var hi0145_0); UnpackNibbles(m2367_0, out var lo2367_0, out var hi2367_0);
                UnpackNibbles(m0145_1, out var lo0145_1, out var hi0145_1); UnpackNibbles(m2367_1, out var lo2367_1, out var hi2367_1);
                UnpackNibbles(m0145_2, out var lo0145_2, out var hi0145_2); UnpackNibbles(m2367_2, out var lo2367_2, out var hi2367_2);
                UnpackNibbles(m0145_3, out var lo0145_3, out var hi0145_3); UnpackNibbles(m2367_3, out var lo2367_3, out var hi2367_3);

                DuplicateShuffleWeightPattern(lo0145_0, out var rLo0145Sp1_0, out var rLo0145Sp2_0);
                DuplicateShuffleWeightPattern(lo0145_1, out var rLo0145Sp1_1, out var rLo0145Sp2_1);
                DuplicateShuffleWeightPattern(lo0145_2, out var rLo0145Sp1_2, out var rLo0145Sp2_2);
                DuplicateShuffleWeightPattern(lo0145_3, out var rLo0145Sp1_3, out var rLo0145Sp2_3);
                DuplicateShuffleWeightPattern(hi0145_0, out var rHi0145Sp1_0, out var rHi0145Sp2_0);
                DuplicateShuffleWeightPattern(hi0145_1, out var rHi0145Sp1_1, out var rHi0145Sp2_1);
                DuplicateShuffleWeightPattern(hi0145_2, out var rHi0145Sp1_2, out var rHi0145Sp2_2);
                DuplicateShuffleWeightPattern(hi0145_3, out var rHi0145Sp1_3, out var rHi0145Sp2_3);
                DuplicateShuffleWeightPattern(lo2367_0, out var rLo2367Sp1_0, out var rLo2367Sp2_0);
                DuplicateShuffleWeightPattern(lo2367_1, out var rLo2367Sp1_1, out var rLo2367Sp2_1);
                DuplicateShuffleWeightPattern(lo2367_2, out var rLo2367Sp1_2, out var rLo2367Sp2_2);
                DuplicateShuffleWeightPattern(lo2367_3, out var rLo2367Sp1_3, out var rLo2367Sp2_3);
                DuplicateShuffleWeightPattern(hi2367_0, out var rHi2367Sp1_0, out var rHi2367Sp2_0);
                DuplicateShuffleWeightPattern(hi2367_1, out var rHi2367Sp1_1, out var rHi2367Sp2_1);
                DuplicateShuffleWeightPattern(hi2367_2, out var rHi2367Sp1_2, out var rHi2367Sp2_2);
                DuplicateShuffleWeightPattern(hi2367_3, out var rHi2367Sp1_3, out var rHi2367Sp2_3);

                // Activation: same unrolled shape, one call per (token-pair, sub-block, kk).
                sbyte* q0 = q8Ptrs[0] + chunk * 64; sbyte* q1 = q8Ptrs[1] + chunk * 64;
                sbyte* q2 = q8Ptrs[2] + chunk * 64; sbyte* q3 = q8Ptrs[3] + chunk * 64;

                BuildActLhsPair(q0, q1, 0, 0, out var lSb0P01Sp1_0, out var lSb0P01Sp2_0);
                BuildActLhsPair(q0, q1, 0, 1, out var lSb0P01Sp1_1, out var lSb0P01Sp2_1);
                BuildActLhsPair(q0, q1, 0, 2, out var lSb0P01Sp1_2, out var lSb0P01Sp2_2);
                BuildActLhsPair(q0, q1, 0, 3, out var lSb0P01Sp1_3, out var lSb0P01Sp2_3);
                BuildActLhsPair(q0, q1, 32, 0, out var lSb1P01Sp1_0, out var lSb1P01Sp2_0);
                BuildActLhsPair(q0, q1, 32, 1, out var lSb1P01Sp1_1, out var lSb1P01Sp2_1);
                BuildActLhsPair(q0, q1, 32, 2, out var lSb1P01Sp1_2, out var lSb1P01Sp2_2);
                BuildActLhsPair(q0, q1, 32, 3, out var lSb1P01Sp1_3, out var lSb1P01Sp2_3);
                BuildActLhsPair(q2, q3, 0, 0, out var lSb0P23Sp1_0, out var lSb0P23Sp2_0);
                BuildActLhsPair(q2, q3, 0, 1, out var lSb0P23Sp1_1, out var lSb0P23Sp2_1);
                BuildActLhsPair(q2, q3, 0, 2, out var lSb0P23Sp1_2, out var lSb0P23Sp2_2);
                BuildActLhsPair(q2, q3, 0, 3, out var lSb0P23Sp1_3, out var lSb0P23Sp2_3);
                BuildActLhsPair(q2, q3, 32, 0, out var lSb1P23Sp1_0, out var lSb1P23Sp2_0);
                BuildActLhsPair(q2, q3, 32, 1, out var lSb1P23Sp1_1, out var lSb1P23Sp2_1);
                BuildActLhsPair(q2, q3, 32, 2, out var lSb1P23Sp1_2, out var lSb1P23Sp2_2);
                BuildActLhsPair(q2, q3, 32, 3, out var lSb1P23Sp1_3, out var lSb1P23Sp2_3);

                // rawScales: all 8 natural-order columns, each value duplicated once (matches
                // seam 7's expected input layout, derived by hand + confirmed by the composition
                // proof test) -- built once per (chunk, sub-block), shared by both token pairs.
                for (int col = 0; col < 8; col++)
                {
                    short v = utmp[sb0 * 16 + col];
                    rawScalesBuf[col * 2] = v; rawScalesBuf[col * 2 + 1] = v;
                }
                var rawScalesSb0 = Vector256.LoadUnsafe(ref *rawScalesBuf);
                for (int col = 0; col < 8; col++)
                {
                    short v = utmp[sb1 * 16 + col];
                    rawScalesBuf[col * 2] = v; rawScalesBuf[col * 2 + 1] = v;
                }
                var rawScalesSb1 = Vector256.LoadUnsafe(ref *rawScalesBuf);

                var mat00 = ScaleAndReduce0145(Avx2.Add(
                    MaddubsAccumulate4(rLo0145Sp1_0, lSb0P01Sp1_0, rLo0145Sp1_1, lSb0P01Sp1_1, rLo0145Sp1_2, lSb0P01Sp1_2, rLo0145Sp1_3, lSb0P01Sp1_3),
                    MaddubsAccumulate4(rLo0145Sp2_0, lSb0P01Sp2_0, rLo0145Sp2_1, lSb0P01Sp2_1, rLo0145Sp2_2, lSb0P01Sp2_2, rLo0145Sp2_3, lSb0P01Sp2_3)), rawScalesSb0);
                var mat01 = ScaleAndReduce2367(Avx2.Add(
                    MaddubsAccumulate4(rLo2367Sp1_0, lSb0P01Sp1_0, rLo2367Sp1_1, lSb0P01Sp1_1, rLo2367Sp1_2, lSb0P01Sp1_2, rLo2367Sp1_3, lSb0P01Sp1_3),
                    MaddubsAccumulate4(rLo2367Sp2_0, lSb0P01Sp2_0, rLo2367Sp2_1, lSb0P01Sp2_1, rLo2367Sp2_2, lSb0P01Sp2_2, rLo2367Sp2_3, lSb0P01Sp2_3)), rawScalesSb0);
                var mat10 = ScaleAndReduce0145(Avx2.Add(
                    MaddubsAccumulate4(rLo0145Sp1_0, lSb0P23Sp1_0, rLo0145Sp1_1, lSb0P23Sp1_1, rLo0145Sp1_2, lSb0P23Sp1_2, rLo0145Sp1_3, lSb0P23Sp1_3),
                    MaddubsAccumulate4(rLo0145Sp2_0, lSb0P23Sp2_0, rLo0145Sp2_1, lSb0P23Sp2_1, rLo0145Sp2_2, lSb0P23Sp2_2, rLo0145Sp2_3, lSb0P23Sp2_3)), rawScalesSb0);
                var mat11 = ScaleAndReduce2367(Avx2.Add(
                    MaddubsAccumulate4(rLo2367Sp1_0, lSb0P23Sp1_0, rLo2367Sp1_1, lSb0P23Sp1_1, rLo2367Sp1_2, lSb0P23Sp1_2, rLo2367Sp1_3, lSb0P23Sp1_3),
                    MaddubsAccumulate4(rLo2367Sp2_0, lSb0P23Sp2_0, rLo2367Sp2_1, lSb0P23Sp2_1, rLo2367Sp2_2, lSb0P23Sp2_2, rLo2367Sp2_3, lSb0P23Sp2_3)), rawScalesSb0);
                StraightenToRowVectors(mat00, mat01, mat10, mat11, out var rowSb0_0, out var rowSb0_1, out var rowSb0_2, out var rowSb0_3);

                var mat00h = ScaleAndReduce0145(Avx2.Add(
                    MaddubsAccumulate4(rHi0145Sp1_0, lSb1P01Sp1_0, rHi0145Sp1_1, lSb1P01Sp1_1, rHi0145Sp1_2, lSb1P01Sp1_2, rHi0145Sp1_3, lSb1P01Sp1_3),
                    MaddubsAccumulate4(rHi0145Sp2_0, lSb1P01Sp2_0, rHi0145Sp2_1, lSb1P01Sp2_1, rHi0145Sp2_2, lSb1P01Sp2_2, rHi0145Sp2_3, lSb1P01Sp2_3)), rawScalesSb1);
                var mat01h = ScaleAndReduce2367(Avx2.Add(
                    MaddubsAccumulate4(rHi2367Sp1_0, lSb1P01Sp1_0, rHi2367Sp1_1, lSb1P01Sp1_1, rHi2367Sp1_2, lSb1P01Sp1_2, rHi2367Sp1_3, lSb1P01Sp1_3),
                    MaddubsAccumulate4(rHi2367Sp2_0, lSb1P01Sp2_0, rHi2367Sp2_1, lSb1P01Sp2_1, rHi2367Sp2_2, lSb1P01Sp2_2, rHi2367Sp2_3, lSb1P01Sp2_3)), rawScalesSb1);
                var mat10h = ScaleAndReduce0145(Avx2.Add(
                    MaddubsAccumulate4(rHi0145Sp1_0, lSb1P23Sp1_0, rHi0145Sp1_1, lSb1P23Sp1_1, rHi0145Sp1_2, lSb1P23Sp1_2, rHi0145Sp1_3, lSb1P23Sp1_3),
                    MaddubsAccumulate4(rHi0145Sp2_0, lSb1P23Sp2_0, rHi0145Sp2_1, lSb1P23Sp2_1, rHi0145Sp2_2, lSb1P23Sp2_2, rHi0145Sp2_3, lSb1P23Sp2_3)), rawScalesSb1);
                var mat11h = ScaleAndReduce2367(Avx2.Add(
                    MaddubsAccumulate4(rHi2367Sp1_0, lSb1P23Sp1_0, rHi2367Sp1_1, lSb1P23Sp1_1, rHi2367Sp1_2, lSb1P23Sp1_2, rHi2367Sp1_3, lSb1P23Sp1_3),
                    MaddubsAccumulate4(rHi2367Sp2_0, lSb1P23Sp2_0, rHi2367Sp2_1, lSb1P23Sp2_1, rHi2367Sp2_2, lSb1P23Sp2_2, rHi2367Sp2_3, lSb1P23Sp2_3)), rawScalesSb1);
                StraightenToRowVectors(mat00h, mat01h, mat10h, mat11h, out var rowSb1_0, out var rowSb1_1, out var rowSb1_2, out var rowSb1_3);

                var iaccRow0 = Avx2.Add(rowSb0_0, rowSb1_0);
                var iaccRow1 = Avx2.Add(rowSb0_1, rowSb1_1);
                var iaccRow2 = Avx2.Add(rowSb0_2, rowSb1_2);
                var iaccRow3 = Avx2.Add(rowSb0_3, rowSb1_3);

                // mins01: natural column order, sb0/sb1 interleaved (derived by hand -- see
                // docs/real-avx2-gemm-port-plan.md's composition section).
                for (int col = 0; col < 8; col++)
                {
                    minsArr[col * 2] = utmp[sb0 * 16 + 8 + col];
                    minsArr[col * 2 + 1] = utmp[sb1 * 16 + 8 + col];
                }

                // bsumsHsum: dword t = (hsum_sb0(t), hsum_sb1(t)), repeated in both 128-bit
                // halves so BsumsMinCorrection's per-lane broadcast picks up the right token
                // regardless of which 128-bit lane it reads from.
                for (int t = 0; t < 4; t++)
                {
                    short* b = bsumsPtrs[t];
                    short hsum0 = (short)(b[4 * chunk + 0] + b[4 * chunk + 1]);
                    short hsum1 = (short)(b[4 * chunk + 2] + b[4 * chunk + 3]);
                    bsumsHsumArr[t * 2] = hsum0; bsumsHsumArr[t * 2 + 1] = hsum1;
                    bsumsHsumArr[8 + t * 2] = hsum0; bsumsHsumArr[8 + t * 2 + 1] = hsum1;
                }

                {
                    var mins01 = Vector256.LoadUnsafe(ref *minsArr);
                    var bsumsHsum = Vector256.LoadUnsafe(ref *bsumsHsumArr);

                    var iaccRowMin0 = BsumsMinCorrection(bsumsHsum, 0, mins01);
                    var iaccRowMin1 = BsumsMinCorrection(bsumsHsum, 1, mins01);
                    var iaccRowMin2 = BsumsMinCorrection(bsumsHsum, 2, mins01);
                    var iaccRowMin3 = BsumsMinCorrection(bsumsHsum, 3, mins01);

                    ScaleAndAccumulateRow(iaccRow0, iaccRowMin0, colScale, colDmin, rowScale, 0, accRow[0], accMinRow[0], out accRow[0], out accMinRow[0]);
                    ScaleAndAccumulateRow(iaccRow1, iaccRowMin1, colScale, colDmin, rowScale, 1, accRow[1], accMinRow[1], out accRow[1], out accMinRow[1]);
                    ScaleAndAccumulateRow(iaccRow2, iaccRowMin2, colScale, colDmin, rowScale, 2, accRow[2], accMinRow[2], out accRow[2], out accMinRow[2]);
                    ScaleAndAccumulateRow(iaccRow3, iaccRowMin3, colScale, colDmin, rowScale, 3, accRow[3], accMinRow[3], out accRow[3], out accMinRow[3]);
                }
            }
        }

        Span<float> final0 = stackalloc float[8], final1 = stackalloc float[8], final2 = stackalloc float[8], final3 = stackalloc float[8];
        Vector256.StoreUnsafe(Avx.Subtract(accRow[0], accMinRow[0]), ref final0[0]);
        Vector256.StoreUnsafe(Avx.Subtract(accRow[1], accMinRow[1]), ref final1[0]);
        Vector256.StoreUnsafe(Avx.Subtract(accRow[2], accMinRow[2]), ref final2[0]);
        Vector256.StoreUnsafe(Avx.Subtract(accRow[3], accMinRow[3]), ref final3[0]);
        for (int col = 0; col < 8; col++)
        {
            out0[col] = final0[col]; out1[col] = final1[col]; out2[col] = final2[col]; out3[col] = final3[col];
        }
    }
}
