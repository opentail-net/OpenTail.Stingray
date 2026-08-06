using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// <b>Path 2</b> — a deliberately literal C# port of llama.cpp's AVX2
/// <c>ggml_gemm_q4_K_8x8_q8_K</c> (ggml/src/ggml-cpu/arch/x86/repack.cpp, lines 2816–3487).
/// </summary>
/// <remarks>
/// <para><b>Porting rule.</b> Structure, variable names and intrinsic order follow the original as
/// closely as C# permits, so the two can be diffed side by side. Where a deviation is forced (or is
/// a strict improvement made possible by our repack), it is marked <c>DEVIATION:</c> inline and
/// recorded in docs/repack-gemm/port-log.md with its rationale.</para>
///
/// <para><b>Shape.</b> The original processes 16 token rows x 8 weight columns per pass:
/// <c>a_ptrs[4]</c>, each a <c>block_q8_Kx4</c> holding 4 interleaved token rows, against one
/// <c>block_q4_Kx8</c> column group, accumulating into <c>acc_rows[16]</c> plus
/// <c>acc_min_rows[16]</c> for the fused Q4_K min correction.</para>
///
/// <para><b>Incomplete by design.</b> Every entry point may return <c>false</c> for shapes not yet
/// implemented; <see cref="SimdKernels.TryMatMulBatchedQ4Kx8"/> then falls through to Path 1. An
/// unfinished Path 2 therefore costs performance, never correctness.</para>
/// </remarks>
public static unsafe class RepackedGemmPath2
{
    private static int _engagedCalls;

    /// <summary>
    /// Number of times the ported GEMM has actually run. Exists because a null A/B result is
    /// ambiguous between "no speed difference" and "never executed" — and during this port it was
    /// the latter (see docs/repack-gemm/port-log.md, "the repacked path is OFF BY DEFAULT").
    /// Read it to prove engagement rather than infer it from configuration.
    /// </summary>
    public static int EngagedCalls => Volatile.Read(ref _engagedCalls);

    /// <summary>Token rows carried by one <c>block_q8_Kx4</c>.</summary>
    public const int Q8Kx4Rows = 4;

    /// <summary>Weight columns carried by one <c>block_q4_Kx8</c> group.</summary>
    public const int NbCols = 8;

    /// <summary>Interleave granularity of both repacked layouts, in bytes.</summary>
    public const int BlockLen = 8;

    /// <summary>Elements per super-block (<c>QK_K</c>).</summary>
    public const int QK_K = 256;

    /// <summary>
    /// Bytes per <c>block_q8_Kx4</c> — one super-block of 4 interleaved token rows.
    /// Mirrors repack.h:96 exactly: <c>float d[4]</c>, <c>int8_t qs[QK_K*4]</c>,
    /// <c>int16_t bsums[QK_K/4]</c>.
    /// </summary>
    public const int Q8Kx4BlockBytes = 4 * sizeof(float) + QK_K * 4 + (QK_K / 4) * sizeof(short); // 1168

    /// <summary>Bytes to hold <paramref name="cols"/> elements of 4 interleaved token rows.</summary>
    public static long Q8Kx4Bytes(int cols) => (long)(cols / QK_K) * Q8Kx4BlockBytes;

    /// <summary>
    /// Port of llama.cpp's <c>nearest_int</c> (repack.cpp:27) — the add-magic-constant rounding
    /// trick, kept bit-exact rather than swapped for <c>MathF.Round</c> so Path 2's quantisation
    /// matches the reference exactly. Round-half-to-even, valid for |fval| &lt;= 4194303.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int NearestInt(float fval)
    {
        float val = fval + 12582912.0f;
        int i = BitConverter.SingleToInt32Bits(val);
        return (i & 0x007fffff) - 0x00400000;
    }

    /// <summary>
    /// Port of <c>ggml_quantize_mat_q8_K_4x8_generic</c> (repack.cpp:262): quantise 4 token rows
    /// into one <c>block_q8_Kx4</c> stream, interleaved 8 bytes at a time.
    /// </summary>
    /// <remarks>
    /// <para>The scalar reference is ported first, deliberately: it is unambiguous and becomes the
    /// oracle for the vectorised variant (arch/x86/repack.cpp:290) later. Variable names follow the
    /// original — <c>nb</c>, <c>srcv</c>, <c>iscale</c>, <c>src_offset</c>, <c>src_id</c>,
    /// <c>index</c>, <c>x0</c>.</para>
    /// <para>Note <c>iscale = -127/max</c> uses the <b>signed</b> value at the position of maximum
    /// magnitude, not <c>amax</c>, so <c>iscale</c> is positive when that value is negative. This
    /// is faithful to the original and is not a typo.</para>
    /// </remarks>
    /// <param name="x">4 consecutive token rows, row-major, each <paramref name="k"/> long.</param>
    /// <param name="vy">Destination, <see cref="Q8Kx4Bytes"/>(k) bytes.</param>
    /// <param name="k">Elements per row; must be a multiple of <see cref="QK_K"/>.</param>
    public static void QuantizeMatQ8Kx4(float* x, byte* vy, long k)
    {
        int nb = (int)(k / QK_K);
        const int blck_size_interleave = 8;

        float* srcv = stackalloc float[4 * QK_K];
        float* iscale = stackalloc float[4];

        for (int i = 0; i < nb; i++)
        {
            byte* blk = vy + (long)i * Q8Kx4BlockBytes;
            float* d = (float*)blk;
            sbyte* qs = (sbyte*)(blk + 16);
            short* bsums = (short*)(blk + 16 + QK_K * 4);

            for (int row_iter = 0; row_iter < 4; row_iter++)
            {
                float amax = 0.0f;
                float max = 0;

                for (int j = 0; j < QK_K; j++)
                {
                    float v = x[(long)row_iter * k + (long)i * QK_K + j];
                    srcv[row_iter * QK_K + j] = v;
                    if (amax < MathF.Abs(v)) { amax = MathF.Abs(v); max = v; }
                }

                iscale[row_iter] = amax != 0 ? -127.0f / max : 0;
                d[row_iter] = amax != 0 ? 1 / iscale[row_iter] : 0;
            }

            for (int j = 0; j < QK_K / 4; j++) bsums[j] = 0;

            // Quants are interleaved in runs of eight bytes from the four source rows; bsums are
            // interleaved four at a time from each row (original comment, repack.cpp:297-299).
            for (int j = 0; j < QK_K * 4; j++)
            {
                int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
                int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
                src_offset += j % blck_size_interleave;
                int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

                float x0 = srcv[src_id * QK_K + src_offset] * iscale[src_id];
                qs[j] = (sbyte)NearestInt(x0);
                bsums[index] += qs[j];
            }
        }
    }

    /// <summary>
    /// AVX2 port of <c>ggml_quantize_mat_q8_K_4x8</c> (arch/x86/repack.cpp:297–506). Same output as
    /// <see cref="QuantizeMatQ8Kx4"/>; that scalar version remains the oracle in tests.
    /// </summary>
    /// <remarks>
    /// <para><b>DEVIATION 5 — sign recovery.</b> The original finds the <i>signed</i> value at the
    /// maximum-magnitude position with an accumulated compare-mask across sub-blocks
    /// (<c>maskAbs</c>/<c>mask_prev</c>/<c>mask_next</c>, lines 321–386). Path 2 tracks running
    /// <c>max</c> and <c>min</c> instead: the largest-magnitude element is <c>max</c> when
    /// <c>max &gt;= -min</c> and <c>min</c> otherwise. Fewer operations and much easier to verify.
    /// <br/>Tie case: when <c>max == -min</c> the two disagree about which element "wins" — this
    /// picks the positive one, the scalar version picks whichever occurs first. Both are valid
    /// quantisations of the same data and differ only in the sign of <c>d</c> together with the
    /// sign of every quant, so the reconstructed values are identical. It is measure-zero on real
    /// activations regardless.</para>
    ///
    /// <para><b>DEVIATION 6 — bsums.</b> The original computes them with a shuffle/blend dance over
    /// three hand-built masks (lines 427–506, ~70 lines). Path 2 derives them directly from the
    /// stored layout: for output group <c>g</c>, row <c>r</c> occupies bytes <c>g*32 + r*8 .. +8</c>,
    /// and <c>bsums[r*4 + (g/8)*16 + ((g%8)/2)]</c> accumulates two adjacent groups. A
    /// <c>maddubs</c>+<c>madd</c> pair reduces 32 bytes to four row sums. Same values, a fraction of
    /// the code.</para>
    /// </remarks>
    public static void QuantizeMatQ8Kx4Avx2(float* x, byte* vy, long k)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) { QuantizeMatQ8Kx4(x, vy, k); return; }

        int nb = (int)(k / QK_K);
        Vector256<float>* srcv = stackalloc Vector256<float>[4 * 32];
        float* iscale = stackalloc float[4];
        var one8 = Vector256.Create((byte)1);
        var one16 = Vector256.Create((short)1);
        var perm = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);

        for (int i = 0; i < nb; i++)
        {
            byte* blk = vy + (long)i * Q8Kx4BlockBytes;
            float* d = (float*)blk;
            sbyte* qs = (sbyte*)(blk + 16);
            short* bsums = (short*)(blk + 16 + QK_K * 4);

            for (int row_iter = 0; row_iter < 4; row_iter++)
            {
                var vmx = Vector256.Create(float.NegativeInfinity);
                var vmn = Vector256.Create(float.PositiveInfinity);
                float* src = x + (long)row_iter * k + (long)i * QK_K;
                for (int j = 0; j < 32; j++)
                {
                    var v = Vector256.LoadUnsafe(ref *(src + j * 8));
                    srcv[row_iter * 32 + j] = v;
                    vmx = Avx.Max(vmx, v);
                    vmn = Avx.Min(vmn, v);
                }
                float mx = HMax(vmx), mn = HMin(vmn);
                float amax = MathF.Max(mx, -mn);
                float max = mx >= -mn ? mx : mn;      // DEVIATION 5
                iscale[row_iter] = amax != 0 ? -127.0f / max : 0;
                d[row_iter] = amax != 0 ? 1 / iscale[row_iter] : 0;
            }

            for (int j = 0; j < QK_K / 4; j++) bsums[j] = 0;

            var s0 = Vector256.Create(iscale[0]);
            var s1 = Vector256.Create(iscale[1]);
            var s2 = Vector256.Create(iscale[2]);
            var s3 = Vector256.Create(iscale[3]);

            for (int j = 0; j < 32; j++)
            {
                var v0 = Avx.Multiply(srcv[0 * 32 + j], s0);
                var v1 = Avx.Multiply(srcv[1 * 32 + j], s1);
                var v2 = Avx.Multiply(srcv[2 * 32 + j], s2);
                var v3 = Avx.Multiply(srcv[3 * 32 + j], s3);

                v0 = Avx.RoundToNearestInteger(v0);
                v1 = Avx.RoundToNearestInteger(v1);
                v2 = Avx.RoundToNearestInteger(v2);
                v3 = Avx.RoundToNearestInteger(v3);

                var i0 = Avx.ConvertToVector256Int32(v0);
                var i1 = Avx.ConvertToVector256Int32(v1);
                var i2 = Avx.ConvertToVector256Int32(v2);
                var i3 = Avx.ConvertToVector256Int32(v3);

                var p0 = Avx2.PackSignedSaturate(i0, i1);          // r0e0-3 r1e0-3 | r0e4-7 r1e4-7
                var p2 = Avx2.PackSignedSaturate(i2, i3);
                var q = Avx2.PackSignedSaturate(p0, p2);           // dwords r0,r1,r2,r3 | r0,r1,r2,r3
                q = Avx2.PermuteVar8x32(q.AsInt32(), perm).AsSByte();  // -> r0[8] r1[8] r2[8] r3[8]

                q.StoreUnsafe(ref *(qs + 32 * j));

                // DEVIATION 6: four row sums straight out of the stored group.
                var p16 = Avx2.MultiplyAddAdjacent(one8, q);        // pairs
                var p32 = Avx2.MultiplyAddAdjacent(p16, one16);     // quads: lanes 2r, 2r+1 are row r
                int bIdx = (j / 8) * 16 + ((j % 8) / 2);
                for (int r = 0; r < 4; r++)
                    bsums[r * 4 + bIdx] += (short)(p32.GetElement(2 * r) + p32.GetElement(2 * r + 1));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HMax(Vector256<float> v)
    {
        var h = Sse.Max(v.GetLower(), v.GetUpper());
        h = Sse.Max(h, Sse.MoveHighToLow(h, h));
        h = Sse.MaxScalar(h, Sse3.MoveHighAndDuplicate(h));
        return h.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HMin(Vector256<float> v)
    {
        var h = Sse.Min(v.GetLower(), v.GetUpper());
        h = Sse.Min(h, Sse.MoveHighToLow(h, h));
        h = Sse.MinScalar(h, Sse3.MoveHighAndDuplicate(h));
        return h.ToScalar();
    }

    /// <summary>
    /// Batched prefill matmul over a repacked Q4_K matrix using the ported GEMM.
    /// Returns <c>false</c> when the shape is not yet supported, so the caller can use Path 1.
    /// </summary>
    /// <param name="output">Destination, <c>output[token * rows + row]</c>.</param>
    /// <param name="packed">Weights already in <c>block_q4_Kx8</c> form (see
    /// <see cref="SimdKernels.RepackQ4KMatrix"/>).</param>
    /// <param name="input">Row-major F32 activations, <c>batchSize x cols</c>.</param>
    public static bool TryMatMulBatched(float* output, byte* packed, float* input,
        int batchSize, int rows, int cols)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return false;
        if (rows % NbCols != 0 || cols % 256 != 0 || batchSize < 1) return false;

        // Increment 8b: EVERY batch size is handled. The M<=3 remainder is zero-padded up to a
        // full block_q8_Kx4 rather than declined.
        //
        // Declining would have been easier, but it reintroduces the exact defect ForwardPass.cs:813
        // warns about: Path 1 and Path 2 use different activation quantisations (Q8_KS vs Q8_K), so
        // a prompt whose chunks straddle the threshold would have some positions computed one way
        // and some the other, making chunked and unchunked prefill of the same prompt disagree.
        // Padding keeps the whole prompt on ONE scheme. Cost: at most 3 wasted token rows.
        //
        // Ported GEMV (repack.cpp:1464) is therefore NOT needed for correctness — only as a
        // possible optimisation for the M<=3 case, where padding wastes up to 75% of a pass.

        Interlocked.Increment(ref _engagedCalls);

        int nb = cols / QK_K;
        long q8Stride = Q8Kx4Bytes(cols);                 // bytes per block_q8_Kx4 row-group
        long bytesPerGroup = (long)nb * SimdKernels.Q4Kx8BlockBytes;

        int fullGroups = batchSize / Q8Kx4Rows;
        int rem = batchSize % Q8Kx4Rows;
        int rowGroups = fullGroups + (rem > 0 ? 1 : 0);

        byte* aBase = (byte*)NativeMemory.Alloc((nuint)(q8Stride * rowGroups));
        try
        {
            Parallel.For(0, fullGroups, rg =>
                QuantizeMatQ8Kx4Avx2(input + (long)rg * Q8Kx4Rows * cols, aBase + (long)rg * q8Stride, cols));

            if (rem > 0)
            {
                // Zero-pad the final partial group. Zero rows quantise to d=0 (amax==0), so they
                // contribute exactly 0 and cannot perturb the real rows.
                long padBytes = (long)sizeof(float) * Q8Kx4Rows * cols;
                float* pad = (float*)NativeMemory.AllocZeroed((nuint)padBytes);
                try
                {
                    Buffer.MemoryCopy(input + (long)fullGroups * Q8Kx4Rows * cols, pad,
                                      padBytes, (long)sizeof(float) * rem * cols);
                    QuantizeMatQ8Kx4Avx2(pad, aBase + (long)fullGroups * q8Stride, cols);
                }
                finally { NativeMemory.Free(pad); }
            }

            Parallel.For(0, rows / NbCols, x =>
                GemmQ4Kx8Q8Kx4(cols, output, rows, packed + (long)x * bytesPerGroup,
                               aBase, q8Stride, rowGroups * Q8Kx4Rows, batchSize, x));
        }
        finally
        {
            NativeMemory.Free(aBase);
        }
        return true;
    }

    /// <summary>
    /// Port of the AVX2 arm of <c>ggml_gemm_q4_K_8x8_q8_K</c> (repack.cpp:2818–3156) — the
    /// 16-token-row x 8-column main loop.
    /// </summary>
    /// <remarks>
    /// <para>Original variable names are preserved throughout (<c>rhs_raw_mat_0123_0</c>,
    /// <c>lhs_mat_01_00_sp1</c>, <c>iacc_mat_00_0_sp1</c>, <c>iacc_row_0</c>, …) so the two can be
    /// diffed side by side.</para>
    ///
    /// <para><b>DEVIATION 1 — scales.</b> The original unpacks Q4_K's 6-bit scales/mins in the hot
    /// loop via <c>utmp_0/utmp_1</c> and <c>kmask1..3</c> (repack.cpp:2953–2981). Our
    /// <c>RepackQ4K8Rows</c> already decodes them at pack time into <c>sc[64]</c>/<c>mn[64]</c>,
    /// sub-block-major. The whole <c>utmp</c> block is therefore replaced by two 8-byte loads that
    /// reproduce the identical lane layout: <c>scales_0</c> = duplicate-and-widen of
    /// <c>sc[(2sb)*8..]</c>, <c>mins_01</c> = byte-interleave of <c>mn[(2sb)*8..]</c> and
    /// <c>mn[(2sb+1)*8..]</c>.</para>
    ///
    /// <para><b>DEVIATION 2 — d/dmin.</b> Loaded as 8 floats rather than via
    /// <c>GGML_F32Cx8_LOAD</c> (<c>_mm256_cvtph_ps</c>), which has no .NET equivalent. Our repack
    /// stores them already converted. Strictly cheaper than the original.</para>
    ///
    /// <para><b>Verified identical:</b> the <c>qs</c> interleave. Ours is
    /// <c>qs[(g*8+r)*8]</c> = row r at source offset <c>g*8</c>, which is exactly the original's
    /// <c>rhs_raw_mat_0123_0</c> = rows 0–3 / <c>_4567_0</c> = rows 4–7 grouping.</para>
    /// </remarks>
    /// <param name="nr">Token rows to compute, including zero padding (always a multiple of 4).</param>
    /// <param name="validRows">Token rows to actually store; padded rows above this are computed
    /// and discarded, which is cheaper than branching inside the inner loop.</param>
    private static void GemmQ4Kx8Q8Kx4(int n, float* s, int bs, byte* b_ptr_start,
                                       byte* a_ptr_start, long q8Stride, int nr, int validRows, int x)
    {
        int nb = n / QK_K;

        // Mask to mask out nibbles from packed bytes
        var m4b = Vector256.Create((byte)0x0F);
        // Permute mask used for easier vector processing at later stages
        // _mm256_set_epi32(3,2,1,0,7,6,5,4) is low-to-high lanes 4,5,6,7,0,1,2,3.
        var requiredOrder = Vector256.Create(4, 5, 6, 7, 0, 1, 2, 3);

        int nGroups = nr / 4;   // block_q8_Kx4 groups; each carries 4 token rows

        Vector256<float>* acc_rows = stackalloc Vector256<float>[16];
        Vector256<float>* acc_min_rows = stackalloc Vector256<float>[16];
        byte** a_ptrs = stackalloc byte*[4];

        // DEVIATION 4: the original writes two near-identical ~320-line bodies — a main loop taking
        // four block_q8_Kx4 groups (acc_rows[16], repack.cpp:2818) and a tail taking one
        // (acc_rows[4], repack.cpp:3158). Here one body carries `nrp` groups, 1..4. Semantics are
        // identical; only the duplication is removed. This is the single place where staying
        // literal would have meant copying 320 lines to change two constants.
        for (int y = 0; y < nGroups; )
        {
            int nrp = Math.Min(4, nGroups - y);

            a_ptrs[0] = a_ptr_start + (long)y * q8Stride;
            for (int i = 0; i < nrp - 1; ++i) a_ptrs[i + 1] = a_ptrs[i] + q8Stride;

            // Master FP accumulators
            for (int i = 0; i < nrp * 4; i++) acc_rows[i] = Vector256<float>.Zero;
            for (int i = 0; i < nrp * 4; i++) acc_min_rows[i] = Vector256<float>.Zero;

            // For super block
            for (int b = 0; b < nb; b++)
            {
                byte* bblk = b_ptr_start + (long)b * SimdKernels.Q4Kx8BlockBytes;
                float* bd = (float*)bblk;                 // d[8]
                float* bdmin = (float*)(bblk + 32);       // dmin[8]
                byte* bsc = bblk + 64;                    // sc[64]  sub-block-major
                byte* bmn = bblk + 128;                   // mn[64]
                byte* bqs = bblk + 192;                   // qs[1024]

                // DEVIATION 2: plain float loads, see remarks.
                var col_scale_f32 = Vector256.LoadUnsafe(ref *bd);
                var col_dmin_f32 = Vector256.LoadUnsafe(ref *bdmin);

                // Loop to iterate over the eight sub blocks of a super block - two per iteration
                for (int sb = 0; sb < QK_K / 64; sb++)
                {
                    byte* q = bqs + sb * 256;
                    var rhs_raw_mat_0123_0 = Vector256.LoadUnsafe(ref *(q + 0));
                    var rhs_raw_mat_4567_0 = Vector256.LoadUnsafe(ref *(q + 32));
                    var rhs_raw_mat_0123_1 = Vector256.LoadUnsafe(ref *(q + 64));
                    var rhs_raw_mat_4567_1 = Vector256.LoadUnsafe(ref *(q + 96));
                    var rhs_raw_mat_0123_2 = Vector256.LoadUnsafe(ref *(q + 128));
                    var rhs_raw_mat_4567_2 = Vector256.LoadUnsafe(ref *(q + 160));
                    var rhs_raw_mat_0123_3 = Vector256.LoadUnsafe(ref *(q + 192));
                    var rhs_raw_mat_4567_3 = Vector256.LoadUnsafe(ref *(q + 224));

                    // Save the values in the formats B0B1B4B5, B2B3B6B7
                    var rhs_raw_mat_0145_0 = Blend240(rhs_raw_mat_0123_0, Perm8x32(rhs_raw_mat_4567_0, requiredOrder));
                    var rhs_raw_mat_2367_0 = Blend240(Perm8x32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0);
                    var rhs_raw_mat_0145_1 = Blend240(rhs_raw_mat_0123_1, Perm8x32(rhs_raw_mat_4567_1, requiredOrder));
                    var rhs_raw_mat_2367_1 = Blend240(Perm8x32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1);
                    var rhs_raw_mat_0145_2 = Blend240(rhs_raw_mat_0123_2, Perm8x32(rhs_raw_mat_4567_2, requiredOrder));
                    var rhs_raw_mat_2367_2 = Blend240(Perm8x32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2);
                    var rhs_raw_mat_0145_3 = Blend240(rhs_raw_mat_0123_3, Perm8x32(rhs_raw_mat_4567_3, requiredOrder));
                    var rhs_raw_mat_2367_3 = Blend240(Perm8x32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3);

                    // 4-bit -> 8-bit, first sub block of the two
                    var rhs_mat_0145_00 = Avx2.And(rhs_raw_mat_0145_0, m4b);
                    var rhs_mat_2367_00 = Avx2.And(rhs_raw_mat_2367_0, m4b);
                    var rhs_mat_0145_01 = Avx2.And(rhs_raw_mat_0145_1, m4b);
                    var rhs_mat_2367_01 = Avx2.And(rhs_raw_mat_2367_1, m4b);
                    var rhs_mat_0145_02 = Avx2.And(rhs_raw_mat_0145_2, m4b);
                    var rhs_mat_2367_02 = Avx2.And(rhs_raw_mat_2367_2, m4b);
                    var rhs_mat_0145_03 = Avx2.And(rhs_raw_mat_0145_3, m4b);
                    var rhs_mat_2367_03 = Avx2.And(rhs_raw_mat_2367_3, m4b);

                    // second sub block
                    var rhs_mat_0145_10 = Avx2.And(Shr4(rhs_raw_mat_0145_0), m4b);
                    var rhs_mat_2367_10 = Avx2.And(Shr4(rhs_raw_mat_2367_0), m4b);
                    var rhs_mat_0145_11 = Avx2.And(Shr4(rhs_raw_mat_0145_1), m4b);
                    var rhs_mat_2367_11 = Avx2.And(Shr4(rhs_raw_mat_2367_1), m4b);
                    var rhs_mat_0145_12 = Avx2.And(Shr4(rhs_raw_mat_0145_2), m4b);
                    var rhs_mat_2367_12 = Avx2.And(Shr4(rhs_raw_mat_2367_2), m4b);
                    var rhs_mat_0145_13 = Avx2.And(Shr4(rhs_raw_mat_0145_3), m4b);
                    var rhs_mat_2367_13 = Avx2.And(Shr4(rhs_raw_mat_2367_3), m4b);

                    // Shuffle pattern one - right side input
                    var rhs_mat_0145_00_sp1 = Sh136(rhs_mat_0145_00);
                    var rhs_mat_2367_00_sp1 = Sh136(rhs_mat_2367_00);
                    var rhs_mat_0145_01_sp1 = Sh136(rhs_mat_0145_01);
                    var rhs_mat_2367_01_sp1 = Sh136(rhs_mat_2367_01);
                    var rhs_mat_0145_02_sp1 = Sh136(rhs_mat_0145_02);
                    var rhs_mat_2367_02_sp1 = Sh136(rhs_mat_2367_02);
                    var rhs_mat_0145_03_sp1 = Sh136(rhs_mat_0145_03);
                    var rhs_mat_2367_03_sp1 = Sh136(rhs_mat_2367_03);
                    var rhs_mat_0145_10_sp1 = Sh136(rhs_mat_0145_10);
                    var rhs_mat_2367_10_sp1 = Sh136(rhs_mat_2367_10);
                    var rhs_mat_0145_11_sp1 = Sh136(rhs_mat_0145_11);
                    var rhs_mat_2367_11_sp1 = Sh136(rhs_mat_2367_11);
                    var rhs_mat_0145_12_sp1 = Sh136(rhs_mat_0145_12);
                    var rhs_mat_2367_12_sp1 = Sh136(rhs_mat_2367_12);
                    var rhs_mat_0145_13_sp1 = Sh136(rhs_mat_0145_13);
                    var rhs_mat_2367_13_sp1 = Sh136(rhs_mat_2367_13);

                    // Shuffle pattern two - right side input
                    var rhs_mat_0145_00_sp2 = Sh221(rhs_mat_0145_00);
                    var rhs_mat_2367_00_sp2 = Sh221(rhs_mat_2367_00);
                    var rhs_mat_0145_01_sp2 = Sh221(rhs_mat_0145_01);
                    var rhs_mat_2367_01_sp2 = Sh221(rhs_mat_2367_01);
                    var rhs_mat_0145_02_sp2 = Sh221(rhs_mat_0145_02);
                    var rhs_mat_2367_02_sp2 = Sh221(rhs_mat_2367_02);
                    var rhs_mat_0145_03_sp2 = Sh221(rhs_mat_0145_03);
                    var rhs_mat_2367_03_sp2 = Sh221(rhs_mat_2367_03);
                    var rhs_mat_0145_10_sp2 = Sh221(rhs_mat_0145_10);
                    var rhs_mat_2367_10_sp2 = Sh221(rhs_mat_2367_10);
                    var rhs_mat_0145_11_sp2 = Sh221(rhs_mat_0145_11);
                    var rhs_mat_2367_11_sp2 = Sh221(rhs_mat_2367_11);
                    var rhs_mat_0145_12_sp2 = Sh221(rhs_mat_0145_12);
                    var rhs_mat_2367_12_sp2 = Sh221(rhs_mat_2367_12);
                    var rhs_mat_0145_13_sp2 = Sh221(rhs_mat_0145_13);
                    var rhs_mat_2367_13_sp2 = Sh221(rhs_mat_2367_13);

                    // DEVIATION 1: utmp/kmask block replaced by direct loads of the pre-decoded
                    // scales/mins. Lane layout is identical to the original's scales_0/scales_1
                    // (s0,s0,s1,s1,...) and mins_01 (mA0,mB0,mA1,mB1,...).
                    var scales_0 = DupWiden(bsc + (2 * sb) * 8);
                    var scales_1 = DupWiden(bsc + (2 * sb + 1) * 8);
                    var mins_01 = InterleaveWiden(bmn + (2 * sb) * 8, bmn + (2 * sb + 1) * 8);

                    var scale_0145_0 = Sh68s(scales_0);
                    var scale_2367_0 = Sh238s(scales_0);
                    var scale_0145_1 = Sh68s(scales_1);
                    var scale_2367_1 = Sh238s(scales_1);

                    for (int rp = 0; rp < nrp; rp++)
                    {
                        byte* ablk = a_ptrs[rp] + (long)b * Q8Kx4BlockBytes;
                        float* ad = (float*)ablk;
                        sbyte* aqs = (sbyte*)(ablk + 16);
                        short* absums = (short*)(ablk + 16 + QK_K * 4);
                        sbyte* aq = aqs + 256 * sb;

                        // Load the four block_q8_k values interleaved in chunks of eight bytes
                        var lhs_mat_0123_00 = Vector256.LoadUnsafe(ref *(aq + 0)).AsByte();
                        var lhs_mat_01_00 = Perm0(lhs_mat_0123_00);
                        var lhs_mat_23_00 = Perm17(lhs_mat_0123_00);
                        var lhs_mat_0123_01 = Vector256.LoadUnsafe(ref *(aq + 32)).AsByte();
                        var lhs_mat_01_01 = Perm0(lhs_mat_0123_01);
                        var lhs_mat_23_01 = Perm17(lhs_mat_0123_01);
                        var lhs_mat_0123_02 = Vector256.LoadUnsafe(ref *(aq + 64)).AsByte();
                        var lhs_mat_01_02 = Perm0(lhs_mat_0123_02);
                        var lhs_mat_23_02 = Perm17(lhs_mat_0123_02);
                        var lhs_mat_0123_03 = Vector256.LoadUnsafe(ref *(aq + 96)).AsByte();
                        var lhs_mat_01_03 = Perm0(lhs_mat_0123_03);
                        var lhs_mat_23_03 = Perm17(lhs_mat_0123_03);
                        var lhs_mat_0123_10 = Vector256.LoadUnsafe(ref *(aq + 128)).AsByte();
                        var lhs_mat_01_10 = Perm0(lhs_mat_0123_10);
                        var lhs_mat_23_10 = Perm17(lhs_mat_0123_10);
                        var lhs_mat_0123_11 = Vector256.LoadUnsafe(ref *(aq + 160)).AsByte();
                        var lhs_mat_01_11 = Perm0(lhs_mat_0123_11);
                        var lhs_mat_23_11 = Perm17(lhs_mat_0123_11);
                        var lhs_mat_0123_12 = Vector256.LoadUnsafe(ref *(aq + 192)).AsByte();
                        var lhs_mat_01_12 = Perm0(lhs_mat_0123_12);
                        var lhs_mat_23_12 = Perm17(lhs_mat_0123_12);
                        var lhs_mat_0123_13 = Vector256.LoadUnsafe(ref *(aq + 224)).AsByte();
                        var lhs_mat_01_13 = Perm0(lhs_mat_0123_13);
                        var lhs_mat_23_13 = Perm17(lhs_mat_0123_13);

                        // Bsums - four bsums for two sub blocks from the different Q8_K blocks
                        var lhs_bsums_0123_01 = Vector256.LoadUnsafe(ref *(absums + 16 * sb));
                        var lhs_bsums_hsum_0123_01 =
                            Ssse3.HorizontalAdd(lhs_bsums_0123_01.GetLower(), lhs_bsums_0123_01.GetUpper())
                                 .ToVector256Unsafe();
                        lhs_bsums_hsum_0123_01 = Perm0s(lhs_bsums_hsum_0123_01);

                        // Shuffle pattern one - left side input
                        var lhs_mat_01_00_sp1 = Sh160(lhs_mat_01_00);
                        var lhs_mat_23_00_sp1 = Sh160(lhs_mat_23_00);
                        var lhs_mat_01_01_sp1 = Sh160(lhs_mat_01_01);
                        var lhs_mat_23_01_sp1 = Sh160(lhs_mat_23_01);
                        var lhs_mat_01_02_sp1 = Sh160(lhs_mat_01_02);
                        var lhs_mat_23_02_sp1 = Sh160(lhs_mat_23_02);
                        var lhs_mat_01_03_sp1 = Sh160(lhs_mat_01_03);
                        var lhs_mat_23_03_sp1 = Sh160(lhs_mat_23_03);
                        var lhs_mat_01_10_sp1 = Sh160(lhs_mat_01_10);
                        var lhs_mat_23_10_sp1 = Sh160(lhs_mat_23_10);
                        var lhs_mat_01_11_sp1 = Sh160(lhs_mat_01_11);
                        var lhs_mat_23_11_sp1 = Sh160(lhs_mat_23_11);
                        var lhs_mat_01_12_sp1 = Sh160(lhs_mat_01_12);
                        var lhs_mat_23_12_sp1 = Sh160(lhs_mat_23_12);
                        var lhs_mat_01_13_sp1 = Sh160(lhs_mat_01_13);
                        var lhs_mat_23_13_sp1 = Sh160(lhs_mat_23_13);

                        // Shuffle pattern two - left side input
                        var lhs_mat_01_00_sp2 = Sh245(lhs_mat_01_00);
                        var lhs_mat_23_00_sp2 = Sh245(lhs_mat_23_00);
                        var lhs_mat_01_01_sp2 = Sh245(lhs_mat_01_01);
                        var lhs_mat_23_01_sp2 = Sh245(lhs_mat_23_01);
                        var lhs_mat_01_02_sp2 = Sh245(lhs_mat_01_02);
                        var lhs_mat_23_02_sp2 = Sh245(lhs_mat_23_02);
                        var lhs_mat_01_03_sp2 = Sh245(lhs_mat_01_03);
                        var lhs_mat_23_03_sp2 = Sh245(lhs_mat_23_03);
                        var lhs_mat_01_10_sp2 = Sh245(lhs_mat_01_10);
                        var lhs_mat_23_10_sp2 = Sh245(lhs_mat_23_10);
                        var lhs_mat_01_11_sp2 = Sh245(lhs_mat_01_11);
                        var lhs_mat_23_11_sp2 = Sh245(lhs_mat_23_11);
                        var lhs_mat_01_12_sp2 = Sh245(lhs_mat_01_12);
                        var lhs_mat_23_12_sp2 = Sh245(lhs_mat_23_12);
                        var lhs_mat_01_13_sp2 = Sh245(lhs_mat_01_13);
                        var lhs_mat_23_13_sp2 = Sh245(lhs_mat_23_13);

                        // maddubs within 32-bit lanes, chained in int16 (safe: 4 * 2*15*127 < 32767)
                        var iacc_mat_00_0_sp1 = Add16(Add16(Add16(Mul(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), Mul(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), Mul(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), Mul(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                        var iacc_mat_01_0_sp1 = Add16(Add16(Add16(Mul(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), Mul(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), Mul(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), Mul(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                        var iacc_mat_10_0_sp1 = Add16(Add16(Add16(Mul(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), Mul(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), Mul(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), Mul(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                        var iacc_mat_11_0_sp1 = Add16(Add16(Add16(Mul(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), Mul(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), Mul(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), Mul(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                        var iacc_mat_00_1_sp1 = Add16(Add16(Add16(Mul(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), Mul(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), Mul(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), Mul(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                        var iacc_mat_01_1_sp1 = Add16(Add16(Add16(Mul(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), Mul(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), Mul(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), Mul(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                        var iacc_mat_10_1_sp1 = Add16(Add16(Add16(Mul(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), Mul(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), Mul(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), Mul(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                        var iacc_mat_11_1_sp1 = Add16(Add16(Add16(Mul(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), Mul(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), Mul(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), Mul(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                        var iacc_mat_00_0_sp2 = Add16(Add16(Add16(Mul(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), Mul(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), Mul(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), Mul(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                        var iacc_mat_01_0_sp2 = Add16(Add16(Add16(Mul(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), Mul(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), Mul(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), Mul(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                        var iacc_mat_10_0_sp2 = Add16(Add16(Add16(Mul(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), Mul(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), Mul(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), Mul(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                        var iacc_mat_11_0_sp2 = Add16(Add16(Add16(Mul(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), Mul(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), Mul(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), Mul(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                        var iacc_mat_00_1_sp2 = Add16(Add16(Add16(Mul(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), Mul(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), Mul(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), Mul(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                        var iacc_mat_01_1_sp2 = Add16(Add16(Add16(Mul(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), Mul(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), Mul(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), Mul(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                        var iacc_mat_10_1_sp2 = Add16(Add16(Add16(Mul(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), Mul(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), Mul(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), Mul(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                        var iacc_mat_11_1_sp2 = Add16(Add16(Add16(Mul(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), Mul(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), Mul(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), Mul(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                        // Outputs of both shuffle patterns are added to sum all 32 values in block
                        var iacc_mat_00_0 = Avx2.Add(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        var iacc_mat_01_0 = Avx2.Add(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        var iacc_mat_10_0 = Avx2.Add(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        var iacc_mat_11_0 = Avx2.Add(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);
                        var iacc_mat_00_1 = Avx2.Add(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        var iacc_mat_01_1 = Avx2.Add(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        var iacc_mat_10_1 = Avx2.Add(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        var iacc_mat_11_1 = Avx2.Add(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        var i00_0 = Avx2.MultiplyAddAdjacent(iacc_mat_00_0, scale_0145_0);
                        var i01_0 = Avx2.MultiplyAddAdjacent(iacc_mat_01_0, scale_2367_0);
                        var i10_0 = Avx2.MultiplyAddAdjacent(iacc_mat_10_0, scale_0145_0);
                        var i11_0 = Avx2.MultiplyAddAdjacent(iacc_mat_11_0, scale_2367_0);
                        var i00_1 = Avx2.MultiplyAddAdjacent(iacc_mat_00_1, scale_0145_1);
                        var i01_1 = Avx2.MultiplyAddAdjacent(iacc_mat_01_1, scale_2367_1);
                        var i10_1 = Avx2.MultiplyAddAdjacent(iacc_mat_10_1, scale_0145_1);
                        var i11_1 = Avx2.MultiplyAddAdjacent(iacc_mat_11_1, scale_2367_1);

                        // Straighten out to make 4 row vectors
                        var iacc_row_0_0 = Avx2.Blend(i00_0, Avx2.Shuffle(i01_0, 78), 204);
                        var iacc_row_1_0 = Avx2.Blend(Avx2.Shuffle(i00_0, 78), i01_0, 204);
                        var iacc_row_2_0 = Avx2.Blend(i10_0, Avx2.Shuffle(i11_0, 78), 204);
                        var iacc_row_3_0 = Avx2.Blend(Avx2.Shuffle(i10_0, 78), i11_0, 204);
                        var iacc_row_0_1 = Avx2.Blend(i00_1, Avx2.Shuffle(i01_1, 78), 204);
                        var iacc_row_1_1 = Avx2.Blend(Avx2.Shuffle(i00_1, 78), i01_1, 204);
                        var iacc_row_2_1 = Avx2.Blend(i10_1, Avx2.Shuffle(i11_1, 78), 204);
                        var iacc_row_3_1 = Avx2.Blend(Avx2.Shuffle(i10_1, 78), i11_1, 204);

                        var iacc_row_0 = Avx2.Add(iacc_row_0_0, iacc_row_0_1);
                        var iacc_row_1 = Avx2.Add(iacc_row_1_0, iacc_row_1_1);
                        var iacc_row_2 = Avx2.Add(iacc_row_2_0, iacc_row_2_1);
                        var iacc_row_3 = Avx2.Add(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale values for the 4 Q8_K blocks and repeat across lanes
                        var row_scale_f32_sse = Vector128.LoadUnsafe(ref *ad);
                        var row_scale_f32 = Vector256.Create(row_scale_f32_sse, row_scale_f32_sse);

                        acc_rows[rp * 4] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_0), Avx.Multiply(col_scale_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_1), Avx.Multiply(col_scale_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_2), Avx.Multiply(col_scale_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_3), Avx.Multiply(col_scale_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        var iacc_row_min_0 = Avx2.MultiplyAddAdjacent(Sh0s(lhs_bsums_hsum_0123_01), mins_01);
                        var iacc_row_min_1 = Avx2.MultiplyAddAdjacent(Sh85s(lhs_bsums_hsum_0123_01), mins_01);
                        var iacc_row_min_2 = Avx2.MultiplyAddAdjacent(Sh170s(lhs_bsums_hsum_0123_01), mins_01);
                        var iacc_row_min_3 = Avx2.MultiplyAddAdjacent(Sh255s(lhs_bsums_hsum_0123_01), mins_01);

                        acc_min_rows[rp * 4] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_min_0), Avx.Multiply(col_dmin_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_min_1), Avx.Multiply(col_dmin_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_min_2), Avx.Multiply(col_dmin_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = Fma.MultiplyAdd(Avx.ConvertToVector256Single(iacc_row_min_3), Avx.Multiply(col_dmin_f32, Avx.Shuffle(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);
                    }
                }
            }

            // Store the accumulated values. Output is [token * rows + col], so a whole 8-column
            // group for one token is contiguous — matching the original's row-major store.
            for (int i = 0; i < nrp * 4; i++)
            {
                int tokenRow = y * 4 + i;
                if (tokenRow >= validRows) break;      // zero-padded tail: computed, not stored
                var v = Avx.Subtract(acc_rows[i], acc_min_rows[i]);
                v.StoreUnsafe(ref *(s + (long)tokenRow * bs + (long)x * 8));
            }

            y += nrp;
        }
    }

    // ---- 1:1 wrappers ----
    //
    // PORTING TRAP (CA1857). The obvious C# rendering of these is one helper taking the shuffle
    // immediate as a `byte c` parameter. That does NOT work: Avx2.Shuffle / Blend / Permute2x128
    // require a compile-time-constant immediate, and a runtime value makes the JIT emit a slow
    // dispatch path instead of the single instruction. The analyser flags it as CA1857 and
    // TreatWarningsAsErrors turns it fatal — which is fortunate, because otherwise the wrappers
    // would have silently gutted the kernel's performance while looking tidy.
    //
    // Hence one specialisation per immediate actually used. Verbose, but each is a literal.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Blend240(Vector256<byte> a, Vector256<byte> b)
        => Avx2.Blend(a.AsInt32(), b.AsInt32(), 240).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Perm8x32(Vector256<byte> a, Vector256<int> idx)
        => Avx2.PermuteVar8x32(a.AsInt32(), idx).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Shr4(Vector256<byte> a)
        => Avx2.ShiftRightLogical(a.AsInt16(), 4).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Sh136(Vector256<byte> a) => Avx2.Shuffle(a.AsInt32(), 136).AsByte();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Sh221(Vector256<byte> a) => Avx2.Shuffle(a.AsInt32(), 221).AsByte();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Sh160(Vector256<byte> a) => Avx2.Shuffle(a.AsInt32(), 160).AsByte();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Sh245(Vector256<byte> a) => Avx2.Shuffle(a.AsInt32(), 245).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Sh68s(Vector256<short> a) => Avx2.Shuffle(a.AsInt32(), 68).AsInt16();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Sh238s(Vector256<short> a) => Avx2.Shuffle(a.AsInt32(), 238).AsInt16();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Sh0s(Vector256<short> a) => Avx2.Shuffle(a.AsInt32(), 0).AsInt16();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Sh85s(Vector256<short> a) => Avx2.Shuffle(a.AsInt32(), 85).AsInt16();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Sh170s(Vector256<short> a) => Avx2.Shuffle(a.AsInt32(), 170).AsInt16();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Sh255s(Vector256<short> a) => Avx2.Shuffle(a.AsInt32(), 255).AsInt16();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Perm0(Vector256<byte> a)
        => Avx2.Permute2x128(a.AsInt32(), a.AsInt32(), 0).AsByte();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Perm17(Vector256<byte> a)
        => Avx2.Permute2x128(a.AsInt32(), a.AsInt32(), 17).AsByte();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Perm0s(Vector256<short> a)
        => Avx2.Permute2x128(a.AsInt32(), a.AsInt32(), 0).AsInt16();

    /// <summary>_mm256_maddubs_epi16 — unsigned B against signed A.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Mul(Vector256<byte> rhs, Vector256<byte> lhs)
        => Avx2.MultiplyAddAdjacent(rhs, lhs.AsSByte());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Add16(Vector256<short> a, Vector256<short> b) => Avx2.Add(a, b);

    /// <summary>Replaces the original's scales_0/_1: duplicate 8 bytes and widen to int16.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> DupWiden(byte* p)
    {
        var v = Vector128.LoadUnsafe(ref *p);
        return Avx2.ConvertToVector256Int16(Sse2.UnpackLow(v, v));
    }

    /// <summary>Replaces the original's mins_01: byte-interleave two sub-blocks' mins, widen.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> InterleaveWiden(byte* a, byte* b)
    {
        var va = Vector128.LoadUnsafe(ref *a);
        var vb = Vector128.LoadUnsafe(ref *b);
        return Avx2.ConvertToVector256Int16(Sse2.UnpackLow(va, vb));
    }
}
