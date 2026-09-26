namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Batched (prefill) Q6_K x Q8_K GEMM that pairs Q6_K's 16-element scale groups in registers so
/// two <c>vpmaddubsw</c> results share one scale <c>vpmaddwd</c>. Reads the stock GGUF bytes: no
/// repacked copy. Original work — llama.cpp has no x86 Q6_K GEMM (repack.cpp covers q4_0, q4_K,
/// iq4_nl, mxfp4, q2_K).
/// </summary>
/// <remarks>
/// <para><b>The trick.</b> Q6_K has one int8 scale per 16 weights. The stock decode yields, per half
/// super-block, four 32-byte vectors <c>w0..w3</c>, each spanning two whole groups (<c>w0</c> =
/// groups 0|1, <c>w1</c> = 2|3, <c>w2</c> = 4|5, <c>w3</c> = 6|7), so the row-major kernel needs a
/// scale <c>vpmaddwd</c> after every <c>vpmaddubsw</c>. Here <c>vpunpcklqdq(w0, w1)</c> gives
/// 8-byte runs <c>[g0 0-7 | g2 0-7 | g1 0-7 | g3 0-7]</c> and <c>vpunpckhqdq(w0, w1)</c> the
/// matching elements 8-15 of the same groups in the same positions. Their products add in int16
/// (a <c>maddubs</c> pair is at most 2·63·127 = 16002, two of them 32004 &lt; 32767; Q8_K
/// activations are clamped to [-127, 127]) before a single scale <c>vpmaddwd</c>: 6 multiplies per
/// 128 weights per token instead of 8. The unpacks are paid once per row per super-block and
/// amortised over a token group. Activations are permuted to the same element order once, right
/// after quantisation; <c>bsums</c> are per group and so unaffected.</para>
///
/// <para><b>Loop shape.</b> Row-outer, token-inner per super-block: a half super-block's four
/// weight vectors and two scale vectors stay in registers while each token of the group streams
/// its activations as memory operands (no spills — the row-major <c>DotQ6K_Q8K_8In</c> carries
/// eight accumulators plus eight weight/scale vectors and spills).</para>
///
/// <para><b>Numerics.</b> Each super-block's integer dot is exact, as in the row-major kernel, but
/// the int32 lane partition differs, so the per-lane float accumulation can differ from
/// <see cref="SimdKernels.DotQ6K_Q8K"/> in the last bits (test tolerance 1e-5 relative). Used
/// for every prefill batch size, so chunked and unchunked prefill of one prompt stay identical.</para>
/// </remarks>
public static unsafe class Q6KPrefillGemm
{
    private const int BlockBytes = 210;
    private const int RowBlock = 16;
    private const int TokenGroup = 16;

    /// <summary>Whether this kernel handles the shape on this CPU.</summary>
    public static bool CanUse(int rows, int cols)
        => Avx2.IsSupported && Fma.IsSupported && rows > 0 && cols > 0 && cols % 256 == 0;

    /// <summary>
    /// Original half-super-block element feeding activation slot <c>s*32 + p</c>: vector pair
    /// <c>P = s/2</c>, element half <c>s%2</c>, qword <c>p/8</c> -> group <c>4P + {0,2,1,3}[p/8]</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SourceIndex(int slot)
    {
        int s = slot >> 5, qi = (slot >> 3) & 3;
        int g = 4 * (s >> 1) + ((qi & 1) << 1) + (qi >> 1);
        return 16 * g + (s & 1) * 8 + (slot & 7);
    }

    /// <summary>Permute a Q8_K scratch row's quants into the kernel's element order (in place).</summary>
    public static void PermuteActivationRow(byte* scratch, int cols)
    {
        int nb = cols / 256;
        sbyte* qs = (sbyte*)(scratch + nb * 4);
        sbyte* tmp = stackalloc sbyte[128];
        for (int half = 0; half < nb * 2; half++)
        {
            sbyte* x = qs + half * 128;
            Buffer.MemoryCopy(x, tmp, 128, 128);
            for (int slot = 0; slot < 128; slot++) x[slot] = tmp[SourceIndex(slot)];
        }
    }

    /// <summary>
    /// <c>output[token * rows + row]</c> = stock Q6_K <paramref name="weights"/> x activations,
    /// like <see cref="SimdKernels.MatMulBatched"/>. Any batch size.
    /// </summary>
    public static bool TryMatMulBatched(float* output, byte* weights, float* input,
        int batchSize, int rows, int cols)
    {
        if (!CanUse(rows, cols) || batchSize < 1) return false;

        const int MaxChunk = 512;
        if (batchSize > MaxChunk)
        {
            for (int start = 0; start < batchSize; start += MaxChunk)
            {
                int n = Math.Min(MaxChunk, batchSize - start);
                TryMatMulBatched(output + (long)start * rows, weights, input + (long)start * cols, n, rows, cols);
            }
            return true;
        }

        int stride = SimdKernels.Q8KScratchBytes(cols);
        byte* scratch = (byte*)NativeMemory.Alloc((nuint)((long)stride * batchSize));
        try
        {
            if (batchSize >= 4)
                Parallel.For(0, batchSize, SimdKernels.ParallelOpts, n =>
                {
                    SimdKernels.QuantizeRowToQ8K(input + (long)n * cols, cols, scratch + (long)n * stride);
                    PermuteActivationRow(scratch + (long)n * stride, cols);
                });
            else
                for (int n = 0; n < batchSize; n++)
                {
                    SimdKernels.QuantizeRowToQ8K(input + (long)n * cols, cols, scratch + (long)n * stride);
                    PermuteActivationRow(scratch + (long)n * stride, cols);
                }

            int rowBlocks = (rows + RowBlock - 1) / RowBlock;
            if (rowBlocks > 1)
                Parallel.For(0, rowBlocks, SimdKernels.ParallelOpts, blk =>
                    RowBlockKernel(output, weights, scratch, stride, batchSize, rows, cols, blk));
            else
                RowBlockKernel(output, weights, scratch, stride, batchSize, rows, cols, 0);
        }
        finally
        {
            NativeMemory.Free(scratch);
        }
        return true;
    }

    private static void RowBlockKernel(float* output, byte* weights, byte* scratch, int stride,
        int batchSize, int rows, int cols, int blk)
    {
        int nb = cols / 256;
        long bytesPerRow = (long)nb * BlockBytes;
        int r0 = blk * RowBlock, r1 = Math.Min(rows, r0 + RowBlock);

        Vector256<float>* acc = stackalloc Vector256<float>[TokenGroup];
        Vector256<int>* sumi = stackalloc Vector256<int>[TokenGroup];

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);

        int qsOff = nb * 4, bsOff = nb * 4 + nb * 256;

        for (int t0 = 0; t0 < batchSize; t0 += TokenGroup)
        {
            int nt = Math.Min(TokenGroup, batchSize - t0);
            byte* act0 = scratch + (long)t0 * stride;

            for (int r = r0; r < r1; r++)
            {
                byte* row = weights + r * bytesPerRow;
                for (int t = 0; t < nt; t++) acc[t] = Vector256<float>.Zero;

                for (int b = 0; b < nb; b++)
                {
                    byte* x = row + b * BlockBytes;
                    var scales128 = Vector128.LoadUnsafe(ref *(x + 192)).AsSByte();
                    float dw = (float)BitConverter.UInt16BitsToHalf((ushort)(x[208] | (x[209] << 8)));
                    long qb = qsOff + (long)b * 256;

                    // ---- half 0: groups 0-7 (scales 0-7) ----
                    {
                        Decode(x, 0, m3, m12, m48, m192, m15, out var w0, out var w1, out var w2, out var w3);
                        var scA = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales128, Vector128.Create(
                            (sbyte)0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3)));
                        var scB = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales128, Vector128.Create(
                            (sbyte)4, 4, 4, 4, 6, 6, 6, 6, 5, 5, 5, 5, 7, 7, 7, 7)));
                        byte* a = act0 + qb;
                        for (int t = 0; t < nt; t++, a += stride)
                            sumi[t] = HalfDot(a, w0, w1, w2, w3, scA, scB);
                    }

                    // ---- half 1: groups 8-15, then offset correction + float accumulate ----
                    {
                        Decode(x, 1, m3, m12, m48, m192, m15, out var w0, out var w1, out var w2, out var w3);
                        var scA = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales128, Vector128.Create(
                            (sbyte)8, 8, 8, 8, 10, 10, 10, 10, 9, 9, 9, 9, 11, 11, 11, 11)));
                        var scB = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales128, Vector128.Create(
                            (sbyte)12, 12, 12, 12, 14, 14, 14, 14, 13, 13, 13, 13, 15, 15, 15, 15)));
                        var scales16 = Avx2.ConvertToVector256Int16(scales128);
                        byte* a = act0 + qb + 128;
                        byte* tok = act0;
                        for (int t = 0; t < nt; t++, a += stride, tok += stride)
                        {
                            var s = Avx2.Add(sumi[t], HalfDot(a, w0, w1, w2, w3, scA, scB));
                            var bsums = Vector256.LoadUnsafe(ref *(short*)(tok + bsOff + b * 32));
                            var corr = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(bsums, scales16), 5);
                            float d = dw * ((float*)tok)[b];
                            acc[t] = Fma.MultiplyAdd(Vector256.Create(d),
                                Avx.ConvertToVector256Single(Avx2.Subtract(s, corr)), acc[t]);
                        }
                    }
                }

                for (int t = 0; t < nt; t++)
                    output[(long)(t0 + t) * rows + r] = HSum(acc[t]);
            }
        }
    }

    /// <summary>
    /// Stock ggml Q6_K half-block sextet decode followed by the group-pairing unpacks: returns
    /// <c>unpacklo/hi(q0, q1)</c> and <c>unpacklo/hi(q2, q3)</c> of the stock vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Decode(byte* x, int h,
        Vector256<byte> m3, Vector256<byte> m12, Vector256<byte> m48, Vector256<byte> m192, Vector256<byte> m15,
        out Vector256<byte> w0, out Vector256<byte> w1, out Vector256<byte> w2, out Vector256<byte> w3)
    {
        var bits1 = Vector256.LoadUnsafe(ref *(x + h * 64));
        var bits2 = Vector256.LoadUnsafe(ref *(x + h * 64 + 32));
        var bitsH = Vector256.LoadUnsafe(ref *(x + 128 + h * 32));
        var q0 = Avx2.Or(Avx2.And(bits1, m15), Avx2.ShiftLeftLogical(Avx2.And(bitsH, m3).AsInt16(), 4).AsByte());
        var q1 = Avx2.Or(Avx2.And(bits2, m15), Avx2.ShiftLeftLogical(Avx2.And(bitsH, m12).AsInt16(), 2).AsByte());
        var q2 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(bits1.AsInt16(), 4).AsByte(), m15), Avx2.And(bitsH, m48));
        var q3 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(bits2.AsInt16(), 4).AsByte(), m15),
            Avx2.ShiftRightLogical(Avx2.And(bitsH, m192).AsInt16(), 2).AsByte());
        w0 = Avx2.UnpackLow(q0.AsUInt64(), q1.AsUInt64()).AsByte();
        w1 = Avx2.UnpackHigh(q0.AsUInt64(), q1.AsUInt64()).AsByte();
        w2 = Avx2.UnpackLow(q2.AsUInt64(), q3.AsUInt64()).AsByte();
        w3 = Avx2.UnpackHigh(q2.AsUInt64(), q3.AsUInt64()).AsByte();
    }

    /// <summary>One token x one half super-block: 4 maddubs, 2 int16 adds, 2 scale madds.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> HalfDot(byte* a,
        Vector256<byte> w0, Vector256<byte> w1, Vector256<byte> w2, Vector256<byte> w3,
        Vector256<short> scA, Vector256<short> scB)
    {
        var pA = Avx2.Add(
            Avx2.MultiplyAddAdjacent(w0, Vector256.LoadUnsafe(ref *(sbyte*)a)),
            Avx2.MultiplyAddAdjacent(w1, Vector256.LoadUnsafe(ref *(sbyte*)(a + 32))));
        var pB = Avx2.Add(
            Avx2.MultiplyAddAdjacent(w2, Vector256.LoadUnsafe(ref *(sbyte*)(a + 64))),
            Avx2.MultiplyAddAdjacent(w3, Vector256.LoadUnsafe(ref *(sbyte*)(a + 96))));
        return Avx2.Add(Avx2.MultiplyAddAdjacent(pA, scA), Avx2.MultiplyAddAdjacent(pB, scB));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum(Vector256<float> v)
    {
        var s = Sse.Add(v.GetLower(), Avx.ExtractVector128(v, 1));
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        s = Sse.AddScalar(s, Sse.Shuffle(s, s, 1));
        return s.ToScalar();
    }
}
