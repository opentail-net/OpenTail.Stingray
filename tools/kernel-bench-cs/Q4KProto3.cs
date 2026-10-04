using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Variations on <see cref="Q4KProto.Dot"/> to find the last ~4 ns/row against ggml: (A) two integer accumulators, (B) d/dmin of up to 8 blocks converted in one
/// vector pass, (R) row loop inside the kernel (one prologue per matrix instead of per row). Flags are constants at each call site so the JIT specialises each copy.
/// Per-row results are bit-identical to the baseline prototype.
/// </summary>
internal static unsafe class Q4KProto3
{
    private const float HalfRescale = 5.192296858534828e33f;   // 2^112

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> HalfPairToFloat(uint packed)
    {
        var hh = Sse41.ConvertToVector128Int32(Vector128.CreateScalarUnsafe(packed).AsUInt16());
        return Sse.Multiply(Sse2.Or(Sse2.ShiftLeftLogical(Sse2.And(hh, Vector128.Create(0x8000)), 16), Sse2.ShiftLeftLogical(Sse2.And(hh, Vector128.Create(0x7fff)), 13)).AsSingle(), Vector128.Create(HalfRescale));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Cv(Vector256<uint> h) =>
        Avx.Multiply(Avx2.Or(Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x8000u)), 16), Avx2.ShiftLeftLogical(Avx2.And(h, Vector256.Create(0x7fffu)), 13)).AsSingle(), Vector256.Create(HalfRescale));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<int> Chunk(byte* q4, sbyte* q8, Vector256<byte> scalesB, Vector256<byte> m4, ushort maskLo, ushort maskHi)
    {
        var qb = Avx.LoadVector256(q4);
        var pl = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(qb, m4), Avx.LoadVector256(q8)), Avx2.Shuffle(scalesB, Vector256.Create(maskLo).AsByte()).AsInt16());
        var ph = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(Avx2.ShiftRightLogical(qb.AsInt16(), 4).AsByte(), m4), Avx.LoadVector256(q8 + 32)), Avx2.Shuffle(scalesB, Vector256.Create(maskHi).AsByte()).AsInt16());
        return Avx2.Add(pl, ph);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Core(byte* row, byte* y, int numBlocks, bool twoAcc, bool batchHalf, float* dm, uint* w)
    {
        var m4 = Vector256.Create((byte)0x0F);
        Vector256<float> acc = Vector256<float>.Zero;
        Vector128<float> accM = Vector128<float>.Zero;
        for (int i = 0; i < numBlocks; i++, row += 144, y += 292)
        {
            if (batchHalf && (i & 7) == 0)
            {
                // convert d/dmin for the next (up to) 8 blocks at once: 8 packed (d | dmin << 16) words -> 16 floats in dm[]
                int n = Math.Min(8, numBlocks - i);
                for (int k = 0; k < n; k++) w[k] = *(uint*)(row + 144 * k);
                var u = Avx.LoadVector256(w);
                Avx.Store(dm, Cv(Avx2.And(u, Vector256.Create(0xffffu))));
                Avx.Store(dm + 8, Cv(Avx2.ShiftRightLogical(u, 16)));
            }
            float yd = *(float*)y;
            sbyte* q8 = (sbyte*)(y + 4);
            short* bsums = (short*)(y + 260);
            float d, dmin;
            if (batchHalf) { d = yd * dm[i & 7]; dmin = -yd * dm[8 + (i & 7)]; }
            else { var dd = HalfPairToFloat(*(uint*)row); d = yd * dd.ToScalar(); dmin = -yd * dd.GetElement(1); }

            uint u0 = *(uint*)(row + 4), u1 = *(uint*)(row + 8), u2 = *(uint*)(row + 12);
            uint t3 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
            uint t2 = u1 & 0x3f3f3f3fu;
            uint t1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
            uint t0 = u0 & 0x3f3f3f3fu;
            var ms = Avx2.ConvertToVector256Int16(Vector128.Create(t0, t1, t2, t3).AsByte());
            var sums = Avx.LoadVector256(bsums);
            var q8s = Ssse3.HorizontalAdd(sums.GetLower(), sums.GetUpper());
            var prod = Sse2.MultiplyAddAdjacent(ms.GetUpper(), q8s);
            accM = Fma.MultiplyAdd(Vector128.Create(dmin), Sse2.ConvertToVector128Single(prod), accM);

            var sc128 = ms.GetLower();
            var scalesB = Vector256.Create(sc128, sc128).AsByte();
            byte* q4 = row + 16;
            var sumi = Vector256<int>.Zero;
            var sumi2 = Vector256<int>.Zero;
            sumi = Avx2.Add(sumi, Chunk(q4, q8, scalesB, m4, 0x0100, 0x0302));
            if (twoAcc) sumi2 = Avx2.Add(sumi2, Chunk(q4 + 32, q8 + 64, scalesB, m4, 0x0504, 0x0706)); else sumi = Avx2.Add(sumi, Chunk(q4 + 32, q8 + 64, scalesB, m4, 0x0504, 0x0706));
            sumi = Avx2.Add(sumi, Chunk(q4 + 64, q8 + 128, scalesB, m4, 0x0908, 0x0B0A));
            if (twoAcc) sumi2 = Avx2.Add(sumi2, Chunk(q4 + 96, q8 + 192, scalesB, m4, 0x0D0C, 0x0F0E)); else sumi = Avx2.Add(sumi, Chunk(q4 + 96, q8 + 192, scalesB, m4, 0x0D0C, 0x0F0E));
            if (twoAcc) sumi = Avx2.Add(sumi, sumi2);
            acc = Fma.MultiplyAdd(Vector256.Create(d), Avx.ConvertToVector256Single(sumi), acc);
        }
        var h = Sse.Add(acc.GetLower(), acc.GetUpper());
        h = Sse.Add(h, Sse.MoveHighToLow(h, h));
        h = Sse.AddScalar(h, Sse3.MoveHighAndDuplicate(h));
        accM = Sse.Add(accM, Sse.MoveHighToLow(accM, accM));
        accM = Sse.AddScalar(accM, Sse3.MoveHighAndDuplicate(accM));
        return Sse.AddScalar(h, accM).ToScalar();
    }

    // single-row entry points (one prologue per row)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)] public static float DotA(byte* row, byte* y, int nb) => Core(row, y, nb, true, false, null, null);
    [MethodImpl(MethodImplOptions.AggressiveOptimization)] public static float DotB(byte* row, byte* y, int nb) { float* dm = stackalloc float[16]; uint* w = stackalloc uint[8]; return Core(row, y, nb, false, true, dm, w); }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)] public static float DotAB(byte* row, byte* y, int nb) { float* dm = stackalloc float[16]; uint* w = stackalloc uint[8]; return Core(row, y, nb, true, true, dm, w); }

    // row loop inside the kernel (one prologue per matrix)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Rows(byte* m, int rows, int rowBytes, byte* y, int nb, float* o) { for (int r = 0; r < rows; r++) o[r] = Core(m + (long)r * rowBytes, y, nb, false, false, null, null); }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RowsA(byte* m, int rows, int rowBytes, byte* y, int nb, float* o) { for (int r = 0; r < rows; r++) o[r] = Core(m + (long)r * rowBytes, y, nb, true, false, null, null); }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RowsAB(byte* m, int rows, int rowBytes, byte* y, int nb, float* o) { float* dm = stackalloc float[16]; uint* w = stackalloc uint[8]; for (int r = 0; r < rows; r++) o[r] = Core(m + (long)r * rowBytes, y, nb, true, true, dm, w); }
}
