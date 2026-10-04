using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Two-row variant of <see cref="Q4KProto"/>: both rows are dotted against the same Q8_K activation in one pass, so the per-block activation work (yd, the per-16
/// sums reduction) is done once and the two rows' independent integer chains can overlap. Per-row results are bit-identical to <see cref="Q4KProto.Dot"/>.
/// </summary>
internal static unsafe class Q4KProto2
{
    private const float HalfRescale = 5.192296858534828e33f;   // 2^112

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Block(byte* x, sbyte* q8, float yd, Vector128<short> q8s, Vector256<byte> m4, ref Vector256<float> acc, ref Vector128<float> accM)
    {
        var hh = Sse41.ConvertToVector128Int32(Vector128.CreateScalarUnsafe(*(uint*)x).AsUInt16());
        var dd = Sse.Multiply(Sse2.Or(Sse2.ShiftLeftLogical(Sse2.And(hh, Vector128.Create(0x8000)), 16), Sse2.ShiftLeftLogical(Sse2.And(hh, Vector128.Create(0x7fff)), 13)).AsSingle(), Vector128.Create(HalfRescale));
        float d = yd * dd.ToScalar();
        float dmin = -yd * dd.GetElement(1);

        uint u0 = *(uint*)(x + 4), u1 = *(uint*)(x + 8), u2 = *(uint*)(x + 12);
        uint t3 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
        uint t2 = u1 & 0x3f3f3f3fu;
        uint t1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
        uint t0 = u0 & 0x3f3f3f3fu;
        var ms = Avx2.ConvertToVector256Int16(Vector128.Create(t0, t1, t2, t3).AsByte());
        var prod = Sse2.MultiplyAddAdjacent(ms.GetUpper(), q8s);
        accM = Fma.MultiplyAdd(Vector128.Create(dmin), Sse2.ConvertToVector128Single(prod), accM);

        var sc128 = ms.GetLower();
        var scalesB = Vector256.Create(sc128, sc128).AsByte();
        byte* q4 = x + 16;
        var sumi = Vector256<int>.Zero;
        sumi = Avx2.Add(sumi, Q4KProto3.Chunk(q4, q8, scalesB, m4, 0x0100, 0x0302));
        sumi = Avx2.Add(sumi, Q4KProto3.Chunk(q4 + 32, q8 + 64, scalesB, m4, 0x0504, 0x0706));
        sumi = Avx2.Add(sumi, Q4KProto3.Chunk(q4 + 64, q8 + 128, scalesB, m4, 0x0908, 0x0B0A));
        sumi = Avx2.Add(sumi, Q4KProto3.Chunk(q4 + 96, q8 + 192, scalesB, m4, 0x0D0C, 0x0F0E));
        acc = Fma.MultiplyAdd(Vector256.Create(d), Avx.ConvertToVector256Single(sumi), acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Reduce(Vector256<float> acc, Vector128<float> accM)
    {
        var h = Sse.Add(acc.GetLower(), acc.GetUpper());
        h = Sse.Add(h, Sse.MoveHighToLow(h, h));
        h = Sse.AddScalar(h, Sse3.MoveHighAndDuplicate(h));
        accM = Sse.Add(accM, Sse.MoveHighToLow(accM, accM));
        accM = Sse.AddScalar(accM, Sse3.MoveHighAndDuplicate(accM));
        return Sse.AddScalar(h, accM).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Dot2(byte* row0, byte* row1, byte* y, int numBlocks, out float r0, out float r1)
    {
        var m4 = Vector256.Create((byte)0x0F);
        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = Vector256<float>.Zero;
        Vector128<float> accM0 = Vector128<float>.Zero, accM1 = Vector128<float>.Zero;
        for (int i = 0; i < numBlocks; i++, row0 += 144, row1 += 144, y += 292)
        {
            float yd = *(float*)y;
            var sums = Avx.LoadVector256((short*)(y + 260));
            var q8s = Ssse3.HorizontalAdd(sums.GetLower(), sums.GetUpper());
            Block(row0, (sbyte*)(y + 4), yd, q8s, m4, ref acc0, ref accM0);
            Block(row1, (sbyte*)(y + 4), yd, q8s, m4, ref acc1, ref accM1);
        }
        r0 = Reduce(acc0, accM0);
        r1 = Reduce(acc1, accM1);
    }
}
