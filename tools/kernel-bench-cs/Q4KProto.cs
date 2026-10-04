using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Prototype Q4_K x Q8_K row dot in ggml's structure (ggml_vec_dot_q4_K_q8_K, AVX2 path), written in C# intrinsics: scales/mins unpacked once per super-block with
/// scalar bit ops, the 8 scales broadcast as 16-bit lanes by byte shuffle, maddubs + madd against the scales accumulating in INTEGER across the super-block,
/// the mins folded in through the activation's per-16 sums (bsums), and exactly one float conversion + FMA per super-block. Q8_K layout (ggml block_q8_K,
/// 292 bytes): float d; sbyte qs[256]; short bsums[16].
/// </summary>
internal static unsafe class Q4KProto
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Dot(byte* row, byte* y, int numBlocks)
    {
        var m4 = Vector256.Create((byte)0x0F);
        Vector256<float> acc = Vector256<float>.Zero;
        Vector128<float> accM = Vector128<float>.Zero;
        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 144;
            byte* yb = y + i * 292;
            float yd = *(float*)yb;
            sbyte* q8 = (sbyte*)(yb + 4);
            short* bsums = (short*)(yb + 260);
            float d = yd * (float)BitConverter.UInt16BitsToHalf(*(ushort*)x);
            float dmin = -yd * (float)BitConverter.UInt16BitsToHalf(*(ushort*)(x + 2));

            uint u0 = *(uint*)(x + 4), u1 = *(uint*)(x + 8), u2 = *(uint*)(x + 12);
            uint t3 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
            uint t2 = u1 & 0x3f3f3f3fu;
            uint t1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
            uint t0 = u0 & 0x3f3f3f3fu;
            var ms = Avx2.ConvertToVector256Int16(Vector128.Create(t0, t1, t2, t3).AsByte());   // lower 8: scales, upper 8: mins

            var sums = Avx.LoadVector256(bsums);
            var q8s = Ssse3.HorizontalAdd(sums.GetLower(), sums.GetUpper());                   // 8 sums of 32
            var prod = Sse2.MultiplyAddAdjacent(ms.GetUpper(), q8s);                           // mins . sums -> 4 x i32
            accM = Fma.MultiplyAdd(Vector128.Create(dmin), Sse2.ConvertToVector128Single(prod), accM);

            var sc128 = ms.GetLower();
            var scalesB = Vector256.Create(sc128, sc128).AsByte();
            byte* q4 = x + 16;
            var sumi = Vector256<int>.Zero;

            // 4 chunks of 64 weights: low nibbles use scale 2j, high nibbles scale 2j+1.
            {
                var qb = Avx.LoadVector256(q4);
                var pl = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(qb, m4), Avx.LoadVector256(q8)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0100).AsByte()).AsInt16());
                var ph = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(Avx2.ShiftRightLogical(qb.AsInt16(), 4).AsByte(), m4), Avx.LoadVector256(q8 + 32)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0302).AsByte()).AsInt16());
                sumi = Avx2.Add(sumi, Avx2.Add(pl, ph));
            }
            {
                var qb = Avx.LoadVector256(q4 + 32);
                var pl = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(qb, m4), Avx.LoadVector256(q8 + 64)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0504).AsByte()).AsInt16());
                var ph = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(Avx2.ShiftRightLogical(qb.AsInt16(), 4).AsByte(), m4), Avx.LoadVector256(q8 + 96)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0706).AsByte()).AsInt16());
                sumi = Avx2.Add(sumi, Avx2.Add(pl, ph));
            }
            {
                var qb = Avx.LoadVector256(q4 + 64);
                var pl = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(qb, m4), Avx.LoadVector256(q8 + 128)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0908).AsByte()).AsInt16());
                var ph = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(Avx2.ShiftRightLogical(qb.AsInt16(), 4).AsByte(), m4), Avx.LoadVector256(q8 + 160)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0B0A).AsByte()).AsInt16());
                sumi = Avx2.Add(sumi, Avx2.Add(pl, ph));
            }
            {
                var qb = Avx.LoadVector256(q4 + 96);
                var pl = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(qb, m4), Avx.LoadVector256(q8 + 192)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0D0C).AsByte()).AsInt16());
                var ph = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.And(Avx2.ShiftRightLogical(qb.AsInt16(), 4).AsByte(), m4), Avx.LoadVector256(q8 + 224)), Avx2.Shuffle(scalesB, Vector256.Create((ushort)0x0F0E).AsByte()).AsInt16());
                sumi = Avx2.Add(sumi, Avx2.Add(pl, ph));
            }
            acc = Fma.MultiplyAdd(Vector256.Create(d), Avx.ConvertToVector256Single(sumi), acc);
        }
        // horizontal sums: mins term (4 lanes) + main accumulator (8 lanes)
        float m = Vector128.Sum(accM);
        return Vector256.Sum(acc) + m;
    }
}

/// <summary>Accuracy of the three Q4_K dots against an exact double-precision dot of the dequantised weights and the original float activation.</summary>
internal static unsafe class Q4KAccuracy
{
    public static void Report(nint oursDot, nint protoDot, Func<nint, nint, int, float> unused) { }

    public static double ExactDot(byte* row, float* x, int nb)
    {
        double sum = 0; uint* scBuf = stackalloc uint[4];
        for (int i = 0; i < nb; i++)
        {
            byte* b = row + i * 144;
            double d = (double)BitConverter.UInt16BitsToHalf(*(ushort*)b), dmin = (double)BitConverter.UInt16BitsToHalf(*(ushort*)(b + 2));
            uint u0 = *(uint*)(b + 4), u1 = *(uint*)(b + 8), u2 = *(uint*)(b + 12);
            uint[] t = { u0 & 0x3f3f3f3fu, (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4), u1 & 0x3f3f3f3fu, ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4) };
            byte* sc = (byte*)scBuf; for (int k = 0; k < 4; k++) ((uint*)sc)[k] = t[k];
            byte* q = b + 16;
            for (int chunk = 0; chunk < 4; chunk++)
                for (int half = 0; half < 2; half++)
                {
                    int s = 2 * chunk + half;
                    double scale = d * sc[s], min = dmin * sc[8 + s];
                    for (int l = 0; l < 32; l++)
                    {
                        int nib = half == 0 ? (q[chunk * 32 + l] & 0xF) : (q[chunk * 32 + l] >> 4);
                        sum += (scale * nib - min) * x[i * 256 + s * 32 + l];
                    }
                }
        }
        return sum;
    }
}
