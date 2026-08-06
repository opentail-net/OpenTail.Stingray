using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Phase-2 attempt at a literal port of llama.cpp's real AVX2 GEMM kernel
/// (docs/cpu-prefill-repack-gemm-plan.md §32+) -- verified one isolated "seam" at a time
/// against a hand-computed scalar reference, since the source kernel is ~340 lines of
/// interdependent shuffle/blend immediates where a wrong constant produces silently-wrong
/// logits rather than a crash. Do not extend RealAvx2Gemm without a matching seam test.
/// </summary>
public sealed unsafe class RealAvx2GemmSeamTests
{
    /// <summary>
    /// Seam 1 (RearrangeColumnPairs0145_2367): builds one repacked kk-block (64 bytes: columns
    /// 0-3's 8-byte chunks then columns 4-7's, matching RepackedGemm.RepackQ4K8Rows's own
    /// layout) with each column's 8 bytes set to a distinct, recognizable value, then checks
    /// the two output vectors land the columns in the documented [0,1,4,5] / [2,3,6,7] order --
    /// computed directly from the column values, not by calling any other RepackedGemm code.
    /// </summary>
    [Fact]
    public void RearrangeColumnPairs0145_2367_MatchesHandComputedColumnOrder()
    {
        byte[] kkBlock = new byte[64];
        for (int col = 0; col < 8; col++)
            for (int i = 0; i < 8; i++)
                kkBlock[col * 8 + i] = (byte)(col * 10 + i); // col0: 0-7, col1: 10-17, ... col7: 70-77

        fixed (byte* p = kkBlock)
        {
            RealAvx2Gemm.RearrangeColumnPairs0145_2367(p, kk: 0, out var mat0145, out var mat2367);

            Span<byte> out0145 = stackalloc byte[32];
            Span<byte> out2367 = stackalloc byte[32];
            Vector256.StoreUnsafe(mat0145, ref out0145[0]);
            Vector256.StoreUnsafe(mat2367, ref out2367[0]);

            int[] expected0145Cols = [0, 1, 4, 5];
            int[] expected2367Cols = [2, 3, 6, 7];

            for (int slot = 0; slot < 4; slot++)
            {
                int col = expected0145Cols[slot];
                for (int i = 0; i < 8; i++)
                    Assert.Equal((byte)(col * 10 + i), out0145[slot * 8 + i]);
            }
            for (int slot = 0; slot < 4; slot++)
            {
                int col = expected2367Cols[slot];
                for (int i = 0; i < 8; i++)
                    Assert.Equal((byte)(col * 10 + i), out2367[slot * 8 + i]);
            }
        }
    }

    /// <summary>
    /// Seam 2 (UnpackNibbles): every byte's low/high nibble extracted independently, unsigned,
    /// against a plain scalar reference -- no dependency on seam 1 or the repacked layout.
    /// </summary>
    [Fact]
    public void UnpackNibbles_MatchesScalarNibbleExtraction()
    {
        byte[] packed = new byte[32];
        var rng = new Random(999);
        rng.NextBytes(packed);

        fixed (byte* p = packed)
        {
            var packedVec = Vector256.LoadUnsafe(ref *p);
            RealAvx2Gemm.UnpackNibbles(packedVec, out var lo, out var hi);

            Span<byte> loOut = stackalloc byte[32];
            Span<byte> hiOut = stackalloc byte[32];
            Vector256.StoreUnsafe(lo, ref loOut[0]);
            Vector256.StoreUnsafe(hi, ref hiOut[0]);

            for (int i = 0; i < 32; i++)
            {
                Assert.Equal((byte)(packed[i] & 0x0F), loOut[i]);
                Assert.Equal((byte)(packed[i] >> 4), hiOut[i]);
            }
        }
    }

    /// <summary>
    /// Seam 3 (DuplicateShuffleWeightPattern): each 128-bit lane holds [colA bytes0-3, colA
    /// bytes4-7, colB bytes0-3, colB bytes4-7] (a nibble-unpacked 2-column pair); verifies sp1
    /// duplicates the (0-3) dwords and sp2 duplicates the (4-7) dwords, per lane, against
    /// values read directly out of the input buffer -- no dependency on seams 1/2.
    /// </summary>
    [Fact]
    public void DuplicateShuffleWeightPattern_MatchesHandComputedDwordDuplication()
    {
        byte[] input = new byte[32];
        var rng = new Random(4242);
        rng.NextBytes(input);

        fixed (byte* p = input)
        {
            var vec = Vector256.LoadUnsafe(ref *p);
            RealAvx2Gemm.DuplicateShuffleWeightPattern(vec, out var sp1, out var sp2);

            Span<byte> sp1Out = stackalloc byte[32];
            Span<byte> sp2Out = stackalloc byte[32];
            Vector256.StoreUnsafe(sp1, ref sp1Out[0]);
            Vector256.StoreUnsafe(sp2, ref sp2Out[0]);

            // Per 128-bit lane (bytes [0..16) and [16..32)): dword0=colA(0-3), dword1=colA(4-7),
            // dword2=colB(0-3), dword3=colB(4-7). sp1 = [dword0,dword2,dword0,dword2] per lane.
            // sp2 = [dword1,dword3,dword1,dword3] per lane.
            for (int lane = 0; lane < 2; lane++)
            {
                int laneOff = lane * 16;
                byte[] colA0_3 = input.AsSpan(laneOff, 4).ToArray();
                byte[] colA4_7 = input.AsSpan(laneOff + 4, 4).ToArray();
                byte[] colB0_3 = input.AsSpan(laneOff + 8, 4).ToArray();
                byte[] colB4_7 = input.AsSpan(laneOff + 12, 4).ToArray();

                AssertDword(sp1Out, laneOff + 0, colA0_3);
                AssertDword(sp1Out, laneOff + 4, colB0_3);
                AssertDword(sp1Out, laneOff + 8, colA0_3);
                AssertDword(sp1Out, laneOff + 12, colB0_3);

                AssertDword(sp2Out, laneOff + 0, colA4_7);
                AssertDword(sp2Out, laneOff + 4, colB4_7);
                AssertDword(sp2Out, laneOff + 8, colA4_7);
                AssertDword(sp2Out, laneOff + 12, colB4_7);
            }
        }

        static void AssertDword(Span<byte> actual, int offset, byte[] expected)
        {
            for (int i = 0; i < 4; i++)
                Assert.Equal(expected[i], actual[offset + i]);
        }
    }

    /// <summary>
    /// Seam 4 (BroadcastLowHigh128): the low 16 bytes of a 32-byte input must appear in BOTH
    /// halves of broadcastLow, and the high 16 bytes in both halves of broadcastHigh -- checked
    /// against the input buffer directly, no dependency on any other seam.
    /// </summary>
    [Fact]
    public void BroadcastLowHigh128_MatchesHandComputedLaneBroadcast()
    {
        byte[] input = new byte[32];
        var rng = new Random(1357);
        rng.NextBytes(input);

        fixed (byte* p = input)
        {
            var vec = Vector256.LoadUnsafe(ref *p);
            RealAvx2Gemm.BroadcastLowHigh128(vec, out var low, out var high);

            Span<byte> lowOut = stackalloc byte[32];
            Span<byte> highOut = stackalloc byte[32];
            Vector256.StoreUnsafe(low, ref lowOut[0]);
            Vector256.StoreUnsafe(high, ref highOut[0]);

            for (int i = 0; i < 16; i++)
            {
                Assert.Equal(input[i], lowOut[i]);       // low half, first copy
                Assert.Equal(input[i], lowOut[i + 16]);  // low half, second copy
                Assert.Equal(input[16 + i], highOut[i]);      // high half, first copy
                Assert.Equal(input[16 + i], highOut[i + 16]); // high half, second copy
            }
        }
    }

    /// <summary>
    /// Seam 5 (DuplicateShuffleActivationPattern): per 128-bit lane, dword0=tokenA(0-3),
    /// dword1=tokenA(4-7), dword2=tokenB(0-3), dword3=tokenB(4-7). sp1 = [dword0,dword0,
    /// dword2,dword2] per lane (each token's LOW dword duplicated, not alternated like seam
    /// 3). sp2 = [dword1,dword1,dword3,dword3] (HIGH dword duplicated). Verified directly
    /// against the input buffer, no dependency on seams 1-4.
    /// </summary>
    [Fact]
    public void DuplicateShuffleActivationPattern_MatchesHandComputedTokenDuplication()
    {
        byte[] input = new byte[32];
        var rng = new Random(8642);
        rng.NextBytes(input);

        fixed (byte* p = input)
        {
            var vec = Vector256.LoadUnsafe(ref *p);
            RealAvx2Gemm.DuplicateShuffleActivationPattern(vec, out var sp1, out var sp2);

            Span<byte> sp1Out = stackalloc byte[32];
            Span<byte> sp2Out = stackalloc byte[32];
            Vector256.StoreUnsafe(sp1, ref sp1Out[0]);
            Vector256.StoreUnsafe(sp2, ref sp2Out[0]);

            for (int lane = 0; lane < 2; lane++)
            {
                int laneOff = lane * 16;
                byte[] tokenA0_3 = input.AsSpan(laneOff, 4).ToArray();
                byte[] tokenA4_7 = input.AsSpan(laneOff + 4, 4).ToArray();
                byte[] tokenB0_3 = input.AsSpan(laneOff + 8, 4).ToArray();
                byte[] tokenB4_7 = input.AsSpan(laneOff + 12, 4).ToArray();

                AssertDword2(sp1Out, laneOff + 0, tokenA0_3);
                AssertDword2(sp1Out, laneOff + 4, tokenA0_3);
                AssertDword2(sp1Out, laneOff + 8, tokenB0_3);
                AssertDword2(sp1Out, laneOff + 12, tokenB0_3);

                AssertDword2(sp2Out, laneOff + 0, tokenA4_7);
                AssertDword2(sp2Out, laneOff + 4, tokenA4_7);
                AssertDword2(sp2Out, laneOff + 8, tokenB4_7);
                AssertDword2(sp2Out, laneOff + 12, tokenB4_7);
            }
        }

        static void AssertDword2(Span<byte> actual, int offset, byte[] expected)
        {
            for (int i = 0; i < 4; i++)
                Assert.Equal(expected[i], actual[offset + i]);
        }
    }

    /// <summary>
    /// Seam 6 (MaddubsAccumulate4): plain scalar reference -- for each of the 16 int16 output
    /// lanes, sum 8 individual byte products (2 per maddubs call x 4 calls), matching
    /// maddubs_epi16's u8xs8-&gt;i16-pair semantics plus 3 nested add_epi16 calls. Modest byte
    /// magnitudes (weight nibbles 0-15, activation values -16..16) keep every partial sum well
    /// inside int16 range so there's no overflow-behavior ambiguity to also encode. No
    /// dependency on seams 1-5 or on any other RealAvx2Gemm code.
    /// </summary>
    [Fact]
    public void MaddubsAccumulate4_MatchesScalarDotProductSum()
    {
        var rng = new Random(2024);
        byte[][] rhs = new byte[4][];
        sbyte[][] lhs = new sbyte[4][];
        for (int k = 0; k < 4; k++)
        {
            rhs[k] = new byte[32];
            lhs[k] = new sbyte[32];
            for (int i = 0; i < 32; i++)
            {
                rhs[k][i] = (byte)rng.Next(0, 16);       // unsigned weight nibble range
                lhs[k][i] = (sbyte)rng.Next(-16, 17);     // small signed activation range
            }
        }

        short[] expected = new short[16];
        for (int lane = 0; lane < 16; lane++)
        {
            int sum = 0;
            for (int k = 0; k < 4; k++)
            {
                sum += rhs[k][lane * 2] * lhs[k][lane * 2];
                sum += rhs[k][lane * 2 + 1] * lhs[k][lane * 2 + 1];
            }
            expected[lane] = (short)sum;
        }

        fixed (byte* r0 = rhs[0], r1 = rhs[1], r2 = rhs[2], r3 = rhs[3])
        fixed (sbyte* l0 = lhs[0], l1 = lhs[1], l2 = lhs[2], l3 = lhs[3])
        {
            var result = RealAvx2Gemm.MaddubsAccumulate4(
                Vector256.LoadUnsafe(ref *r0), Vector256.LoadUnsafe(ref *l0),
                Vector256.LoadUnsafe(ref *r1), Vector256.LoadUnsafe(ref *l1),
                Vector256.LoadUnsafe(ref *r2), Vector256.LoadUnsafe(ref *l2),
                Vector256.LoadUnsafe(ref *r3), Vector256.LoadUnsafe(ref *l3));

            Span<short> resultOut = stackalloc short[16];
            Vector256.StoreUnsafe(result, ref resultOut[0]);

            for (int lane = 0; lane < 16; lane++)
                Assert.Equal(expected[lane], resultOut[lane]);
        }
    }

    /// <summary>
    /// Seam 7 (ScaleAndReduce0145 / ScaleAndReduce2367): builds a 16-lane raw scale vector by
    /// hand (4 distinct dwords per 128-bit lane, each dword = 2 int16), independently shuffles
    /// it per the documented [0,1,0,1] / [2,3,2,3] dword patterns, then computes the expected
    /// madd_epi16 result as plain scalar adjacent-pair dot products -- entirely independent of
    /// RealAvx2Gemm's own shuffle helpers (seams 3/5 use different immediates for a different
    /// purpose and are not reused here).
    /// </summary>
    [Fact]
    public void ScaleAndReduce_MatchesScalarShuffleThenAdjacentDotProduct()
    {
        var rng = new Random(3141);
        short[] acc = new short[16];
        short[] rawScales = new short[16];
        for (int i = 0; i < 16; i++)
        {
            acc[i] = (short)rng.Next(-500, 501);
            rawScales[i] = (short)rng.Next(0, 64); // scale bytes are unsigned, zero-extended
        }

        // Per 128-bit lane (dwords 0-3, each dword = 2 int16 lanes): pattern [0,1,0,1] for 0145,
        // [2,3,2,3] for 2367.
        short[] ExpectedShuffled(int[] dwordPattern)
        {
            short[] outp = new short[16];
            for (int lane = 0; lane < 2; lane++)
            {
                int laneDwordOff = lane * 4;
                int laneShortOff = lane * 8;
                for (int slot = 0; slot < 4; slot++)
                {
                    int srcDword = laneDwordOff + dwordPattern[slot];
                    outp[laneShortOff + slot * 2] = rawScales[srcDword * 2];
                    outp[laneShortOff + slot * 2 + 1] = rawScales[srcDword * 2 + 1];
                }
            }
            return outp;
        }

        short[] scale0145 = ExpectedShuffled([0, 1, 0, 1]);
        short[] scale2367 = ExpectedShuffled([2, 3, 2, 3]);

        int[] Madd(short[] a, short[] b)
        {
            int[] outp = new int[8];
            for (int i = 0; i < 8; i++)
                outp[i] = a[2 * i] * b[2 * i] + a[2 * i + 1] * b[2 * i + 1];
            return outp;
        }

        int[] expected0145 = Madd(acc, scale0145);
        int[] expected2367 = Madd(acc, scale2367);

        fixed (short* accP = acc, scalesP = rawScales)
        {
            var accVec = Vector256.LoadUnsafe(ref *accP);
            var scalesVec = Vector256.LoadUnsafe(ref *scalesP);

            var result0145 = RealAvx2Gemm.ScaleAndReduce0145(accVec, scalesVec);
            var result2367 = RealAvx2Gemm.ScaleAndReduce2367(accVec, scalesVec);

            Span<int> out0145 = stackalloc int[8];
            Span<int> out2367 = stackalloc int[8];
            Vector256.StoreUnsafe(result0145, ref out0145[0]);
            Vector256.StoreUnsafe(result2367, ref out2367[0]);

            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(expected0145[i], out0145[i]);
                Assert.Equal(expected2367[i], out2367[i]);
            }
        }
    }

    /// <summary>
    /// Seam 8 (StraightenToRowVectors): builds mat00/mat01/mat10/mat11 with distinct
    /// recognizable int32 values per lane, computes the expected row0-3 vectors by hand-applying
    /// the documented [2,3,0,1] per-lane dword swap and the {2,3,6,7}-from-second-operand blend
    /// pattern directly on plain int arrays (no dependency on RealAvx2Gemm's other shuffle
    /// helpers, which use different immediates for different purposes).
    /// </summary>
    [Fact]
    public void StraightenToRowVectors_MatchesHandComputedSwapAndBlend()
    {
        var rng = new Random(5150);
        int[] mat00 = new int[8], mat01 = new int[8], mat10 = new int[8], mat11 = new int[8];
        for (int i = 0; i < 8; i++)
        {
            mat00[i] = rng.Next(-1000, 1000);
            mat01[i] = rng.Next(-1000, 1000);
            mat10[i] = rng.Next(-1000, 1000);
            mat11[i] = rng.Next(-1000, 1000);
        }

        int[] SwapLanes(int[] src)
        {
            int[] outp = new int[8];
            for (int lane = 0; lane < 2; lane++)
            {
                int off = lane * 4;
                int[] pattern = [2, 3, 0, 1];
                for (int slot = 0; slot < 4; slot++)
                    outp[off + slot] = src[off + pattern[slot]];
            }
            return outp;
        }

        // blend imm 204 = 0b11001100: dwords {2,3,6,7} from second operand, {0,1,4,5} from first.
        int[] Blend(int[] first, int[] second)
        {
            int[] outp = new int[8];
            int[] fromSecond = [2, 3, 6, 7];
            for (int i = 0; i < 8; i++)
                outp[i] = fromSecond.Contains(i) ? second[i] : first[i];
            return outp;
        }

        int[] mat00Swapped = SwapLanes(mat00);
        int[] mat01Swapped = SwapLanes(mat01);
        int[] mat10Swapped = SwapLanes(mat10);
        int[] mat11Swapped = SwapLanes(mat11);

        int[] expectedRow0 = Blend(mat00, mat01Swapped);
        int[] expectedRow1 = Blend(mat00Swapped, mat01);
        int[] expectedRow2 = Blend(mat10, mat11Swapped);
        int[] expectedRow3 = Blend(mat10Swapped, mat11);

        fixed (int* p00 = mat00, p01 = mat01, p10 = mat10, p11 = mat11)
        {
            RealAvx2Gemm.StraightenToRowVectors(
                Vector256.LoadUnsafe(ref *p00), Vector256.LoadUnsafe(ref *p01),
                Vector256.LoadUnsafe(ref *p10), Vector256.LoadUnsafe(ref *p11),
                out var row0, out var row1, out var row2, out var row3);

            Span<int> row0Out = stackalloc int[8];
            Span<int> row1Out = stackalloc int[8];
            Span<int> row2Out = stackalloc int[8];
            Span<int> row3Out = stackalloc int[8];
            Vector256.StoreUnsafe(row0, ref row0Out[0]);
            Vector256.StoreUnsafe(row1, ref row1Out[0]);
            Vector256.StoreUnsafe(row2, ref row2Out[0]);
            Vector256.StoreUnsafe(row3, ref row3Out[0]);

            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(expectedRow0[i], row0Out[i]);
                Assert.Equal(expectedRow1[i], row1Out[i]);
                Assert.Equal(expectedRow2[i], row2Out[i]);
                Assert.Equal(expectedRow3[i], row3Out[i]);
            }
        }
    }

    /// <summary>
    /// Seam 9a (BsumsMinCorrection): builds a 16-lane bsums-hsum vector with 4 distinct
    /// recognizable dword values per 128-bit lane, verifies each of the 4 broadcast immediates
    /// (0/85/170/255) replicates the documented dword index across the lane, then checks the
    /// madd_epi16 combine against a plain scalar adjacent-pair dot product -- independent of
    /// seam 7's ScaleAndReduce (different immediates, different broadcast semantics).
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void BsumsMinCorrection_MatchesScalarBroadcastThenDotProduct(int expectedDword)
    {
        var rng = new Random(9999 + expectedDword);
        short[] bsumsHsum = new short[16];
        short[] mins01 = new short[16];
        for (int i = 0; i < 16; i++)
        {
            bsumsHsum[i] = (short)rng.Next(-200, 201);
            mins01[i] = (short)rng.Next(0, 64);
        }

        // Broadcast: per 128-bit lane, all 4 dword slots take the value of dword `expectedDword`.
        short[] Broadcast()
        {
            short[] outp = new short[16];
            for (int lane = 0; lane < 2; lane++)
            {
                int laneShortOff = lane * 8;
                int srcShortOff = laneShortOff + expectedDword * 2;
                for (int slot = 0; slot < 4; slot++)
                {
                    outp[laneShortOff + slot * 2] = bsumsHsum[srcShortOff];
                    outp[laneShortOff + slot * 2 + 1] = bsumsHsum[srcShortOff + 1];
                }
            }
            return outp;
        }

        short[] broadcastExpected = Broadcast();
        int[] expected = new int[8];
        for (int i = 0; i < 8; i++)
            expected[i] = broadcastExpected[2 * i] * mins01[2 * i] + broadcastExpected[2 * i + 1] * mins01[2 * i + 1];

        fixed (short* bp = bsumsHsum, mp = mins01)
        {
            var result = RealAvx2Gemm.BsumsMinCorrection(Vector256.LoadUnsafe(ref *bp), expectedDword, Vector256.LoadUnsafe(ref *mp));
            Span<int> resultOut = stackalloc int[8];
            Vector256.StoreUnsafe(result, ref resultOut[0]);
            for (int i = 0; i < 8; i++)
                Assert.Equal(expected[i], resultOut[i]);
        }
    }

    /// <summary>
    /// Seam 9b (ScaleAndAccumulateRow): uses small exactly-representable-in-float32 integer
    /// values throughout (iacc rows, col scale/dmin, row scale) so the expected result can be
    /// computed with plain scalar float math without FMA-vs-separate-mul-add rounding ambiguity.
    /// Verifies the row-scale broadcast (imm 0/85/170/255, same mechanism as seam 9a but on
    /// float lanes), both fmadd accumulations, and the final subtraction -- independent of any
    /// other RealAvx2Gemm code.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ScaleAndAccumulateRow_MatchesScalarBroadcastFmaSubtract(int tokenIndex)
    {
        int[] iaccRow = [1, 2, 3, 4, 5, 6, 7, 8];
        int[] iaccRowMin = [10, 9, 8, 7, 6, 5, 4, 3];
        float[] colScale = [1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f];
        float[] colDmin = [2f, 1f, 4f, 3f, 2f, 1f, 4f, 3f];
        float[] rowScale = [10f, 20f, 30f, 40f]; // one d-value per token, low 128 = high 128 pattern not required here
        float[] accRowIn = [100f, 100f, 100f, 100f, 100f, 100f, 100f, 100f];
        float[] accMinRowIn = [50f, 50f, 50f, 50f, 50f, 50f, 50f, 50f];

        float tokenScale = rowScale[tokenIndex];
        float[] expectedNewAccRow = new float[8];
        float[] expectedNewAccMinRow = new float[8];
        float[] expectedFinal = new float[8];
        for (int i = 0; i < 8; i++)
        {
            expectedNewAccRow[i] = iaccRow[i] * (colScale[i] * tokenScale) + accRowIn[i];
            expectedNewAccMinRow[i] = iaccRowMin[i] * (colDmin[i] * tokenScale) + accMinRowIn[i];
            expectedFinal[i] = expectedNewAccRow[i] - expectedNewAccMinRow[i];
        }

        // rowScale vector needs both 128-bit lanes populated identically for the broadcast to
        // pick the right token regardless of which lane vshufps reads from -- lane 0 = lane 1.
        float[] rowScaleVec = [rowScale[0], rowScale[1], rowScale[2], rowScale[3], rowScale[0], rowScale[1], rowScale[2], rowScale[3]];

        fixed (int* irP = iaccRow, irmP = iaccRowMin)
        fixed (float* csP = colScale, cdP = colDmin, rsP = rowScaleVec, arP = accRowIn, amrP = accMinRowIn)
        {
            var result = RealAvx2Gemm.ScaleAndAccumulateRow(
                Vector256.LoadUnsafe(ref *irP), Vector256.LoadUnsafe(ref *irmP),
                Vector256.LoadUnsafe(ref *csP), Vector256.LoadUnsafe(ref *cdP), Vector256.LoadUnsafe(ref *rsP), tokenIndex,
                Vector256.LoadUnsafe(ref *arP), Vector256.LoadUnsafe(ref *amrP),
                out var newAccRow, out var newAccMinRow);

            Span<float> resultOut = stackalloc float[8];
            Span<float> newAccRowOut = stackalloc float[8];
            Span<float> newAccMinRowOut = stackalloc float[8];
            Vector256.StoreUnsafe(result, ref resultOut[0]);
            Vector256.StoreUnsafe(newAccRow, ref newAccRowOut[0]);
            Vector256.StoreUnsafe(newAccMinRow, ref newAccMinRowOut[0]);

            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(expectedNewAccRow[i], newAccRowOut[i], 3);
                Assert.Equal(expectedNewAccMinRow[i], newAccMinRowOut[i], 3);
                Assert.Equal(expectedFinal[i], resultOut[i], 3);
            }
        }
    }

    /// <summary>
    /// Composition-level "column identity" proof, ahead of writing the full composed kernel.
    /// Hand-tracing seams 1-8's shuffle/blend semantics together suggested the composed pipeline
    /// produces per-token output rows in NATURAL column order (0..7), with no extra permutation
    /// needed at the end -- despite <c>maddubs_epi16</c> merging adjacent byte pairs partway
    /// through, which made that non-obvious from reading any single seam in isolation. Rather
    /// than trust that algebra alone, this composes seams 1→2→3→6→7→8 end-to-end on synthetic
    /// data (column c's weight nibble = c+1 uniformly, token t's activation byte = t+1
    /// uniformly, unit scales) and checks the result against a first-principles expected value
    /// (32 elements/column/sub-block x columnValue x tokenValue) computed independently of any
    /// RealAvx2Gemm code -- proof, not just derivation.
    /// </summary>
    [Fact]
    public void ComposedSeams1Through8_ProduceNaturalColumnOrderPerToken()
    {
        // One kk-block (64 bytes): columns 0-7, each column's 8 bytes uniformly (c+1) in both
        // nibbles (so lo/hi nibble-unpack -- i.e. sub-block 0 vs 1 -- give identical per-column
        // values; this test only needs to confirm column IDENTITY, not sub-block separation,
        // which seam 2's own isolated test already covers).
        byte[] kkBlock = new byte[64];
        for (int col = 0; col < 8; col++)
        {
            byte nibbleVal = (byte)(col + 1); // 1..8, fits a nibble
            byte packed = (byte)(nibbleVal | (nibbleVal << 4));
            for (int i = 0; i < 8; i++)
                kkBlock[col * 8 + i] = packed;
        }

        fixed (byte* kkP = kkBlock)
        {
            RealAvx2Gemm.RearrangeColumnPairs0145_2367(kkP, kk: 0, out var mat0145, out var mat2367);
            RealAvx2Gemm.UnpackNibbles(mat0145, out var lo0145, out _);
            RealAvx2Gemm.UnpackNibbles(mat2367, out var lo2367, out _);
            RealAvx2Gemm.DuplicateShuffleWeightPattern(lo0145, out var rhs0145Sp1, out var rhs0145Sp2);
            RealAvx2Gemm.DuplicateShuffleWeightPattern(lo2367, out var rhs2367Sp1, out var rhs2367Sp2);

            Vector256<byte> BuildActivationBroadcast(byte tokenAVal, byte tokenBVal)
            {
                byte[] pairBytes = new byte[32];
                for (int i = 0; i < 8; i++) { pairBytes[i] = tokenAVal; pairBytes[8 + i] = tokenBVal; }
                fixed (byte* pb = pairBytes)
                {
                    var vec = Vector256.LoadUnsafe(ref *pb);
                    RealAvx2Gemm.BroadcastLowHigh128(vec, out var bcastLow, out _);
                    return bcastLow;
                }
            }

            var bcast01 = BuildActivationBroadcast(1, 2); // token0=1, token1=2
            var bcast23 = BuildActivationBroadcast(3, 4); // token2=3, token3=4
            RealAvx2Gemm.DuplicateShuffleActivationPattern(bcast01, out var lhsSp1_01, out var lhsSp2_01);
            RealAvx2Gemm.DuplicateShuffleActivationPattern(bcast23, out var lhsSp1_23, out var lhsSp2_23);

            var lhsSp1_01s = lhsSp1_01.AsSByte(); var lhsSp2_01s = lhsSp2_01.AsSByte();
            var lhsSp1_23s = lhsSp1_23.AsSByte(); var lhsSp2_23s = lhsSp2_23.AsSByte();

            // Same kk data reused 4x (MaddubsAccumulate4 combines 4 kk-positions) -- equivalent
            // to a uniform 4-kk chunk, since every kk carries identical synthetic data here.
            var iacc0145_01 = Avx2.Add(
                RealAvx2Gemm.MaddubsAccumulate4(rhs0145Sp1, lhsSp1_01s, rhs0145Sp1, lhsSp1_01s, rhs0145Sp1, lhsSp1_01s, rhs0145Sp1, lhsSp1_01s),
                RealAvx2Gemm.MaddubsAccumulate4(rhs0145Sp2, lhsSp2_01s, rhs0145Sp2, lhsSp2_01s, rhs0145Sp2, lhsSp2_01s, rhs0145Sp2, lhsSp2_01s));
            var iacc2367_01 = Avx2.Add(
                RealAvx2Gemm.MaddubsAccumulate4(rhs2367Sp1, lhsSp1_01s, rhs2367Sp1, lhsSp1_01s, rhs2367Sp1, lhsSp1_01s, rhs2367Sp1, lhsSp1_01s),
                RealAvx2Gemm.MaddubsAccumulate4(rhs2367Sp2, lhsSp2_01s, rhs2367Sp2, lhsSp2_01s, rhs2367Sp2, lhsSp2_01s, rhs2367Sp2, lhsSp2_01s));
            var iacc0145_23 = Avx2.Add(
                RealAvx2Gemm.MaddubsAccumulate4(rhs0145Sp1, lhsSp1_23s, rhs0145Sp1, lhsSp1_23s, rhs0145Sp1, lhsSp1_23s, rhs0145Sp1, lhsSp1_23s),
                RealAvx2Gemm.MaddubsAccumulate4(rhs0145Sp2, lhsSp2_23s, rhs0145Sp2, lhsSp2_23s, rhs0145Sp2, lhsSp2_23s, rhs0145Sp2, lhsSp2_23s));
            var iacc2367_23 = Avx2.Add(
                RealAvx2Gemm.MaddubsAccumulate4(rhs2367Sp1, lhsSp1_23s, rhs2367Sp1, lhsSp1_23s, rhs2367Sp1, lhsSp1_23s, rhs2367Sp1, lhsSp1_23s),
                RealAvx2Gemm.MaddubsAccumulate4(rhs2367Sp2, lhsSp2_23s, rhs2367Sp2, lhsSp2_23s, rhs2367Sp2, lhsSp2_23s, rhs2367Sp2, lhsSp2_23s));

            short[] onesArr = new short[16];
            for (int i = 0; i < 16; i++) onesArr[i] = 1;
            fixed (short* onesP = onesArr)
            {
                var rawScales = Vector256.LoadUnsafe(ref *onesP);

                var mat00 = RealAvx2Gemm.ScaleAndReduce0145(iacc0145_01, rawScales);
                var mat01 = RealAvx2Gemm.ScaleAndReduce2367(iacc2367_01, rawScales);
                var mat10 = RealAvx2Gemm.ScaleAndReduce0145(iacc0145_23, rawScales);
                var mat11 = RealAvx2Gemm.ScaleAndReduce2367(iacc2367_23, rawScales);

                RealAvx2Gemm.StraightenToRowVectors(mat00, mat01, mat10, mat11, out var row0, out var row1, out var row2, out var row3);

                Span<int> row0Out = stackalloc int[8], row1Out = stackalloc int[8], row2Out = stackalloc int[8], row3Out = stackalloc int[8];
                Vector256.StoreUnsafe(row0, ref row0Out[0]);
                Vector256.StoreUnsafe(row1, ref row1Out[0]);
                Vector256.StoreUnsafe(row2, ref row2Out[0]);
                Vector256.StoreUnsafe(row3, ref row3Out[0]);

                // First-principles expectation: 32 elements/column/sub-block, each contributing
                // columnValue x tokenValue, in NATURAL column order (0..7) -- if the composed
                // pipeline instead produced the [0,1,4,5,2,3,6,7]-rearranged order seam 1 uses
                // internally, this assertion would fail at columns 2-7.
                int[] tokenVals = [1, 2, 3, 4];
                void AssertRow(Span<int> row, int tokenVal)
                {
                    for (int col = 0; col < 8; col++)
                        Assert.Equal(32 * (col + 1) * tokenVal, row[col]);
                }
                AssertRow(row0Out, tokenVals[0]);
                AssertRow(row1Out, tokenVals[1]);
                AssertRow(row2Out, tokenVals[2]);
                AssertRow(row3Out, tokenVals[3]);
            }
        }
    }
}
