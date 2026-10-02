namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Bonsai2 PRISM building blocks ("ported, not verified": these pin the arithmetic against independent
/// definitions, not the model against the publisher's reference).
/// <list type="bullet">
/// <item><see cref="PrismHadamard"/>: against an explicit dense Sylvester Hadamard matrix,
/// H[i][j] = (-1)^popcount(i &amp; j).</item>
/// <item><see cref="BonsaiQuant"/> PTQ1_0: decoding a test-side encoder written from the format
/// description (each byte stores ceil(base3 * 256 / 243)).</item>
/// <item>Transcode to Q2_0: the existing Q2_0 dequantizer must reproduce the direct decode bit for
/// bit.</item>
/// </list>
/// </summary>
public sealed unsafe class BonsaiPrismTests
{
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        return a;
    }

    private static float[] Signs(int n, int seed)
    {
        var r = new Random(seed); var s = new float[n];
        for (int i = 0; i < n; i++) s[i] = r.Next(2) == 0 ? -1f : 1f;
        return s;
    }

    private static double[] DenseHadamard(double[] x, int start, int block)
    {
        var y = new double[block];
        double scale = 1.0 / Math.Sqrt(block);
        for (int i = 0; i < block; i++)
        {
            double s = 0;
            for (int j = 0; j < block; j++)
                s += (System.Numerics.BitOperations.PopCount((uint)(i & j)) % 2 == 0 ? 1 : -1) * x[start + j];
            y[i] = s * scale;
        }
        return y;
    }

    [Theory]
    [InlineData(256, 128)]
    [InlineData(2048, 1024)]
    [InlineData(512, 64)]
    public void Forward_MatchesDenseHadamardOfSignedInput(int width, int block)
    {
        var x = Rand(width, width + block); var s = Signs(width, block);
        var y = (float[])x.Clone();
        fixed (float* p = y) PrismHadamard.Forward(p, s, width, block);
        var signed = new double[width];
        for (int i = 0; i < width; i++) signed[i] = x[i] * s[i];
        for (int b = 0; b < width; b += block)
        {
            var reference = DenseHadamard(signed, b, block);
            for (int i = 0; i < block; i++)
                Assert.True(Math.Abs(reference[i] - y[b + i]) < 2e-5, $"[{b + i}] {reference[i]} vs {y[b + i]}");
        }
    }

    [Theory]
    [InlineData(256, 128)]
    [InlineData(2048, 1024)]
    public void Inverse_MatchesSignsTimesDenseHadamard(int width, int block)
    {
        var x = Rand(width, 3 * width); var s = Signs(width, 5 * block);
        var y = (float[])x.Clone();
        fixed (float* p = y) PrismHadamard.Inverse(p, s, width, block);
        var xd = x.Select(v => (double)v).ToArray();
        for (int b = 0; b < width; b += block)
        {
            var h = DenseHadamard(xd, b, block);
            for (int i = 0; i < block; i++)
                Assert.True(Math.Abs(h[i] * s[b + i] - y[b + i]) < 2e-5);
        }
    }

    [Fact]
    public void InverseOfForward_IsIdentity()
    {
        const int width = 5120, block = 1024; // Bonsai2's hidden width and block
        var x = Rand(width, 9); var s = Signs(width, 10);
        var y = (float[])x.Clone();
        fixed (float* p = y) { PrismHadamard.Forward(p, s, width, block); PrismHadamard.Inverse(p, s, width, block); }
        for (int i = 0; i < width; i++) Assert.True(Math.Abs(x[i] - y[i]) < 1e-5, $"[{i}] {x[i]} vs {y[i]}");
    }

    [Fact]
    public void InvalidDimensions_Throw()
    {
        var x = new float[300]; var s = new float[300];
        Assert.Throws<ArgumentException>(() => { fixed (float* p = x) PrismHadamard.Forward(p, s, 300, 128); });
        Assert.Throws<ArgumentException>(() => { fixed (float* p = x) PrismHadamard.Forward(p, new float[256], 256, 96); });
    }

    [Fact]
    public void GroupedHeadReorder_MapsTiledToGroupedOrder()
    {
        const int hd = 2, nk = 3, rep = 2; // tiled head h = r*nk + k -> grouped head k*rep + r
        var src = new float[hd * nk * rep];
        for (int h = 0; h < nk * rep; h++) for (int d = 0; d < hd; d++) src[h * hd + d] = h * 10 + d;
        var dst = new float[src.Length];
        fixed (float* s = src) fixed (float* t = dst) PrismHadamard.GroupedHeadReorder(s, t, hd, nk, rep);
        for (int r = 0; r < rep; r++)
            for (int k = 0; k < nk; k++)
                for (int d = 0; d < hd; d++)
                    Assert.Equal((r * nk + k) * 10 + d, dst[(k * rep + r) * hd + d]);
    }

    // ── BonsaiQuant ──

    /// <summary>Test-side PTQ1_0 encoder from the format description: 16 lanes x 5 trits, 8 x 5, then 2 x 4;
    /// lane byte = ceil(n * 256 / 243) with n the base-3 number whose most significant digit decodes first.</summary>
    private static byte[] EncodePtq1(byte[] codes, ushort scaleBits)
    {
        var b = new byte[28];
        int o = 0;
        void Group(int offset, int lanes, int trits, int start)
        {
            for (int lane = 0; lane < lanes; lane++)
            {
                int n = 0;
                for (int p = 0; p < trits; p++) n += codes[start + p * lanes + lane] * (int)Math.Pow(3, 4 - p);
                b[offset + lane] = (byte)((n * 256 + 242) / 243);
            }
        }
        Group(0, 16, 5, o); o += 80;
        Group(16, 8, 5, o); o += 40;
        Group(24, 2, 4, o);
        b[26] = (byte)scaleBits; b[27] = (byte)(scaleBits >> 8);
        return b;
    }

    [Fact]
    public void Ptq1_DecodesTheReferenceEncoding()
    {
        var r = new Random(42);
        for (int trial = 0; trial < 200; trial++)
        {
            var codes = new byte[128];
            for (int i = 0; i < 128; i++) codes[i] = (byte)r.Next(3);
            ushort sb = BitConverter.HalfToUInt16Bits((Half)((r.NextDouble() - 0.5) * 0.1));
            var block = EncodePtq1(codes, sb);
            var got = new byte[128];
            fixed (byte* src = block) fixed (byte* c = got)
            {
                BonsaiQuant.UnpackBlock(DType.PTQ1_0, src, c, out ushort scale);
                Assert.Equal(sb, scale);
            }
            Assert.Equal(codes, got);
        }
    }

    [Fact]
    public void Pq2_DecodesSequentialTwoBitCodes()
    {
        var block = new byte[34];
        ushort sb = BitConverter.HalfToUInt16Bits((Half)0.25f);
        block[0] = (byte)sb; block[1] = (byte)(sb >> 8);
        var codes = new byte[128];
        for (int j = 0; j < 128; j++) { codes[j] = (byte)(j % 4); block[2 + j / 4] |= (byte)(codes[j] << (2 * (j % 4))); }
        var y = new float[128];
        Dequantize.ToFloat32(block, y, DType.PQ2_0, 128);
        for (int j = 0; j < 128; j++) Assert.Equal((codes[j] - 1) * 0.25f, y[j]);
    }

    [Theory]
    [InlineData(DType.PQ2_0)]
    [InlineData(DType.PTQ1_0)]
    public void TranscodeToQ2_0_IsLossless(DType dtype)
    {
        const int blocks = 64, n = blocks * 128;
        int stride = dtype == DType.PQ2_0 ? 34 : 28;
        var r = new Random((int)dtype);
        var src = new byte[blocks * stride];
        for (int b = 0; b < blocks; b++)
        {
            var codes = new byte[128];
            for (int i = 0; i < 128; i++) codes[i] = (byte)r.Next(dtype == DType.PQ2_0 ? 4 : 3);
            ushort sb = BitConverter.HalfToUInt16Bits((Half)((r.NextDouble() - 0.5) * 0.2));
            byte[] blk;
            if (dtype == DType.PTQ1_0) blk = EncodePtq1(codes, sb);
            else
            {
                blk = new byte[34]; blk[0] = (byte)sb; blk[1] = (byte)(sb >> 8);
                for (int j = 0; j < 128; j++) blk[2 + j / 4] |= (byte)(codes[j] << (2 * (j % 4)));
            }
            Buffer.BlockCopy(blk, 0, src, b * stride, stride);
        }
        var direct = new float[n];
        Dequantize.ToFloat32(src, direct, dtype, n);
        var q2 = new byte[BonsaiQuant.Q2_0Bytes(n)];
        fixed (byte* s = src) fixed (byte* d = q2) BonsaiQuant.TranscodeToQ2_0(dtype, s, n, d);
        var viaQ2 = new float[n];
        Dequantize.ToFloat32(q2, viaQ2, DType.Q2_0, n);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(direct[i]), BitConverter.SingleToInt32Bits(viaQ2[i]));
    }

    [Fact]
    public void GgufTypeIds_142_143_MapToBonsaiTypes()
    {
        Assert.True(DTypeInfo.TryFromGgufType(142, out var a)); Assert.Equal(DType.PQ2_0, a);
        Assert.True(DTypeInfo.TryFromGgufType(143, out var b)); Assert.Equal(DType.PTQ1_0, b);
        Assert.False(DTypeInfo.TryFromGgufType(141, out _));
        Assert.Equal(128, DTypeInfo.BlockSize(DType.PQ2_0)); Assert.Equal(34, DTypeInfo.BytesPerBlock(DType.PQ2_0));
        Assert.Equal(128, DTypeInfo.BlockSize(DType.PTQ1_0)); Assert.Equal(28, DTypeInfo.BytesPerBlock(DType.PTQ1_0));
    }
}
