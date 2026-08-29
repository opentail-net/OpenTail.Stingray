
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// <see cref="SimdKernels.TransposeBlock8x8"/> replaces the scalar K-pack transpose in Flash-64
/// prefill. A transpose only moves floats, so the vector form must be <b>bit-identical</b> to the
/// scalar one — not close, identical. That is a stronger contract than most kernel swaps get, and
/// it is worth pinning cheaply here rather than relying solely on the model-backed
/// <c>TileJobs_MatchHeadJobs_BitExactly</c>, which takes minutes and needs a GGUF on disk.
/// </summary>
public sealed unsafe class TransposeBlock8x8Tests
{
    /// <summary>Values that are exactly representable and individually identifiable per cell.</summary>
    private static float Cell(int row, int col) => row * 100 + col;

    [Fact]
    public void TransposeBlock8x8_MatchesScalarTranspose_BitExactly()
    {
        Assert.SkipUnless(Avx.IsSupported, "AVX not supported on this host");

        // Source rows are deliberately NOT one contiguous block: in the real kernel each key row is
        // a separate pointer into the paged KV cache, so the kernel may not assume adjacency.
        const int srcWidth = 24;      // wider than the 8 columns read, to catch offset mistakes
        const int srcOffset = 8;      // read columns [8,16), not from the start
        const int dstRowStride = 64;  // the real Tile stride, wider than the 8 written

        var rows = new float[8][];
        var handles = new GCHandle[8];
        var rowPtrs = stackalloc float*[8];
        for (int r = 0; r < 8; r++)
        {
            rows[r] = new float[srcWidth];
            for (int c = 0; c < srcWidth; c++) rows[r][c] = Cell(r, c);
            handles[r] = GCHandle.Alloc(rows[r], GCHandleType.Pinned);
            rowPtrs[r] = (float*)handles[r].AddrOfPinnedObject();
        }

        var dst = new float[8 * dstRowStride];
        var expected = new float[8 * dstRowStride];
        Array.Fill(dst, float.NaN);
        Array.Fill(expected, float.NaN);

        // Scalar reference: exactly what the kernel's fallback path does.
        for (int d = 0; d < 8; d++)
            for (int j = 0; j < 8; j++)
                expected[d * dstRowStride + j] = rows[j][srcOffset + d];

        try
        {
            fixed (float* pDst = dst)
                SimdKernels.TransposeBlock8x8(rowPtrs, srcOffset, pDst, dstRowStride);
        }
        finally
        {
            for (int r = 0; r < 8; r++) handles[r].Free();
        }

        // BitConverter comparison, not an epsilon: this must be exact, and it also catches a NaN
        // written where an untouched cell should remain untouched.
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(dst[i]),
                $"index {i} (row {i / dstRowStride}, col {i % dstRowStride}): expected {expected[i]}, got {dst[i]}");
        }
    }

    /// <summary>
    /// The kernel must write only the 8×8 window it owns. The K-pack writes blocks side by side
    /// into a Tile-wide row, so a stray store past column 8 would silently corrupt the neighbouring
    /// block — which the GEMM would then consume as valid key data.
    /// </summary>
    [Fact]
    public void TransposeBlock8x8_WritesOnlyItsOwnWindow()
    {
        Assert.SkipUnless(Avx.IsSupported, "AVX not supported on this host");

        const int dstRowStride = 64;
        var rows = new float[8][];
        var handles = new GCHandle[8];
        var rowPtrs = stackalloc float*[8];
        for (int r = 0; r < 8; r++)
        {
            rows[r] = new float[8];
            for (int c = 0; c < 8; c++) rows[r][c] = Cell(r, c) + 0.5f;
            handles[r] = GCHandle.Alloc(rows[r], GCHandleType.Pinned);
            rowPtrs[r] = (float*)handles[r].AddrOfPinnedObject();
        }

        var dst = new float[8 * dstRowStride];
        const float sentinel = -12345.75f;
        Array.Fill(dst, sentinel);

        try
        {
            fixed (float* pDst = dst)
                SimdKernels.TransposeBlock8x8(rowPtrs, 0, pDst + 16, dstRowStride); // write at column 16
        }
        finally
        {
            for (int r = 0; r < 8; r++) handles[r].Free();
        }

        for (int d = 0; d < 8; d++)
        {
            for (int c = 0; c < dstRowStride; c++)
            {
                float actual = dst[d * dstRowStride + c];
                if (c >= 16 && c < 24)
                    Assert.Equal(rows[c - 16][d], actual);
                else
                    Assert.True(actual == sentinel, $"row {d} col {c} was overwritten: {actual}");
            }
        }
    }
}
