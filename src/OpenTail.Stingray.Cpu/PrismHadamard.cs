namespace OpenTail.Stingray.Cpu;

/// <summary>
/// The signed, normalized Sylvester-Walsh-Hadamard transforms that Bonsai2 (PrismML "PRISM") checkpoints
/// declare in <c>prism.hadamard.*</c> metadata. Per block of <c>blockSize</c> elements:
/// <list type="bullet">
/// <item>Forward (projection inputs): <c>y = H(D x) / sqrt(blockSize)</c>.</item>
/// <item>Inverse (embedding rows): <c>y = D (H x) / sqrt(blockSize)</c>.</item>
/// </list>
/// D is the explicit ±1 sign vector and H the unnormalized Hadamard matrix in natural (Sylvester) order,
/// H[i][j] = (-1)^popcount(i &amp; j). The normalized H is orthogonal and its own inverse, so Inverse(Forward(x))
/// returns x. Spec read from TensorSharp's port (BonsaiHadamardMetadata.cs, ggml_ops_bonsai.cpp, BSD-3) and
/// its model card; the publisher reference is PrismML-Eng/llama.cpp (branch <c>prism</c>).
/// </summary>
public static unsafe class PrismHadamard
{
    /// <summary>In-place unnormalized fast Walsh-Hadamard transform of each block, then scale.</summary>
    internal static void FwhtBlocks(float* x, int width, int blockSize, float scale)
    {
        for (int start = 0; start < width; start += blockSize)
        {
            float* b = x + start;
            for (int stride = 1; stride < blockSize; stride <<= 1)
                for (int group = 0; group < blockSize; group += stride << 1)
                    for (int i = 0; i < stride; i++)
                    {
                        float a = b[group + i], c = b[group + stride + i];
                        b[group + i] = a + c;
                        b[group + stride + i] = a - c;
                    }
            for (int i = 0; i < blockSize; i++) b[i] *= scale;
        }
    }

    /// <summary>
    /// In-place orthonormal Sylvester-Walsh-Hadamard transform on blocks of size <paramref name="blockSize"/>
    /// scaled by 1/sqrt(blockSize). Used by DSA / lightning-indexer (GLM-DSA, DeepSeek-V3.2, etc.).
    /// </summary>
    public static void ApplySylvesterHadamard(float* x, int count, int blockSize)
    {
        if (blockSize < 2 || (blockSize & (blockSize - 1)) != 0 || count % blockSize != 0)
            throw new ArgumentException($"Invalid Hadamard transform: count {count}, block {blockSize}.");
        FwhtBlocks(x, count, blockSize, 1f / MathF.Sqrt(blockSize));
    }

    private static void Validate(int width, int signsLength, int blockSize)
    {
        if (blockSize < 2 || (blockSize & (blockSize - 1)) != 0 || width % blockSize != 0 || signsLength != width)
            throw new ArgumentException($"Invalid PRISM transform: width {width}, signs {signsLength}, block {blockSize}.");
    }

    /// <summary>Forward transform for a projection input: x &lt;- H(D x)/sqrt(block).</summary>
    public static void Forward(float* x, ReadOnlySpan<float> signs, int width, int blockSize)
    {
        Validate(width, signs.Length, blockSize);
        for (int i = 0; i < width; i++) x[i] *= signs[i];
        FwhtBlocks(x, width, blockSize, 1f / MathF.Sqrt(blockSize));
    }

    /// <summary>Inverse transform for an embedding row: x &lt;- D (H x)/sqrt(block).</summary>
    public static void Inverse(float* x, ReadOnlySpan<float> signs, int width, int blockSize)
    {
        Validate(width, signs.Length, blockSize);
        FwhtBlocks(x, width, blockSize, 1f / MathF.Sqrt(blockSize));
        for (int i = 0; i < width; i++) x[i] *= signs[i];
    }

    /// <summary>
    /// Grouped-GDN head reorder applied to the ssm_out input before its forward transform when
    /// <c>prism.hadamard.gdn_v_grouped</c> is set. The input is in tiled value-head order
    /// (head h = r * kGroups + k, i.e. k = h % kGroups, as <see cref="GdnKernels.TileHeads"/> pairs
    /// heads); it is rewritten in the publisher's grouped order (head k * repeat + r). Mirrors
    /// ggml_ops_bonsai.cpp: reshape_4d(x, hd, nk, rep, n) then permute(0, 2, 1, 3).
    /// </summary>
    public static void GroupedHeadReorder(float* src, float* dst, int headDim, int kGroups, int repeat)
    {
        for (int r = 0; r < repeat; r++)
            for (int k = 0; k < kGroups; k++)
                new ReadOnlySpan<float>(src + ((long)r * kGroups + k) * headDim, headDim)
                    .CopyTo(new Span<float>(dst + ((long)k * repeat + r) * headDim, headDim));
    }
}
