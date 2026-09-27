namespace OpenTail.Stingray.Vision;

/// <summary>
/// DeepSeek-OCR2 global view, as llama.cpp's <c>mtmd_image_preprocessor_deepseekocr</c>: aspect-preserving
/// fit into a <c>1024 x 1024</c> canvas with Pillow-compatible bicubic resampling
/// (<c>RESIZE_ALGO_BICUBIC_PILLOW</c>), centred on a (127, 127, 127) gray pad (<c>PAD_NEAREST</c>), then
/// normalised with mean = std = 0.5 (<c>clip.vision.image_mean/std</c>). Output is CHW.
/// </summary>
/// <remarks>
/// llama.cpp additionally cuts 768-px tiles when either side exceeds 768 px; those are not produced here
/// yet (see docs/102 #7), so large pages are read from the global view only.
/// </remarks>
public static class DeepSeekOcr2ImagePreprocessor
{
    public const int BaseSize = 1024;
    private const byte PadValue = 127;

    public static float[] PreprocessGlobalView(ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (width <= 0 || height <= 0) throw new ArgumentException("Width and height must be positive.");
        float scale = Math.Min((float)BaseSize / width, (float)BaseSize / height);
        int newW = Math.Min((int)MathF.Round(width * scale), BaseSize);
        int newH = Math.Min((int)MathF.Round(height * scale), BaseSize);
        byte[] resized = (newW == width && newH == height) ? rgb.ToArray() : PillowResize.Bicubic(rgb, width, height, newW, newH);

        int offX = (int)MathF.Round((BaseSize - newW) / 2.0f);
        int offY = (int)MathF.Round((BaseSize - newH) / 2.0f);
        const int plane = BaseSize * BaseSize;
        var chw = new float[3 * plane];
        float padNorm = (PadValue / 255f - 0.5f) / 0.5f;
        Array.Fill(chw, padNorm);
        for (int y = 0; y < newH; y++)
            for (int x = 0; x < newW; x++)
            {
                int src = (y * newW + x) * 3;
                int dst = (y + offY) * BaseSize + (x + offX);
                for (int c = 0; c < 3; c++)
                    chw[c * plane + dst] = (resized[src + c] / 255f - 0.5f) / 0.5f;
            }
        return chw;
    }
}

/// <summary>
/// Pillow's <c>ImagingResample</c> bicubic (a = -0.5), fixed-point exactly as llama.cpp's
/// <c>img_tool::resize_pillow</c>: per-output weights normalised, scaled by 2^22 and rounded; a horizontal
/// pass then a vertical pass, each rounded and clipped to 8 bits. The filter widens when downscaling.
/// </summary>
internal static class PillowResize
{
    private const int PrecisionBits = 32 - 8 - 2;

    private static double Filter(double x)
    {
        const double a = -0.5;
        if (x < 0.0) x = -x;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    }

    private static int PrecomputeWeights(int inSize, int outSize, out int[] bounds, out int[] weights)
    {
        double scale = (double)inSize / outSize;
        double filterscale = Math.Max(scale, 1.0);
        double support = 2.0 * filterscale;
        int ksize = (int)Math.Ceiling(support) * 2 + 1;
        var pre = new double[outSize * ksize];
        bounds = new int[outSize * 2];
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            double ww = 0.0, ss = 1.0 / filterscale;
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inSize) - xmin;
            for (int x = 0; x < xmax; x++)
            {
                double w = Filter((x + xmin - center + 0.5) * ss);
                pre[xx * ksize + x] = w;
                ww += w;
            }
            if (ww != 0.0)
                for (int x = 0; x < xmax; x++) pre[xx * ksize + x] /= ww;
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        weights = new int[outSize * ksize];
        double fxp = Math.ScaleB(1.0, PrecisionBits);
        for (int i = 0; i < weights.Length; i++)
        {
            double t = pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5);
            weights[i] = (int)Math.Clamp(Math.Round(t), int.MinValue, int.MaxValue);
        }
        return ksize;
    }

    private static byte Clip8(long v)
    {
        v >>= PrecisionBits;
        return v < 0 ? (byte)0 : v > 255 ? (byte)255 : (byte)v;
    }

    public static byte[] Bicubic(ReadOnlySpan<byte> rgb, int inW, int inH, int outW, int outH)
    {
        byte[] src = rgb.ToArray();
        // Horizontal pass: inW x inH -> outW x inH.
        int kx = PrecomputeWeights(inW, outW, out var bx, out var wx);
        var tmp = new byte[outW * inH * 3];
        Parallel.For(0, inH, y =>
        {
            for (int xx = 0; xx < outW; xx++)
            {
                int xmin = bx[xx * 2], xcnt = bx[xx * 2 + 1];
                long s0 = 1L << (PrecisionBits - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xcnt; x++)
                {
                    int w = wx[xx * kx + x];
                    int p = (y * inW + x + xmin) * 3;
                    s0 += (long)src[p] * w; s1 += (long)src[p + 1] * w; s2 += (long)src[p + 2] * w;
                }
                int o = (y * outW + xx) * 3;
                tmp[o] = Clip8(s0); tmp[o + 1] = Clip8(s1); tmp[o + 2] = Clip8(s2);
            }
        });
        // Vertical pass: outW x inH -> outW x outH.
        int ky = PrecomputeWeights(inH, outH, out var by, out var wy);
        var dst = new byte[outW * outH * 3];
        Parallel.For(0, outH, yy =>
        {
            int ymin = by[yy * 2], ycnt = by[yy * 2 + 1];
            for (int x = 0; x < outW; x++)
            {
                long s0 = 1L << (PrecisionBits - 1), s1 = s0, s2 = s0;
                for (int y = 0; y < ycnt; y++)
                {
                    int w = wy[yy * ky + y];
                    int p = ((y + ymin) * outW + x) * 3;
                    s0 += (long)tmp[p] * w; s1 += (long)tmp[p + 1] * w; s2 += (long)tmp[p + 2] * w;
                }
                int o = (yy * outW + x) * 3;
                dst[o] = Clip8(s0); dst[o + 1] = Clip8(s1); dst[o + 2] = Clip8(s2);
            }
        });
        return dst;
    }
}
