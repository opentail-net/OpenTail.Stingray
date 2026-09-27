namespace OpenTail.Stingray.Vision;

/// <summary>One view of a Step3-VL image: CHW pixels, already normalised, of a square <c>Size</c>.</summary>
public sealed record Step3VlView(float[] Chw, int Size);

/// <summary>Step3-VL views: row-major crops (each 504 px) on a <c>GridW x GridH</c> grid, then the 728-px overview.</summary>
public sealed record Step3VlPreprocessedImage(IReadOnlyList<Step3VlView> Crops, int GridW, int GridH, Step3VlView Overview);

/// <summary>
/// Step3-VL image preprocessing, ported from llama.cpp <c>mtmd_image_preprocessor_step3vl</c> (2026-09-27):
/// <list type="bullet">
/// <item>prepare: pad very small and very wide images square, then downscale so the long side is at most
///   <c>preproc_image_size</c> (3024), with u8 bilinear (<c>RESIZE_ALGO_BILINEAR</c>, align-corners, truncating);</item>
/// <item>window: none when the long side is at most <c>image_size</c> (728) and the aspect ratio is at most 1.5;
///   the short side when the long side is at most 728; else 504 (or the short side, if it is under 504 and the
///   aspect ratio exceeds 4);</item>
/// <item>the image is resized to a whole number of windows per axis (a fraction above 0.2 rounds up), and
///   windows are cut on a grid whose last start is pulled back to the edge;</item>
/// <item>each crop and the overview go through a half-pixel bilinear resize to 504 / 728 with normalisation.</item>
/// </list>
/// </summary>
public static class Step3VlImagePreprocessor
{
    public const int CropSize = 504;
    private const float SmallAspect = 1.5f, WideAspect = 4.0f, CropRounding = 0.2f;

    public static Step3VlPreprocessedImage Preprocess(
        ReadOnlySpan<byte> rgb, int width, int height,
        int imageSize, int longestEdge, float[] mean, float[] std)
    {
        (byte[] img, int w, int h) = Prepare(rgb.ToArray(), width, height, longestEdge);
        var overview = new Step3VlView(ResizeNormalize(img, w, h, imageSize, mean, std), imageSize);

        int window = WindowSize(imageSize, Math.Max(w, h), Math.Min(w, h));
        if (window <= 0) return new Step3VlPreprocessedImage([], 0, 0, overview);

        int cropW = CropExtent(w, window), cropH = CropExtent(h, window);
        byte[] refined = img;
        if (cropW != w || cropH != h) refined = LlamaImgTool.ResizeBilinear(img, w, h, cropW, cropH);
        int[] xs = Grid(cropW, window), ys = Grid(cropH, window);
        var crops = new List<Step3VlView>(xs.Length * ys.Length);
        foreach (int y in ys)
            foreach (int x in xs)
            {
                byte[] patch = CropBlackPadded(refined, cropW, cropH, x, y, window);
                crops.Add(new Step3VlView(ResizeNormalize(patch, window, window, CropSize, mean, std), CropSize));
            }
        return new Step3VlPreprocessedImage(crops, xs.Length, ys.Length, overview);
    }

    private static (byte[] Img, int W, int H) Prepare(byte[] img, int w, int h, int longestEdge)
    {
        float aspect = h > 0 ? (float)w / h : 1f;
        if (Math.Min(w, h) < 32 && (aspect > WideAspect || aspect < 1f / WideAspect))
        {
            int sq = Math.Max(w, h);
            var padded = new byte[sq * sq * 3];
            for (int y = 0; y < h; y++) Array.Copy(img, y * w * 3, padded, y * sq * 3, w * 3);
            (img, w, h) = (padded, sq, sq);
        }
        if (Math.Max(w, h) > longestEdge)
        {
            float scale = (float)longestEdge / Math.Max(w, h);
            int nw = Math.Max(1, (int)MathF.Floor(w * scale)), nh = Math.Max(1, (int)MathF.Floor(h * scale));
            img = LlamaImgTool.ResizeBilinear(img, w, h, nw, nh);
            (w, h) = (nw, nh);
        }
        return (img, w, h);
    }

    private static int WindowSize(int imageSize, int longer, int shorter)
    {
        float aspect = (float)longer / shorter;
        if (longer <= imageSize) return aspect > SmallAspect ? shorter : 0;
        return aspect > WideAspect ? Math.Min(shorter, CropSize) : CropSize;
    }

    private static int CropExtent(int length, int window)
    {
        float ratio = (float)length / window;
        if (ratio < 1f) return length;
        float dec = ratio - MathF.Floor(ratio);
        int rounded = dec > CropRounding ? (int)MathF.Floor(ratio) + 1 : (int)MathF.Floor(ratio);
        return window * rounded;
    }

    private static int[] Grid(int length, int window)
    {
        int n = length <= window ? 1 : (int)MathF.Ceiling((float)(length - window) / window + 1f);
        var starts = new int[n];
        for (int i = 0; i < n; i++) starts[i] = window * i;
        if (n > 1 && starts[^1] + window > length) starts[^1] = length - window;
        return starts;
    }

    private static byte[] CropBlackPadded(byte[] img, int w, int h, int x, int y, int size)
    {
        var dst = new byte[size * size * 3];
        int x0 = Math.Max(0, x), y0 = Math.Max(0, y), x1 = Math.Min(w, x + size), y1 = Math.Min(h, y + size);
        for (int yy = y0; yy < y1; yy++)
            Array.Copy(img, (yy * w + x0) * 3, dst, ((yy - y) * size + (x0 - x)) * 3, (x1 - x0) * 3);
        return dst;
    }

    /// <summary>img_u8_resize_bilinear_to_f32: half-pixel bilinear on normalised values, CHW output.</summary>
    private static float[] ResizeNormalize(byte[] src, int sw, int sh, int size, float[] mean, float[] std)
    {
        int plane = size * size;
        var chw = new float[3 * plane];
        float sx = (float)sw / size, sy = (float)sh / size;
        Parallel.For(0, size, y =>
        {
            float fy = (y + 0.5f) * sy - 0.5f;
            int yf = (int)MathF.Floor(fy);
            int y0 = Math.Clamp(yf, 0, sh - 1), y1 = Math.Clamp(yf + 1, 0, sh - 1);
            float ly = fy - yf;
            for (int x = 0; x < size; x++)
            {
                float fx = (x + 0.5f) * sx - 0.5f;
                int xf = (int)MathF.Floor(fx);
                int x0 = Math.Clamp(xf, 0, sw - 1), x1 = Math.Clamp(xf + 1, 0, sw - 1);
                float lx = fx - xf;
                for (int c = 0; c < 3; c++)
                {
                    float N(int px, int py) => (src[(py * sw + px) * 3 + c] / 255f - mean[c]) / std[c];
                    float top = N(x0, y0) + (N(x1, y0) - N(x0, y0)) * lx;
                    float bot = N(x0, y1) + (N(x1, y1) - N(x0, y1)) * lx;
                    chw[c * plane + y * size + x] = top + (bot - top) * ly;
                }
            }
        });
        return chw;
    }
}
