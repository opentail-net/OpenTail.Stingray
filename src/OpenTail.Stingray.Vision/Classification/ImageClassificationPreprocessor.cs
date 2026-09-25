namespace OpenTail.Stingray.Vision.Classification;

/// <summary>
/// The standard timm / torchvision eval transform: resize the shorter side to <c>floor(inputSize / cropPct)</c> with a
/// PIL-style antialiased filter (bicubic a = -0.5 or bilinear, support widened by the downscale factor), round back to
/// 8-bit as PIL does, center-crop <c>inputSize</c> (offsets <c>round((dim - size) / 2)</c>), then <c>(x/255 - mean) / std</c>.
/// Output is planar CHW float.
/// </summary>
public static class ImageClassificationPreprocessor
{
    public static float[] ResizeCenterCropNormalize(ReadOnlySpan<byte> rgb, int width, int height, int inputSize, float cropPct,
        ReadOnlySpan<float> mean, ReadOnlySpan<float> std, bool bicubic = true)
    {
        int scaleSize = (int)Math.Floor(inputSize / cropPct);
        int newW, newH;
        if (width <= height) { newW = scaleSize; newH = (int)((long)scaleSize * height / width); }
        else { newH = scaleSize; newW = (int)((long)scaleSize * width / height); }

        var resized = Resize(rgb, width, height, newW, newH, bicubic);
        int top = (int)Math.Round((newH - inputSize) / 2.0, MidpointRounding.ToEven);
        int left = (int)Math.Round((newW - inputSize) / 2.0, MidpointRounding.ToEven);
        int plane = inputSize * inputSize;
        var chw = new float[3 * plane];
        for (int y = 0; y < inputSize; y++)
            for (int x = 0; x < inputSize; x++)
                for (int c = 0; c < 3; c++)
                    chw[c * plane + y * inputSize + x] = (resized[((top + y) * newW + left + x) * 3 + c] / 255f - mean[c]) / std[c];
        return chw;
    }

    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0;
    }

    private static double Linear(double x)
    {
        x = Math.Abs(x);
        return x < 1 ? 1 - x : 0;
    }

    /// <summary>PIL <c>ImagingResample</c>: separable, horizontal then vertical, each pass rounded to 8-bit.</summary>
    public static byte[] Resize(ReadOnlySpan<byte> rgb, int w, int h, int newW, int newH, bool bicubic = true)
    {
        var tmp = new byte[newW * h * 3];
        var (hx, hw, hs) = Coefficients(w, newW, bicubic);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < newW; x++)
                for (int c = 0; c < 3; c++)
                {
                    double s = 0;
                    for (int k = 0; k < hs[x]; k++) s += rgb[(y * w + hx[x] + k) * 3 + c] * hw[x][k];
                    tmp[(y * newW + x) * 3 + c] = Clip8(s);
                }
        var outp = new byte[newW * newH * 3];
        var (vy, vw, vs) = Coefficients(h, newH, bicubic);
        for (int y = 0; y < newH; y++)
            for (int x = 0; x < newW; x++)
                for (int c = 0; c < 3; c++)
                {
                    double s = 0;
                    for (int k = 0; k < vs[y]; k++) s += tmp[((vy[y] + k) * newW + x) * 3 + c] * vw[y][k];
                    outp[(y * newW + x) * 3 + c] = Clip8(s);
                }
        return outp;
    }

    private static byte Clip8(double v) => (byte)Math.Clamp((int)Math.Round(v, MidpointRounding.AwayFromZero), 0, 255);

    private static (int[] Start, double[][] Weights, int[] Size) Coefficients(int inSize, int outSize, bool bicubic)
    {
        double scale = (double)inSize / outSize, filterScale = Math.Max(scale, 1.0);
        double support = (bicubic ? 2.0 : 1.0) * filterScale;
        var start = new int[outSize];
        var weights = new double[outSize][];
        var size = new int[outSize];
        for (int i = 0; i < outSize; i++)
        {
            double center = (i + 0.5) * scale;
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inSize);
            var w = new double[xmax - xmin];
            double total = 0;
            for (int x = xmin; x < xmax; x++)
            {
                double arg = (x - center + 0.5) / filterScale;
                total += w[x - xmin] = bicubic ? Cubic(arg) : Linear(arg);
            }
            if (total != 0) for (int k = 0; k < w.Length; k++) w[k] /= total;
            (start[i], weights[i], size[i]) = (xmin, w, w.Length);
        }
        return (start, weights, size);
    }
}
