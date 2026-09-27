namespace OpenTail.Stingray.Vision;

/// <summary>
/// Byte-exact ports of llama.cpp <c>tools/mtmd/mtmd-image.cpp</c> <c>img_tool</c> resizers used by several
/// preprocessors (Step3-VL, LLaVA-NeXT/Granite anyres). Images are interleaved RGB u8.
/// </summary>
internal static class LlamaImgTool
{
    /// <summary><c>img_tool::resize_bilinear</c>: align-corners mapping, result truncated to u8.</summary>
    public static byte[] ResizeBilinear(byte[] src, int sw, int sh, int tw, int th)
    {
        var dst = new byte[tw * th * 3];
        float xr = tw > 1 ? (float)(sw - 1) / (tw - 1) : 0f;
        float yr = th > 1 ? (float)(sh - 1) / (th - 1) : 0f;
        Parallel.For(0, th, y =>
        {
            float py = y * yr;
            int y0 = Math.Min((int)py, sh - 1), y1 = Math.Min(y0 + 1, sh - 1);
            float yf = py - y0;
            for (int x = 0; x < tw; x++)
            {
                float px = x * xr;
                int x0 = Math.Min((int)px, sw - 1), x1 = Math.Min(x0 + 1, sw - 1);
                float xf = px - x0;
                for (int c = 0; c < 3; c++)
                {
                    float top = Lerp(src[(y0 * sw + x0) * 3 + c], src[(y0 * sw + x1) * 3 + c], xf);
                    float bot = Lerp(src[(y1 * sw + x0) * 3 + c], src[(y1 * sw + x1) * 3 + c], xf);
                    dst[(y * tw + x) * 3 + c] = (byte)Lerp(top, bot, yf);
                }
            }
        });
        return dst;
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    /// <summary><c>img_tool::resize_bicubic</c>: source position <c>(int)(t*j)</c> (no half-pixel offset), cubic
    /// through 4 neighbours per axis with clamped edges, rounded and clamped to u8.</summary>
    public static byte[] ResizeBicubic(byte[] src, int nx, int ny, int tw, int th)
    {
        var dst = new byte[tw * th * 3];
        float tx = (float)nx / tw, ty = (float)ny / th;
        Parallel.For(0, th, i =>
        {
            Span<float> cc = stackalloc float[4];
            for (int j = 0; j < tw; j++)
            {
                int x = (int)(tx * j), y = (int)(ty * i);
                float dx = tx * j - x, dy = ty * i - y;
                for (int k = 0; k < 3; k++)
                {
                    for (int jj = 0; jj <= 3; jj++)
                    {
                        int row = Math.Clamp(y - 1 + jj, 0, ny - 1);
                        float P(int xo) => src[(row * nx + Math.Clamp(x + xo, 0, nx - 1)) * 3 + k];
                        float a0 = P(0);
                        float d0 = P(-1) - a0, d2 = P(1) - a0, d3 = P(2) - a0;
                        float a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
                        float a2 = 1.0f / 2 * d0 + 1.0f / 2 * d2;
                        float a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
                        cc[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;
                    }
                    float e0 = cc[0] - cc[1], e2 = cc[2] - cc[1], e3 = cc[3] - cc[1];
                    float b0 = cc[1];
                    float b1 = -1.0f / 3 * e0 + e2 - 1.0f / 6 * e3;
                    float b2 = 1.0f / 2 * e0 + 1.0f / 2 * e2;
                    float b3 = -1.0f / 6 * e0 - 1.0f / 2 * e2 + 1.0f / 6 * e3;
                    float v = b0 + b1 * dy + b2 * dy * dy + b3 * dy * dy * dy;
                    dst[(i * tw + j) * 3 + k] = (byte)Math.Min(Math.Max(MathF.Round(v, MidpointRounding.AwayFromZero), 0f), 255f);
                }
            }
        });
        return dst;
    }

    /// <summary><c>img_tool::resize</c> with <c>PAD_CEIL</c>: fit keeping aspect (ceil sizes), then centre on
    /// <paramref name="pad"/> (offset floor).</summary>
    public static byte[] ResizePadCeil(byte[] src, int sw, int sh, int tw, int th, Func<byte[], int, int, int, int, byte[]> resize, byte pad)
    {
        if (sw == tw && sh == th) return (byte[])src.Clone();
        float scale = Math.Min((float)tw / sw, (float)th / sh);
        int nw = Math.Min((int)MathF.Ceiling(sw * scale), tw), nh = Math.Min((int)MathF.Ceiling(sh * scale), th);
        byte[] r = resize(src, sw, sh, nw, nh);
        var dst = new byte[tw * th * 3];
        if (pad != 0) Array.Fill(dst, pad);
        int ox = (tw - nw) / 2, oy = (th - nh) / 2;
        for (int y = 0; y < nh; y++) Array.Copy(r, y * nw * 3, dst, ((y + oy) * tw + ox) * 3, nw * 3);
        return dst;
    }

    /// <summary>u8 RGB -> normalised CHW float ((x/255 - mean)/std).</summary>
    public static float[] ToChw(byte[] img, int w, int h, float[] mean, float[] std)
    {
        int plane = w * h;
        var chw = new float[3 * plane];
        for (int i = 0; i < plane; i++)
            for (int c = 0; c < 3; c++)
                chw[c * plane + i] = (img[i * 3 + c] / 255f - mean[c]) / std[c];
        return chw;
    }
}
