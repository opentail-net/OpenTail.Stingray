using System;

namespace OpenTail.Stingray.Vision;

public sealed record MiniCpmPreprocessedSlice(float[] Chw, int Width, int Height, int GridX, int GridY);

/// <summary>
/// Image preprocessor for MiniCPM-V 2.6 HD image slicing and normalization.
/// </summary>
public static class MiniCpmImagePreprocessor
{
    private static readonly float[] Mean = { 0.5f, 0.5f, 0.5f };
    private static readonly float[] Std = { 0.5f, 0.5f, 0.5f };

    public static MiniCpmPreprocessedSlice[] Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int imageSize = 448, int maxSlices = 9)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException("Width and height must be positive.");

        // 1. Always create the global thumbnail slice (448x448)
        var thumbChw = ResizeAndNormalize(rgb, width, height, imageSize, imageSize);
        var thumbnail = new MiniCpmPreprocessedSlice(thumbChw, imageSize, imageSize, 0, 0);

        // If image is small or single-tile, return just the thumbnail
        if (width <= imageSize && height <= imageSize)
        {
            return new[] { thumbnail };
        }

        // 2. Compute best grid for HD slicing (e.g., 2x2, 1x2, 3x1)
        (int gridX, int gridY) = FindBestGrid(width, height, imageSize, maxSlices);
        int totalSlices = gridX * gridY;

        var slices = new MiniCpmPreprocessedSlice[totalSlices + 1];
        slices[0] = thumbnail;

        int tileTargetW = width / gridX;
        int tileTargetH = height / gridY;

        int sliceIdx = 1;
        for (int gy = 0; gy < gridY; gy++)
        {
            for (int gx = 0; gx < gridX; gx++)
            {
                int cropX = gx * tileTargetW;
                int cropY = gy * tileTargetH;
                int cropW = Math.Min(tileTargetW, width - cropX);
                int cropH = Math.Min(tileTargetH, height - cropY);

                var croppedRgb = Crop(rgb, width, height, cropX, cropY, cropW, cropH);
                var sliceChw = ResizeAndNormalize(croppedRgb, cropW, cropH, imageSize, imageSize);

                slices[sliceIdx++] = new MiniCpmPreprocessedSlice(sliceChw, imageSize, imageSize, gx, gy);
            }
        }

        return slices;
    }

    private static (int GridX, int GridY) FindBestGrid(int width, int height, int imageSize, int maxSlices)
    {
        double aspect = (double)width / height;
        int bestX = 1, bestY = 1;
        double bestError = double.MaxValue;

        for (int x = 1; x <= maxSlices; x++)
        {
            for (int y = 1; y <= maxSlices; y++)
            {
                if (x * y > maxSlices) continue;
                double gridAspect = (double)x / y;
                double error = Math.Abs(aspect - gridAspect);
                if (error < bestError)
                {
                    bestError = error;
                    bestX = x;
                    bestY = y;
                }
            }
        }
        return (bestX, bestY);
    }

    private static byte[] Crop(ReadOnlySpan<byte> src, int srcW, int srcH, int cropX, int cropY, int cropW, int cropH)
    {
        var dst = new byte[cropW * cropH * 3];
        for (int y = 0; y < cropH; y++)
        {
            int srcRow = (cropY + y) * srcW * 3;
            int dstRow = y * cropW * 3;
            for (int x = 0; x < cropW * 3; x++)
            {
                dst[dstRow + x] = src[srcRow + cropX * 3 + x];
            }
        }
        return dst;
    }

    public static float[] ResizeAndNormalize(ReadOnlySpan<byte> rgb, int srcW, int srcH, int dstW, int dstH)
    {
        var chw = new float[3 * dstW * dstH];
        int planeSize = dstW * dstH;

        float xRatio = (float)srcW / dstW;
        float yRatio = (float)srcH / dstH;

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = (dy + 0.5f) * yRatio - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(sy), 0, srcH - 1);
            int y1 = Math.Clamp(y0 + 1, 0, srcH - 1);
            float wy1 = sy - y0;
            float wy0 = 1.0f - wy1;

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = (dx + 0.5f) * xRatio - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(sx), 0, srcW - 1);
                int x1 = Math.Clamp(x0 + 1, 0, srcW - 1);
                float wx1 = sx - x0;
                float wx0 = 1.0f - wx1;

                int idx00 = (y0 * srcW + x0) * 3;
                int idx01 = (y0 * srcW + x1) * 3;
                int idx10 = (y1 * srcW + x0) * 3;
                int idx11 = (y1 * srcW + x1) * 3;

                int outIdx = dy * dstW + dx;

                for (int c = 0; c < 3; c++)
                {
                    float v00 = rgb[idx00 + c] / 255.0f;
                    float v01 = rgb[idx01 + c] / 255.0f;
                    float v10 = rgb[idx10 + c] / 255.0f;
                    float v11 = rgb[idx11 + c] / 255.0f;

                    float val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                    float norm = (val - Mean[c]) / Std[c];
                    chw[c * planeSize + outIdx] = norm;
                }
            }
        }
        return chw;
    }
}
