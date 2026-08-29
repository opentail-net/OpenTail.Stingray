
namespace OpenTail.Stingray.Vision;

public sealed record Glm4PreprocessedImage(float[] Chw, int TargetWidth, int TargetHeight, int PatchesX, int PatchesY);

/// <summary>
/// Image preprocessor for Zhipu AI GLM-4V, GLM-4.5V, and GLM-OCR.
/// </summary>
public static class Glm4ImagePreprocessor
{
    private static readonly float[] Mean = { 0.48145466f, 0.4578275f, 0.40821073f };
    private static readonly float[] Std = { 0.26862954f, 0.26130258f, 0.27577711f };

    public static Glm4PreprocessedImage Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int patchSize = 14, int mergeFactor = 2, int maxDim = 1024)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException("Width and height must be positive.");

        int align = patchSize * mergeFactor; // 28
        double scale = Math.Min(1.0, (double)maxDim / Math.Max(width, height));
        int targetW = Math.Max(align, (int)Math.Round(width * scale / align) * align);
        int targetH = Math.Max(align, (int)Math.Round(height * scale / align) * align);

        int patchesX = targetW / patchSize;
        int patchesY = targetH / patchSize;

        var chw = new float[3 * targetW * targetH];
        int planeSize = targetW * targetH;

        float xRatio = (float)width / targetW;
        float yRatio = (float)height / targetH;

        for (int dy = 0; dy < targetH; dy++)
        {
            float sy = (dy + 0.5f) * yRatio - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(sy), 0, height - 1);
            int y1 = Math.Clamp(y0 + 1, 0, height - 1);
            float wy1 = sy - y0;
            float wy0 = 1.0f - wy1;

            for (int dx = 0; dx < targetW; dx++)
            {
                float sx = (dx + 0.5f) * xRatio - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(sx), 0, width - 1);
                int x1 = Math.Clamp(x0 + 1, 0, width - 1);
                float wx1 = sx - x0;
                float wx0 = 1.0f - wx1;

                int idx00 = (y0 * width + x0) * 3;
                int idx01 = (y0 * width + x1) * 3;
                int idx10 = (y1 * width + x0) * 3;
                int idx11 = (y1 * width + x1) * 3;

                int outIdx = dy * targetW + dx;

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

        return new Glm4PreprocessedImage(chw, targetW, targetH, patchesX, patchesY);
    }
}
