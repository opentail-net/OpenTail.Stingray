namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// 3D Rotary Positional Embedding (3D-RoPE) for Qwen Image.
/// Decomposes 128 head dimension into 3 axes: {16, 56, 56} for (time, height, width) with theta = 10000.
/// Reference: stable-diffusion.cpp:src/model/diffusion/qwen_image.hpp:axes_dim
/// </summary>
public static class QwenImageRoPE
{
    public static (float[] cos, float[] sin) Compute3DRoPE(
        int txtLen,
        int imgH,
        int imgW,
        int headDim = 128,
        float theta = 10000.0f)
    {
        // Axes dimensions: 16 (t), 56 (h), 56 (w)
        int dimT = 16;
        int dimH = 56;
        int dimW = 56;

        int imgLen = imgH * imgW;
        int totalLen = txtLen + imgLen;

        var cos = new float[totalLen * headDim];
        var sin = new float[totalLen * headDim];

        // 1. Text tokens: use 1D sequence positions for text [0..txtLen-1]
        for (int i = 0; i < txtLen; i++)
        {
            int off = i * headDim;
            FillFrequencies(cos, sin, off, pos: i, dim: headDim, theta: theta);
        }

        // 2. Image tokens: 3D coordinates (t=0, y=row, x=col)
        for (int row = 0; row < imgH; row++)
        {
            for (int col = 0; col < imgW; col++)
            {
                int tokenIdx = txtLen + row * imgW + col;
                int baseOff = tokenIdx * headDim;

                // Time axis: pos=0, dim=16
                FillFrequencies(cos, sin, baseOff + 0, pos: 0, dim: dimT, theta: theta);

                // Height axis: pos=row, dim=56
                FillFrequencies(cos, sin, baseOff + dimT, pos: row, dim: dimH, theta: theta);

                // Width axis: pos=col, dim=56
                FillFrequencies(cos, sin, baseOff + dimT + dimH, pos: col, dim: dimW, theta: theta);
            }
        }

        return (cos, sin);
    }

    private static void FillFrequencies(float[] cos, float[] sin, int offset, float pos, int dim, float theta)
    {
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Pow(theta, -2.0f * i / dim);
            float angle = pos * freq;
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            cos[offset + i] = c;
            cos[offset + half + i] = c;

            sin[offset + i] = s;
            sin[offset + half + i] = s;
        }
    }

    /// <summary>
    /// Applies 3D RoPE in-place to Q or K tensor [seqLen, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
    {
        int half = headDim / 2;
        for (int s = 0; s < seqLen; s++)
        {
            int peOff = s * headDim;
            for (int h = 0; h < numHeads; h++)
            {
                int headBase = (s * numHeads + h) * headDim;
                for (int d = 0; d < half; d++)
                {
                    float x1 = qk[headBase + d];
                    float x2 = qk[headBase + half + d];
                    float c = cos[peOff + d];
                    float sRot = sin[peOff + d];

                    qk[headBase + d] = x1 * c - x2 * sRot;
                    qk[headBase + half + d] = x1 * sRot + x2 * c;
                }
            }
        }
    }
}
