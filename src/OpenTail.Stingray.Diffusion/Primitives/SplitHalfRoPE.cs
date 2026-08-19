namespace OpenTail.Stingray.Diffusion.Primitives;

/// <summary>
/// Shared kernels for the "split-half" 3D-RoPE layout used by Wan, HunyuanVideo and QwenImage:
/// each axis writes cos/sin into its own contiguous slice of the head dimension, and apply
/// rotates the [0, half) / [half, dim) pair. Per-model axis-dim constants and token iteration
/// order stay in each model's own RoPE file.
/// </summary>
internal static class SplitHalfRoPE
{
    public static void FillFrequencies(float[] cos, float[] sin, int offset, float pos, int dim, float theta)
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
