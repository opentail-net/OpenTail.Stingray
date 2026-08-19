namespace OpenTail.Stingray.Diffusion.Primitives;

/// <summary>
/// Shared kernels for the "interleaved" RoPE layout used by Flux2 and Flux3: each axis writes
/// (cos,cos)/(sin,sin) pairs at even/odd offsets within its slice of the head dimension, and
/// apply rotates adjacent (d, d+half) pairs using the even-indexed table entries. Per-model
/// axis composition (image-index/y/x, t/y/x, t/freq) stays in each model's own RoPE file.
/// </summary>
internal static class InterleavedRoPE
{
    public static float[] ComputeInvFreqs(int dim, float theta)
    {
        int half = dim / 2;
        var invFreq = new float[half];
        for (int i = 0; i < half; i++)
        {
            invFreq[i] = (float)(1.0 / Math.Pow(theta, (2.0 * i) / dim));
        }
        return invFreq;
    }

    public static void FillAxisFreqs(Span<float> cosOut, Span<float> sinOut, int pos, ReadOnlySpan<float> invFreq)
    {
        int half = invFreq.Length;
        for (int i = 0; i < half; i++)
        {
            float angle = pos * invFreq[i];
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            cosOut[i * 2 + 0] = c;
            cosOut[i * 2 + 1] = c;
            sinOut[i * 2 + 0] = s;
            sinOut[i * 2 + 1] = s;
        }
    }

    public static void ApplyRoPE(Span<float> tensor, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int nTokens, int numHeads, int headDim)
    {
        int halfHead = headDim / 2;

        for (int i = 0; i < nTokens; i++)
        {
            var cosToken = cos.Slice(i * headDim, headDim);
            var sinToken = sin.Slice(i * headDim, headDim);

            for (int h = 0; h < numHeads; h++)
            {
                int offset = (i * numHeads + h) * headDim;
                var headSpan = tensor.Slice(offset, headDim);

                for (int d = 0; d < halfHead; d++)
                {
                    float x0 = headSpan[d];
                    float x1 = headSpan[d + halfHead];
                    float c = cosToken[d * 2];
                    float s = sinToken[d * 2];

                    headSpan[d] = x0 * c - x1 * s;
                    headSpan[d + halfHead] = x0 * s + x1 * c;
                }
            }
        }
    }
}
