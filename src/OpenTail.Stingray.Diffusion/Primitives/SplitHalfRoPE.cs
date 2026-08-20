using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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

    public static unsafe void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
    {
        int half = headDim / 2;
        fixed (float* pQk = qk, pCos = cos, pSin = sin)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* pCosTok = pCos + s * headDim;
                float* pSinTok = pSin + s * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = pQk + (s * numHeads + h) * headDim;
                    int d = 0;
                    if (Avx.IsSupported && Fma.IsSupported)
                    {
                        for (; d + 8 <= half; d += 8)
                        {
                            var x1 = Avx.LoadVector256(head + d);
                            var x2 = Avx.LoadVector256(head + half + d);
                            var c = Avx.LoadVector256(pCosTok + d);
                            var sRot = Avx.LoadVector256(pSinTok + d);

                            var r0 = Fma.MultiplySubtract(x1, c, Avx.Multiply(x2, sRot));
                            var r1 = Fma.MultiplyAdd(x1, sRot, Avx.Multiply(x2, c));

                            Avx.Store(head + d, r0);
                            Avx.Store(head + half + d, r1);
                        }
                    }
                    for (; d < half; d++)
                    {
                        float x1 = head[d];
                        float x2 = head[half + d];
                        float c = pCosTok[d];
                        float sRot = pSinTok[d];

                        head[d] = x1 * c - x2 * sRot;
                        head[half + d] = x1 * sRot + x2 * c;
                    }
                }
            }
        }
    }
}
