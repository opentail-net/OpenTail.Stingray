
namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// 3D Rotary Positional Embedding (3D-RoPE) for HunyuanVideo.
/// Decomposes 128 head dimension into 3 coordinate axes: {16, 56, 56} for (temporal frames t, height y, width x) with theta = 256.0 / 10000.0.
/// Reference: stable-diffusion.cpp:src/model/diffusion/hunyuan.hpp:axes_dim
/// </summary>
public static class HunyuanVideoRoPE
{
    public static (float[] cos, float[] sin) Compute3DRoPE(
        int numFrames,
        int patchH,
        int patchW,
        int headDim = 128,
        float theta = 256.0f)
    {
        int dimT = 16;
        int dimH = 56;
        int dimW = 56;

        int totalTokens = numFrames * patchH * patchW;
        var cos = new float[totalTokens * headDim];
        var sin = new float[totalTokens * headDim];

        for (int t = 0; t < numFrames; t++)
        {
            for (int y = 0; y < patchH; y++)
            {
                for (int x = 0; x < patchW; x++)
                {
                    int tokenIdx = (t * patchH + y) * patchW + x;
                    int baseOff = tokenIdx * headDim;

                    // Temporal axis: pos=t, dim=16
                    SplitHalfRoPE.FillFrequencies(cos, sin, baseOff + 0, pos: t, dim: dimT, theta: theta);

                    // Height axis: pos=y, dim=56
                    SplitHalfRoPE.FillFrequencies(cos, sin, baseOff + dimT, pos: y, dim: dimH, theta: theta);

                    // Width axis: pos=x, dim=56
                    SplitHalfRoPE.FillFrequencies(cos, sin, baseOff + dimT + dimH, pos: x, dim: dimW, theta: theta);
                }
            }
        }

        return (cos, sin);
    }

    public static void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
        => SplitHalfRoPE.ApplyRoPE(qk, cos, sin, seqLen, numHeads, headDim);
}
