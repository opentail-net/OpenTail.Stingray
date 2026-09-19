using OpenTail.Stingray.Diffusion.Primitives;

namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// 3D Rotary Positional Embedding (3D-RoPE) for HunyuanVideo.
/// Decomposes 128 head dimension into 3 coordinate axes: {16, 56, 56} for (temporal frames t, height y, width x) with theta = 256.0 / 10000.0.
/// Reference: stable-diffusion.cpp:src/model/diffusion/hunyuan.hpp:axes_dim
///
/// <para><b>REAL BUG FOUND AND FIXED 2026-09-19 (docs/092)</b>: this file previously delegated to
/// <see cref="SplitHalfRoPE"/> (split-half/"NEOX" pairing `(x[i], x[i+dim/2])`), but the real
/// reference uses adjacent-pair ("interleaved"/"GPT-J") pairing `(x[2i], x[2i+1])`. Confirmed
/// against `examples/diffusers/src/diffusers/models/embeddings.py`'s `apply_rotary_emb` --
/// HunyuanVideo's call sites (`transformer_hunyuan_video.py`'s
/// `apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)`) use the function's default
/// `use_real_unbind_dim=-1` branch, `x.reshape(*x.shape[:-1], -1, 2).unbind(-1)` -- explicitly
/// commented "Used for flux, cogvideox, hunyuan-dit" in the real source, i.e. the SAME convention
/// FLUX.2 needed and the SAME bug class already found and fixed for Wan on 2026-08-31 (see
/// `Wan/WanRoPE.cs`'s own doc comment for that precedent). Axis-dim split (t=16, h=56, w=56) and
/// theta are UNCHANGED by this fix -- only the pairing/table-layout convention.</para>
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

        var invFreqT = InterleavedRoPE.ComputeInvFreqs(dimT, theta);
        var invFreqH = InterleavedRoPE.ComputeInvFreqs(dimH, theta);
        var invFreqW = InterleavedRoPE.ComputeInvFreqs(dimW, theta);

        for (int t = 0; t < numFrames; t++)
        {
            for (int y = 0; y < patchH; y++)
            {
                for (int x = 0; x < patchW; x++)
                {
                    int tokenIdx = (t * patchH + y) * patchW + x;
                    int baseOff = tokenIdx * headDim;

                    // Temporal axis: pos=t, dim=16
                    InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff, dimT), sin.AsSpan(baseOff, dimT), t, invFreqT);

                    // Height axis: pos=y, dim=56
                    InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff + dimT, dimH), sin.AsSpan(baseOff + dimT, dimH), y, invFreqH);

                    // Width axis: pos=x, dim=56
                    InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff + dimT + dimH, dimW), sin.AsSpan(baseOff + dimT + dimH, dimW), x, invFreqW);
                }
            }
        }

        return (cos, sin);
    }

    public static void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
        => InterleavedRoPE.ApplyRoPE(qk, cos, sin, seqLen, numHeads, headDim);
}
