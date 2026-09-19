using OpenTail.Stingray.Diffusion.Primitives;

namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// 3D Rotary Positional Embedding (3D-RoPE) for Qwen Image.
/// Decomposes 128 head dimension into 3 axes: {16, 56, 56} for (time, height, width) with theta = 10000.
/// Reference: examples/stable-diffusion.cpp's `Rope::gen_qwen_image_ids`/`gen_vid_ids`
/// (src/model/common/rope.hpp) -- confirmed in-repo 2026-09-18 during the checkerboard-artifact
/// investigation (docs/089).
///
/// <para><b>REAL BUG FOUND AND FIXED here</b>: the previous version treated text tokens with a
/// completely different scheme from image tokens (full-128-dim single-axis RoPE at sequential
/// positions 0..txtLen-1) -- but the real reference gives text tokens the SAME 3-axis scheme as
/// image tokens, with the SAME position value repeated across all 3 axes (t=h=w=txt_ids[j]),
/// starting at `txt_id_start = max(h_len, w_len) / 2` (integer division), NOT 0. Image tokens
/// also use a CENTERED offset (`scale_rope=true` in the real `gen_vid_ids` call): row/col
/// positions range from `-len/2` to `len-1-len/2`, not `0..len-1`. Mixing an incompatible text
/// position scheme with the image scheme corrupts every joint (image-text) attention term while
/// leaving local image-image relative-RoPE relationships intact -- consistent with the observed
/// severe, periodic checkerboard/tiling artifact (locally-coherent texture, no global structure).
/// </para>
///
/// <para><b>SECOND REAL BUG FOUND AND FIXED 2026-09-19 (docs/092)</b>: this file also delegated to
/// <see cref="SplitHalfRoPE"/> (split-half/"NEOX" pairing), but the real reference uses
/// adjacent-pair ("interleaved"/"GPT-J") pairing. Confirmed against
/// `examples/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py`'s
/// `apply_rotary_emb_qwen` -- every real call site (`img_query`/`img_key`/`txt_query`/`txt_key`)
/// passes `use_real=False`, taking the complex-number path:
/// `x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))` -- still adjacent
/// pairs, just via complex multiplication instead of explicit cos/sin. Same bug class already
/// found and fixed for Wan (2026-08-31) and FLUX.2 (2026-09-18) this session. Axis-dim split
/// (t=16, h=56, w=56), theta, and the txt/img position-scheme fix above are UNCHANGED by this fix
/// -- only the pairing/table-layout convention.</para>
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

        var invFreqT = InterleavedRoPE.ComputeInvFreqs(dimT, theta);
        var invFreqH = InterleavedRoPE.ComputeInvFreqs(dimH, theta);
        var invFreqW = InterleavedRoPE.ComputeInvFreqs(dimW, theta);

        // Real `txt_id_start = max(h_len, w_len) / 2` (integer division) -- NOT 0.
        int txtIdStart = Math.Max(imgH, imgW) / 2;

        // 1. Text tokens: real scheme -- SAME 3-axis split as image tokens, SAME position value
        //    on all 3 axes, linspace-sequential starting at txtIdStart (real `gen_qwen_image_ids`:
        //    `txt_ids = linspace(txt_id_start, txt_id_start + context_len - 1, context_len)`,
        //    each token gets `{txt_ids[j], txt_ids[j], txt_ids[j]}`).
        for (int i = 0; i < txtLen; i++)
        {
            int pos = txtIdStart + i;
            int off = i * headDim;
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(off, dimT), sin.AsSpan(off, dimT), pos, invFreqT);
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(off + dimT, dimH), sin.AsSpan(off + dimT, dimH), pos, invFreqH);
            InterleavedRoPE.FillAxisFreqs(cos.AsSpan(off + dimT + dimH, dimW), sin.AsSpan(off + dimT + dimH, dimW), pos, invFreqW);
        }

        // 2. Image tokens: real scheme -- `scale_rope=true` in the real `gen_vid_ids` call means
        //    row/col offsets are CENTERED (`h_offset = -h_len/2`, `w_offset = -w_len/2`, integer
        //    division), not starting at 0. Time axis is a single frame (t=0, this port doesn't
        //    yet support Qwen-Image-Layered's multi-frame variant).
        int hOffset = -(imgH / 2);
        int wOffset = -(imgW / 2);
        for (int row = 0; row < imgH; row++)
        {
            for (int col = 0; col < imgW; col++)
            {
                int tokenIdx = txtLen + row * imgW + col;
                int baseOff = tokenIdx * headDim;

                int tPos = 0;
                int hPos = hOffset + row;
                int wPos = wOffset + col;

                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff, dimT), sin.AsSpan(baseOff, dimT), tPos, invFreqT);
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff + dimT, dimH), sin.AsSpan(baseOff + dimT, dimH), hPos, invFreqH);
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff + dimT + dimH, dimW), sin.AsSpan(baseOff + dimT + dimH, dimW), wPos, invFreqW);
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Applies 3D RoPE in-place to Q or K tensor [seqLen, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
        => InterleavedRoPE.ApplyRoPE(qk, cos, sin, seqLen, numHeads, headDim);
}
