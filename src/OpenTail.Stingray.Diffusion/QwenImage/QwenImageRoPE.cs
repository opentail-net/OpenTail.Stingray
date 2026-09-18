
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

        // Real `txt_id_start = max(h_len, w_len) / 2` (integer division) -- NOT 0.
        int txtIdStart = Math.Max(imgH, imgW) / 2;

        // 1. Text tokens: real scheme -- SAME 3-axis split as image tokens, SAME position value
        //    on all 3 axes, linspace-sequential starting at txtIdStart (real `gen_qwen_image_ids`:
        //    `txt_ids = linspace(txt_id_start, txt_id_start + context_len - 1, context_len)`,
        //    each token gets `{txt_ids[j], txt_ids[j], txt_ids[j]}`).
        for (int i = 0; i < txtLen; i++)
        {
            float pos = txtIdStart + i;
            int off = i * headDim;
            SplitHalfRoPE.FillFrequencies(cos, sin, off, pos: pos, dim: dimT, theta: theta);
            SplitHalfRoPE.FillFrequencies(cos, sin, off + dimT, pos: pos, dim: dimH, theta: theta);
            SplitHalfRoPE.FillFrequencies(cos, sin, off + dimT + dimH, pos: pos, dim: dimW, theta: theta);
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

                float tPos = 0f;
                float hPos = hOffset + row;
                float wPos = wOffset + col;

                SplitHalfRoPE.FillFrequencies(cos, sin, baseOff, pos: tPos, dim: dimT, theta: theta);
                SplitHalfRoPE.FillFrequencies(cos, sin, baseOff + dimT, pos: hPos, dim: dimH, theta: theta);
                SplitHalfRoPE.FillFrequencies(cos, sin, baseOff + dimT + dimH, pos: wPos, dim: dimW, theta: theta);
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Applies 3D RoPE in-place to Q or K tensor [seqLen, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
        => SplitHalfRoPE.ApplyRoPE(qk, cos, sin, seqLen, numHeads, headDim);
}
