namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// 3D continuous-coordinate Rotary Positional Embedding for LTX-Video.
///
/// Ported directly from HuggingFace `diffusers`' real, released
/// `LTXVideoRotaryPosEmbed`/`apply_rotary_emb` (`diffusers/models/transformers/transformer_ltx.py`)
/// -- NOT the `stable-diffusion.cpp` reference this project otherwise follows, whose
/// `build_video_rope_matrix` (causal temporal shift, "middle indices grid" averaging, split-half
/// rotation) turned out to disagree with the actual PyTorch reference on three separate points once
/// diffusers was available locally to diff against directly:
/// 1. Coordinates are plain `index * rope_interpolation_scale[axis] / base[axis]` -- no causal
///    `+1-scale` shift, no start/end-of-cell averaging.
/// 2. Frequency channels are laid out FREQUENCY-major, `[f0_t,f0_h,f0_w, f1_t,f1_h,f1_w, ...]`, not
///    AXIS-major (`[all t, all h, all w]`).
/// 3. Rotation is INTERLEAVED (pairs `(x[2i],x[2i+1])`, cos/sin values duplicated via
///    `repeat_interleave(2)`), matching this project's own `WanRoPE` convention -- not split-half.
/// `rope_interpolation_scale` real default (`pipeline_ltx.py`): `(vae_temporal_ratio / frame_rate,
/// vae_spatial_ratio, vae_spatial_ratio)` = `(8/25, 32, 32)` for the real released VAE + default
/// `frame_rate=25`.
/// </summary>
public static class LtxVideoRoPE
{
    /// <summary>
    /// Builds per-token cos/sin tables of length <paramref name="headDim"/> each (interleaved
    /// layout: index 2i and 2i+1 share one rotation angle), for a dense (numFrames, patchH, patchW)
    /// video-token grid in raster (frame, then row, then col) order -- matching this project's own
    /// token ordering convention for Wan/patchify.
    /// </summary>
    public static (float[] cos, float[] sin) ComputeContinuous3DRoPE(
        int numFrames,
        int patchH,
        int patchW,
        int headDim = 64,
        float theta = 10000.0f,
        float frameRate = 25.0f,
        int temporalScale = 8,
        int spatialScaleH = 32,
        int spatialScaleW = 32,
        int baseNumFrames = 20,
        int baseHeight = 2048,
        int baseWidth = 2048,
        int patchSizeT = 1,
        int patchSizeS = 1)
    {
        int freqCount = headDim / 6;
        var indices = BuildFreqGrid(theta, freqCount);
        int padHalf = headDim / 2 - freqCount * 3;

        // real: rope_interpolation_scale = (vae_temporal_ratio / frame_rate, vae_spatial, vae_spatial)
        double scaleT = (double)temporalScale / frameRate;
        double coordScaleT = scaleT * patchSizeT / baseNumFrames;
        double coordScaleH = (double)spatialScaleH * patchSizeS / baseHeight;
        double coordScaleW = (double)spatialScaleW * patchSizeS / baseWidth;

        int totalTokens = numFrames * patchH * patchW;
        var cos = new float[totalTokens * headDim];
        var sin = new float[totalTokens * headDim];

        int token = 0;
        for (int t = 0; t < numFrames; t++)
        {
            double coordT = t * coordScaleT;
            for (int h = 0; h < patchH; h++)
            {
                double coordH = h * coordScaleH;
                for (int w = 0; w < patchW; w++)
                {
                    double coordW = w * coordScaleW;
                    int baseOff = token * headDim;
                    int outIdx = padHalf;

                    for (int p = 0; p < padHalf; p++)
                    {
                        WriteAngle(cos, sin, baseOff, p, 0.0);
                    }

                    for (int f = 0; f < freqCount; f++)
                    {
                        double idxVal = indices[f];
                        WriteAngle(cos, sin, baseOff, outIdx++, idxVal * (coordT * 2.0 - 1.0));
                        WriteAngle(cos, sin, baseOff, outIdx++, idxVal * (coordH * 2.0 - 1.0));
                        WriteAngle(cos, sin, baseOff, outIdx++, idxVal * (coordW * 2.0 - 1.0));
                    }

                    token++;
                }
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Builds compact per-token cos/sin tables of length headDim/2 each for GPU Flux2DRoPE kernel.
    /// </summary>
    public static (float[] cos, float[] sin) ComputeContinuous3DRoPECompact(
        int numFrames,
        int patchH,
        int patchW,
        int headDim = 2048,
        float theta = 10000.0f,
        float frameRate = 25.0f,
        int temporalScale = 8,
        int spatialScaleH = 32,
        int spatialScaleW = 32,
        int baseNumFrames = 20,
        int baseHeight = 2048,
        int baseWidth = 2048,
        int patchSizeT = 1,
        int patchSizeS = 1)
    {
        int freqCount = headDim / 6;
        var indices = BuildFreqGrid(theta, freqCount);
        int padHalf = headDim / 2 - freqCount * 3;
        int nPairs = headDim / 2;

        double scaleT = (double)temporalScale / frameRate;
        double coordScaleT = scaleT * patchSizeT / baseNumFrames;
        double coordScaleH = (double)spatialScaleH * patchSizeS / baseHeight;
        double coordScaleW = (double)spatialScaleW * patchSizeS / baseWidth;

        int totalTokens = numFrames * patchH * patchW;
        var cos = new float[totalTokens * nPairs];
        var sin = new float[totalTokens * nPairs];

        int token = 0;
        for (int t = 0; t < numFrames; t++)
        {
            double coordT = t * coordScaleT;
            for (int h = 0; h < patchH; h++)
            {
                double coordH = h * coordScaleH;
                for (int w = 0; w < patchW; w++)
                {
                    double coordW = w * coordScaleW;
                    int baseOff = token * nPairs;
                    int outIdx = padHalf;

                    for (int p = 0; p < padHalf; p++)
                    {
                        cos[baseOff + p] = 1.0f;
                        sin[baseOff + p] = 0.0f;
                    }

                    for (int f = 0; f < freqCount; f++)
                    {
                        double idxVal = indices[f];
                        double aT = idxVal * (coordT * 2.0 - 1.0);
                        double aH = idxVal * (coordH * 2.0 - 1.0);
                        double aW = idxVal * (coordW * 2.0 - 1.0);

                        cos[baseOff + outIdx] = (float)Math.Cos(aT); sin[baseOff + outIdx++] = (float)Math.Sin(aT);
                        cos[baseOff + outIdx] = (float)Math.Cos(aH); sin[baseOff + outIdx++] = (float)Math.Sin(aH);
                        cos[baseOff + outIdx] = (float)Math.Cos(aW); sin[baseOff + outIdx++] = (float)Math.Sin(aW);
                    }

                    token++;
                }
            }
        }

        return (cos, sin);
    }


    /// <summary>Writes one rotation angle at half-dim index <paramref name="halfIdx"/>, duplicated
    /// (`repeat_interleave(2)`) into the full-dim array at `[2*halfIdx, 2*halfIdx+1]`. Computed in
    /// double precision (2026-09-14 fix): at the highest frequency index the raw angle reaches
    /// ~15708 rad, where float32 argument-reduction inside <see cref="MathF"/>'s cos/sin genuinely
    /// diverges from torch's own (effectively float64-internal) reduction by up to ~0.02 -- see
    /// LtxVideoGoldenParityTests' own documented RoPE max-abs-diff tolerance. That per-channel phase
    /// error is small in a flat cosine-similarity check of the whole table, but compounds through
    /// attention's softmax (which is sensitive to exactly the high-frequency/fine-grained phase
    /// terms) -- a leading real candidate for the block0 golden-parity shortfall (cosine-sim
    /// 0.9894 vs required 0.9999) investigated in docs/077. Reduced in double precision here so the
    /// angle is accurate before the final float32 cos/sin, closing that gap at the source rather
    /// than downstream.</summary>
    private static void WriteAngle(float[] cos, float[] sin, int baseOff, int halfIdx, double angle)
    {
        float c = (float)Math.Cos(angle);
        float s = (float)Math.Sin(angle);
        int i = baseOff + 2 * halfIdx;
        cos[i] = c;
        cos[i + 1] = c;
        sin[i] = s;
        sin[i + 1] = s;
    }

    /// <summary>Real `theta ** linspace(0, 1, dim//6) * pi/2`, computed in double precision (see
    /// <see cref="WriteAngle"/>'s doc comment for why).</summary>
    private static double[] BuildFreqGrid(float theta, int freqCount)
    {
        var outArr = new double[freqCount];
        if (freqCount <= 0) return outArr;
        double halfPi = Math.PI / 2.0;
        if (freqCount == 1)
        {
            outArr[0] = halfPi; // theta^0 * pi/2
            return outArr;
        }
        double logTheta = Math.Log(theta);
        for (int i = 0; i < freqCount; i++)
        {
            double ratio = (double)i / (freqCount - 1);
            outArr[i] = Math.Exp(logTheta * ratio) * halfPi;
        }
        return outArr;
    }

    /// <summary>Real `apply_rotary_emb`: interleaved pairs `(x[2i], x[2i+1])` treated as
    /// (real, imag); `x_rotated = [-x_imag, x_real]` (also interleaved); `out = x*cos +
    /// x_rotated*sin`. Since `cos[2i]==cos[2i+1]` and `sin[2i]==sin[2i+1]` (repeat_interleave), this
    /// reduces to the same interleaved rotation this project's `WanRoPE.ApplyRoPE` already
    /// implements -- applied only to attn1 (self-attention); attn2 (cross-attention against the
    /// caption sequence) receives no RoPE at all per the reference (`image_rotary_emb=None`).</summary>
    public static unsafe void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
    {
        fixed (float* pQk = qk, pCos = cos, pSin = sin)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* pCosTok = pCos + (long)s * headDim;
                float* pSinTok = pSin + (long)s * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = pQk + ((long)s * numHeads + h) * headDim;
                    for (int d = 0; d < headDim; d += 2)
                    {
                        float x1 = head[d];
                        float x2 = head[d + 1];
                        float c = pCosTok[d];
                        float sRot = pSinTok[d];

                        head[d] = x1 * c - x2 * sRot;
                        head[d + 1] = x1 * sRot + x2 * c;
                    }
                }
            }
        }
    }
}
