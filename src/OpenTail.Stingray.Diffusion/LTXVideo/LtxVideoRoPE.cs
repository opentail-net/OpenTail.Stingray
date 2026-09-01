namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// 3D continuous-coordinate Rotary Positional Embedding for LTX-Video.
/// Reference: stable-diffusion.cpp:src/model/diffusion/ltxv.hpp
/// (generate_freq_grid / build_video_rope_matrix / apply_hidden_rope).
///
/// Real convention (per the reference, ported literally rather than assumed from Wan's RoPE):
/// positions are CONTINUOUS pixel-space coordinates (latent index * VAE scale factor, with the
/// temporal axis additionally shifted by the model's causal-VAE convention and averaged over the
/// latent cell's start/end -- "middle indices grid", `use_middle_indices_grid=true` in the real
/// config), normalized against `max_pos=[20,2048,2048]` and mapped to `[-1,1]` before being
/// multiplied by a log-spaced frequency ladder scaled by pi/2 -- NOT `pos * theta^(-2i/dim)` the
/// way Wan/Llama-style RoPE computes it. Rotation itself is split-half ("NEOX" style, pairs
/// `(x[i], x[i+dim/2])`), matching the real config's `video_rope_interleaved=false`.
/// </summary>
public static class LtxVideoRoPE
{
    /// <summary>
    /// Builds per-token cos/sin tables of length <paramref name="headDim"/> each (split-half
    /// layout: index i and i+headDim/2 share one rotation angle), for a dense (numFrames, patchH,
    /// patchW) video-token grid in raster (frame, then row, then col) order -- matching this
    /// project's own token ordering convention for Wan/patchify.
    /// </summary>
    public static (float[] cos, float[] sin) ComputeContinuous3DRoPE(
        int numFrames,
        int patchH,
        int patchW,
        int headDim = 64,
        float theta = 10000.0f,
        float frameRate = 24.0f,
        int temporalScale = 8,
        int spatialScaleH = 32,
        int spatialScaleW = 32,
        int maxPosT = 20,
        int maxPosH = 2048,
        int maxPosW = 2048,
        bool causalTemporalPositioning = true,
        bool useMiddleIndicesGrid = true)
    {
        int halfDim = headDim / 2;
        int freqCount = headDim / 6; // 2 * positionalDims(=3)
        var indices = BuildFreqGrid(theta, freqCount);
        int padSize = halfDim - freqCount * 3;

        int totalTokens = numFrames * patchH * patchW;
        var cos = new float[totalTokens * headDim];
        var sin = new float[totalTokens * headDim];

        int token = 0;
        for (int t = 0; t < numFrames; t++)
        {
            float pixelT = t * temporalScale;
            if (causalTemporalPositioning) pixelT = MathF.Max(0f, pixelT + 1f - temporalScale);
            if (useMiddleIndicesGrid)
            {
                float endT = (t + 1) * temporalScale;
                if (causalTemporalPositioning) endT = MathF.Max(0f, endT + 1f - temporalScale);
                pixelT = 0.5f * (pixelT + endT);
            }
            pixelT /= frameRate;

            for (int h = 0; h < patchH; h++)
            {
                float pixelH = h * spatialScaleH;
                if (useMiddleIndicesGrid) pixelH += 0.5f * spatialScaleH;

                for (int w = 0; w < patchW; w++)
                {
                    float pixelW = w * spatialScaleW;
                    if (useMiddleIndicesGrid) pixelW += 0.5f * spatialScaleW;

                    float coordT = pixelT / maxPosT;
                    float coordH = pixelH / maxPosH;
                    float coordW = pixelW / maxPosW;

                    int baseOff = token * headDim;
                    int freqIdx = padSize;

                    // Leading pad entries carry angle 0 (cos=1, sin=0).
                    for (int p = 0; p < padSize; p++)
                    {
                        cos[baseOff + p] = 1f;
                        cos[baseOff + halfDim + p] = 1f;
                        sin[baseOff + p] = 0f;
                        sin[baseOff + halfDim + p] = 0f;
                    }

                    for (int f = 0; f < freqCount; f++)
                    {
                        float idxVal = indices[f];
                        WriteAngle(cos, sin, baseOff, halfDim, freqIdx++, idxVal * (coordT * 2f - 1f));
                        WriteAngle(cos, sin, baseOff, halfDim, freqIdx++, idxVal * (coordH * 2f - 1f));
                        WriteAngle(cos, sin, baseOff, halfDim, freqIdx++, idxVal * (coordW * 2f - 1f));
                    }

                    token++;
                }
            }
        }

        return (cos, sin);
    }

    private static void WriteAngle(float[] cos, float[] sin, int baseOff, int halfDim, int i, float angle)
    {
        float c = MathF.Cos(angle);
        float s = MathF.Sin(angle);
        cos[baseOff + i] = c;
        cos[baseOff + halfDim + i] = c;
        sin[baseOff + i] = s;
        sin[baseOff + halfDim + i] = s;
    }

    /// <summary>Real `generate_freq_grid(theta, positional_dims=3, dim)`: a log-spaced ladder from
    /// 1 to theta, scaled by pi/2 (a single-entry ladder is pi/2 exactly).</summary>
    private static float[] BuildFreqGrid(float theta, int freqCount)
    {
        var outArr = new float[freqCount];
        if (freqCount <= 0) return outArr;
        float halfPi = MathF.PI / 2f;
        if (freqCount == 1)
        {
            outArr[0] = halfPi;
            return outArr;
        }
        float logTheta = MathF.Log(theta);
        for (int i = 0; i < freqCount; i++)
        {
            float ratio = (float)i / (freqCount - 1);
            outArr[i] = MathF.Exp(logTheta * ratio) * halfPi;
        }
        return outArr;
    }

    /// <summary>Split-half ("NEOX") rotation: pairs `(x[i], x[i+dim/2])` share one angle, matching
    /// the real `video_rope_interleaved=false` config -- applied only to attn1 (self-attention);
    /// attn2 (cross-attention against the caption sequence) receives no RoPE at all per the
    /// reference (`pe=nullptr` on that call).</summary>
    public static unsafe void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
    {
        int half = headDim / 2;
        fixed (float* pQk = qk, pCos = cos, pSin = sin)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* pCosTok = pCos + (long)s * headDim;
                float* pSinTok = pSin + (long)s * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = pQk + ((long)s * numHeads + h) * headDim;
                    for (int d = 0; d < half; d++)
                    {
                        float x1 = head[d];
                        float x2 = head[d + half];
                        float c = pCosTok[d];
                        float sRot = pSinTok[d];

                        head[d] = x1 * c - x2 * sRot;
                        head[d + half] = x1 * sRot + x2 * c;
                    }
                }
            }
        }
    }
}
