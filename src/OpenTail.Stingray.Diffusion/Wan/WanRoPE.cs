
namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// 3D Rotary Positional Embedding (3D-RoPE) for Wan Video Diffusion.
/// Decomposes 128 head dimension into 3 axes: {44, 42, 42} for (temporal frames t, height y, width x) with theta = 10000.
/// Reference: stable-diffusion.cpp:src/model/diffusion/wan.hpp:axes_dim
///
/// <para><b>Real rotation convention, confirmed against `transformer_wan.py`'s own
/// `WanAttnProcessor.apply_rotary_emb`/`WanRotaryPosEmbed`</b> (found and fixed 2026-08-31, root-
/// caused via Wan's own non-convergence at real step counts): Wan uses INTERLEAVED
/// ("GPT-J"/classic-Llama style) pairs <c>(x[2i], x[2i+1])</c>, NOT the split-half ("NEOX" style)
/// <c>(x[i], x[i+dim/2])</c> pairing the shared <see cref="Primitives.SplitHalfRoPE"/> kernel
/// implements (real, correct for the OTHER models that kernel serves -- Wan is the one that
/// differs, confirmed directly: `x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)` takes
/// consecutive-pair elements, and `WanRotaryPosEmbed.__init__` builds its frequency tables with
/// `repeat_interleave_real=True`, i.e. each frequency value duplicated at consecutive positions
/// -- matching this file's own <see cref="ComputeFrequency"/> below, NOT the split-half
/// `SplitHalfRoPE.FillFrequencies` this file previously (incorrectly) delegated to). Per-axis
/// dims (t=44, h=42, w=42) were already correct and unaffected by this fix -- confirmed matching
/// the real `h_dim = w_dim = 2*(head_dim//6); t_dim = head_dim - h_dim - w_dim` formula for
/// head_dim=128.</para>
/// </summary>
public static class WanRoPE
{
    public static (float[] cos, float[] sin) Compute3DRoPE(
        int numFrames,
        int patchH,
        int patchW,
        int headDim = 128,
        float theta = 10000.0f)
    {
        // Axes dimensions: 44 (temporal), 42 (height), 42 (width) = 128
        int dimT = 44;
        int dimH = 42;
        int dimW = 42;

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

                    // Temporal axis: pos=t, dim=44
                    FillFrequenciesInterleaved(cos, sin, baseOff + 0, pos: t, dim: dimT, theta: theta);

                    // Height axis: pos=y, dim=42
                    FillFrequenciesInterleaved(cos, sin, baseOff + dimT, pos: y, dim: dimH, theta: theta);

                    // Width axis: pos=x, dim=42
                    FillFrequenciesInterleaved(cos, sin, baseOff + dimT + dimH, pos: x, dim: dimW, theta: theta);
                }
            }
        }

        return (cos, sin);
    }

    /// <summary>Real Wan RoPE frequency layout: for pair index i (0..dim/2), angle = pos * theta^(-2i/dim),
    /// written at BOTH consecutive positions 2i and 2i+1 (repeat-interleaved, matching the real
    /// `repeat_interleave_real=True` reference) -- NOT the split-half [0,half)/[half,dim) layout.</summary>
    private static void FillFrequenciesInterleaved(float[] cos, float[] sin, int offset, float pos, int dim, float theta)
    {
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Pow(theta, -2.0f * i / dim);
            float angle = pos * freq;
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            cos[offset + 2 * i] = c;
            cos[offset + 2 * i + 1] = c;
            sin[offset + 2 * i] = s;
            sin[offset + 2 * i + 1] = s;
        }
    }

    /// <summary>Real Wan interleaved rotation: pairs (x[2i], x[2i+1]), matching
    /// `x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1); out[...,0::2] = x1*cos - x2*sin;
    /// out[...,1::2] = x1*sin + x2*cos`.</summary>
    public static unsafe void ApplyRoPE(float[] qk, float[] cos, float[] sin, int seqLen, int numHeads, int headDim)
    {
        fixed (float* pQk = qk, pCos = cos, pSin = sin)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* pCosTok = pCos + s * headDim;
                float* pSinTok = pSin + s * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = pQk + (s * numHeads + h) * headDim;
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
