using System;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's `TextEmbedding` (dit.py): token ids -> zero-padded-to-audio-length embedding + fixed
/// sinusoidal position embedding + 4x ConvNeXtV2Block, re-masking to zero at padded positions
/// after every stage. Ported directly from `examples/f5-tts-py/f5_tts/model/backbones/dit.py`'s
/// `TextEmbedding.forward` (the `mask_padding=True`, non-tensor-`seq_len`, `average_upsampling=
/// False` branch -- the one this checkpoint's config and single-utterance inference path use).
///
/// Unlike VITS's length regulator, F5-TTS does NOT explicitly align text tokens to audio frames
/// via a duration model inside the DiT itself -- it just zero-pads the RAW token sequence out to
/// the (externally-decided) target frame count and lets the ConvNeXt blocks + later cross-
/// attention-free DiT (which sees text only via this per-frame-position embedding, added
/// alongside x/cond in `InputEmbedding`) learn to use it. If more tokens than frames are given,
/// the reference TRUNCATES to the first `numFrames` tokens (confirmed in the source, not a
/// bug to "fix").
/// </summary>
public static class F5TextEmbedding
{
    /// <summary>tokens are raw (unshifted) character ids. Returns [numFrames, TextDim] embedding, channel-last.</summary>
    public static float[] Forward(F5TtsWeights w, ReadOnlySpan<int> tokens, int numFrames) =>
        Forward(w, tokens, numFrames, dropText: false);

    /// <summary>
    /// dropText=true is CFG's null/unconditional branch: `text = zeros_like(text)` happens AFTER
    /// the pad-mask is computed but BEFORE the embedding lookup (cfm.py/dit.py), so every position
    /// (including originally-valid, non-padded ones) looks up embedding row 0 (the FILLER token,
    /// since the raw drop-value 0 still goes through the same `+1` shift as everything else --
    /// wait: the shift happens before the drop, so dropped text is exactly 0, i.e. filler row 0
    /// directly), while the pad MASK used for zeroing during the ConvNeXt blocks still reflects
    /// the ORIGINAL (pre-drop) token/pad boundary, not "everything is padding".
    /// </summary>
    public static float[] Forward(F5TtsWeights w, ReadOnlySpan<int> tokens, int numFrames, bool dropText)
    {
        int dim = F5TtsWeights.TextDim;
        int numTokens = tokens.Length;

        // text = text + 1 (0 reserved as filler); truncate to numFrames; zero-pad the tail.
        var shifted = new int[numFrames];
        var isPad = new bool[numFrames];
        for (int i = 0; i < numFrames; i++)
        {
            if (i < numTokens && i < numFrames)
            {
                shifted[i] = dropText ? 0 : tokens[i] + 1;
                isPad[i] = false;
            }
            else
            {
                shifted[i] = 0;
                isPad[i] = true;
            }
        }

        var x = new float[numFrames * dim];
        for (int i = 0; i < numFrames; i++)
        {
            int row = shifted[i] * dim;
            int off = i * dim;
            Array.Copy(w.TextEmbedWeight, row, x, off, dim);
        }

        // + sinusoidal freqs_cis(text_dim, position=i): first half cos, second half sin (NOT
        // interleaved -- a separate, simpler formula from the attention RoPE used later).
        int halfDim = dim / 2;
        for (int i = 0; i < numFrames; i++)
        {
            int off = i * dim;
            for (int k = 0; k < halfDim; k++)
            {
                float freq = MathF.Pow(10000f, -(2f * k) / dim);
                float angle = i * freq;
                x[off + k] += MathF.Cos(angle);
                x[off + halfDim + k] += MathF.Sin(angle);
            }
        }

        MaskPad(x, numFrames, dim, isPad);

        for (int b = 0; b < w.TextBlocks.Length; b++)
        {
            x = ConvNeXtV2Block(x, numFrames, dim, w.TextBlocks[b]);
            MaskPad(x, numFrames, dim, isPad);
        }

        return x;
    }

    private static void MaskPad(float[] x, int numFrames, int dim, bool[] isPad)
    {
        for (int i = 0; i < numFrames; i++)
        {
            if (!isPad[i]) continue;
            int off = i * dim;
            Array.Clear(x, off, dim);
        }
    }

    /// <summary>ConvNeXtV2Block.forward: residual + pwconv2(grn(gelu(pwconv1(layernorm(dwconv(x)))))).</summary>
    private static float[] ConvNeXtV2Block(float[] x, int t, int dim, F5TextBlockWeights bw)
    {
        var h = F5Kernels.DepthwiseConv1dSamePad(x, t, dim, bw.DwConvWeight, bw.DwConvBias, kernel: 7);
        h = F5Kernels.LayerNorm(h, t, dim, bw.NormWeight, bw.NormBias);

        int inter = bw.PwConv1Bias.Length;
        h = F5Kernels.Linear(h, t, dim, bw.PwConv1Weight, bw.PwConv1Bias, inter);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.GeluExact(h[i]);
        h = F5Kernels.Grn(h, t, inter, bw.GrnGamma, bw.GrnBeta);
        h = F5Kernels.Linear(h, t, inter, bw.PwConv2Weight, bw.PwConv2Bias, dim);

        var output = new float[x.Length];
        for (int i = 0; i < output.Length; i++) output[i] = x[i] + h[i];
        return output;
    }
}
