
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's `TextEmbedding` (dit.py): token ids -> zero-padded-to-audio-length embedding + fixed
/// sinusoidal position embedding + 4x ConvNeXtV2Block. Ported directly from
/// `examples/f5-tts-py/f5_tts/model/backbones/dit.py`'s `TextEmbedding.forward` (the non-tensor-
/// `seq_len`, `average_upsampling=False` branch -- the one this checkpoint's config and single-
/// utterance inference path use).
///
/// <para><b>Real `F5TTS_Base.yaml` sets `text_mask_padding: False`</b> (confirmed against the
/// real config, not assumed) -- with a non-tensor `seq_len` (this pipeline's own usage), that
/// makes the reference's entire `text_mask == 0` / `masked_fill` re-zeroing mechanism dead code
/// (every branch that would apply it is gated behind `if self.mask_padding`), so padded positions
/// simply flow through the ConvNeXt blocks holding whatever the embedding-row-0 ("filler" token)
/// + sinusoidal-position values naturally are, NOT hard-zeroed before/after each block. This class
/// does NOT re-mask, matching that real behavior.</para>
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
    public static float[] Forward(F5TtsWeights w, ReadOnlySpan<int> tokens, int numFrames, bool dropText, Core.IComputeBackend? backend = null)
    {
        int dim = F5TtsWeights.TextDim;
        int numTokens = tokens.Length;

        // text = text + 1 (0 reserved as filler); truncate to numFrames; zero-pad the tail (real
        // `F.pad(text, ..., value=0)` -- row 0 is the filler token's own LEARNED embedding, not a
        // literal zero vector; no re-masking follows, see class doc).
        var shifted = new int[numFrames];
        for (int i = 0; i < numFrames; i++)
            shifted[i] = (i < numTokens && !dropText) ? tokens[i] + 1 : 0;

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

        for (int b = 0; b < w.TextBlocks.Length; b++)
            x = ConvNeXtV2Block(x, numFrames, dim, w.TextBlocks[b], backend);

        return x;
    }

    /// <summary>ConvNeXtV2Block.forward: residual + pwconv2(grn(gelu(pwconv1(layernorm(dwconv(x)))))).</summary>
    private static float[] ConvNeXtV2Block(float[] x, int t, int dim, F5TextBlockWeights bw, Core.IComputeBackend? backend = null)
    {
        var h = F5Kernels.DepthwiseConv1dSamePad(x, t, dim, bw.DwConvWeight, bw.DwConvBias, kernel: 7);
        h = F5Kernels.LayerNorm(h, t, dim, bw.NormWeight, bw.NormBias);

        int inter = bw.PwConv1Bias.Length;
        h = backend is not null
            ? F5Kernels.LinearGpu(backend, h, t, dim, bw.PwConv1Weight, bw.PwConv1Bias, inter)
            : F5Kernels.Linear(h, t, dim, bw.PwConv1Weight, bw.PwConv1Bias, inter);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.GeluExact(h[i]);
        h = F5Kernels.Grn(h, t, inter, bw.GrnGamma, bw.GrnBeta);
        h = backend is not null
            ? F5Kernels.LinearGpu(backend, h, t, inter, bw.PwConv2Weight, bw.PwConv2Bias, dim)
            : F5Kernels.Linear(h, t, inter, bw.PwConv2Weight, bw.PwConv2Bias, dim);

        var output = new float[x.Length];
        for (int i = 0; i < output.Length; i++) output[i] = x[i] + h[i];
        return output;
    }
}
