
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 DVAE decoder (inference-only path): codebook indices -&gt; mel-like latent,
/// channel-first [80, T]. Ported directly from the real `DiscreteVAE.decode`
/// (`TTS/tts/layers/xtts/dvae.py`): embed lookup -&gt; `decoder.0` (1x1 conv, no activation) -&gt;
/// 3x `ResBlock` (`decoder.1/2/3`) -&gt; 2x `UpsampledConv` (`decoder.4/5`, nearest x2 upsample then
/// a "same"-padded k3 conv, each followed by ReLU) -&gt; `decoder.6` (1x1 conv, no activation).
/// See <see cref="XttsDvaeWeights"/>'s doc comment for the real construction args this topology
/// was confirmed against.
/// </summary>
public static class XttsDvaeDecoder
{
    /// <summary>codes are real codebook indices (0..1023) from the GPT2's autoregressive output. Returns mel-like latent, channel-first [80, T].</summary>
    public static float[] Decode(XttsDvaeWeights w, ReadOnlySpan<int> codes)
    {
        int t = codes.Length;
        const int dim = XttsDvaeWeights.CodebookDim;

        // embed_code: real Quantize.embed is stored [dim, n_embed] -- select COLUMN codes[ti],
        // matching `F.embedding(embed_id, self.embed.transpose(0,1))`'s real lookup semantics.
        var x = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
        {
            int code = codes[ti];
            for (int c = 0; c < dim; c++)
                x[c * t + ti] = w.CodebookEmbed[c * XttsDvaeWeights.NumTokens + code];
        }

        // decoder.0: 1x1 conv, 512 -> 1024, NO activation (plain conv inserted before the resblocks).
        x = VitsAttentionKernels.Conv1x1(x, dim, t, w.Decoder0Weight, w.Decoder0Bias, XttsDvaeWeights.InnermostDim);

        foreach (var rb in w.ResBlocks)
            x = ResBlockForward(x, XttsDvaeWeights.InnermostDim, t, rb);

        (x, t) = UpsampledConvForward(x, XttsDvaeWeights.InnermostDim, t, w.Decoder4Weight, w.Decoder4Bias, XttsDvaeWeights.InnermostDim);
        (x, t) = UpsampledConvForward(x, XttsDvaeWeights.InnermostDim, t, w.Decoder5Weight, w.Decoder5Bias, XttsDvaeWeights.InnermostDim / 2);

        // decoder.6: 1x1 conv, 512 -> 80, NO activation (final layer).
        x = VitsAttentionKernels.Conv1x1(x, XttsDvaeWeights.InnermostDim / 2, t, w.Decoder6Weight, w.Decoder6Bias, XttsDvaeWeights.MelDim);
        return x;
    }

    /// <summary>ResBlock: net = [conv3,ReLU,conv3,ReLU,conv1], output = net(x) + x.</summary>
    private static float[] ResBlockForward(float[] x, int ch, int t, XttsDvaeResBlockWeights rb)
    {
        var h = VitsAttentionKernels.Conv1dSamePad(x, ch, t, rb.Conv0Weight, rb.Conv0Bias, ch, kernel: 3);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f;
        h = VitsAttentionKernels.Conv1dSamePad(h, ch, t, rb.Conv2Weight, rb.Conv2Bias, ch, kernel: 3);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f;
        h = VitsAttentionKernels.Conv1x1(h, ch, t, rb.Conv4Weight, rb.Conv4Bias, ch);

        var output = new float[ch * t];
        for (int i = 0; i < output.Length; i++) output[i] = h[i] + x[i];
        return output;
    }

    /// <summary>UpsampledConv: nearest x2 upsample (real `F.interpolate(..., mode="nearest")`, NOT ConvTranspose1d -- `use_transposed_convs=False`) then a "same"-padded k3 conv, followed by ReLU.</summary>
    private static (float[] Output, int NewT) UpsampledConvForward(float[] x, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        int newT = t * XttsDvaeWeights.UpsampleStride;
        var up = new float[inCh * newT];
        for (int c = 0; c < inCh; c++)
            for (int ti = 0; ti < t; ti++)
            {
                float v = x[c * t + ti];
                int outBase = c * newT + ti * XttsDvaeWeights.UpsampleStride;
                for (int s = 0; s < XttsDvaeWeights.UpsampleStride; s++)
                    up[outBase + s] = v;
            }

        var output = VitsAttentionKernels.Conv1dSamePad(up, inCh, newT, weight, bias, outCh, kernel: XttsDvaeWeights.UpsampleKernel);
        for (int i = 0; i < output.Length; i++) if (output[i] < 0f) output[i] = 0f;
        return (output, newT);
    }
}
