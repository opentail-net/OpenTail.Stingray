using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// `attentions.py`'s `Encoder.forward`: an N-layer relative-attention Transformer with an
/// optional speaker-embedding injection (`x = x + spk_emb_linear(g)`) at a fixed layer index.
/// This exact module is instantiated TWICE in MeloTTS's VITS2 graph -- once as enc_p's top-level
/// encoder (6 layers, ffnKernel=3, see <see cref="MeloTextEncoder"/>) and once INSIDE every
/// `TransformerCouplingLayer` of the normalizing flow (3 layers, ffnKernel=5, see
/// <see cref="MeloFlow"/>) -- so the loop is extracted here rather than duplicated, per the
/// project's shared-kernel convention (see <see cref="VitsAttentionKernels"/> for the analogous
/// extraction one level down).
/// </summary>
public static class MeloRelativeEncoder
{
    public static float[] Forward(
        float[] x, int t, int dim, int heads, int window, int ffnKernel,
        MeloEncoderLayerWeights[] layers, int condLayerIdx, float[]? spkProjected)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            if (i == condLayerIdx && spkProjected != null)
            {
                for (int ti = 0; ti < t; ti++)
                    for (int c = 0; c < dim; c++)
                        x[c * t + ti] += spkProjected[c];
            }

            var layer = layers[i];
            var attnOut = VitsAttentionKernels.RelPositionSelfAttention(
                x, t, dim, heads, window,
                layer.ConvQWeight, layer.ConvQBias,
                layer.ConvKWeight, layer.ConvKBias,
                layer.ConvVWeight, layer.ConvVBias,
                layer.ConvOWeight, layer.ConvOBias,
                layer.EmbRelK, layer.EmbRelV);
            for (int k = 0; k < x.Length; k++) x[k] += attnOut[k];
            x = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, layer.Norm1Gamma, layer.Norm1Beta);

            var ffnOut = Ffn(x, t, dim, ffnKernel, layer);
            for (int k = 0; k < x.Length; k++) x[k] += ffnOut[k];
            x = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, layer.Norm2Gamma, layer.Norm2Beta);
        }
        return x;
    }

    private static float[] Ffn(float[] x, int t, int dim, int kernel, MeloEncoderLayerWeights lw)
    {
        int ffnHidden = lw.Ffn1Bias.Length;
        var h = VitsAttentionKernels.Conv1dSamePad(x, dim, t, lw.Ffn1Weight, lw.Ffn1Bias, ffnHidden, kernel);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f; // ReLU
        return VitsAttentionKernels.Conv1dSamePad(h, ffnHidden, t, lw.Ffn2Weight, lw.Ffn2Bias, dim, kernel);
    }

    /// <summary>
    /// weight is [inDim, outDim] row-major -- NOT torch's usual [outDim, inDim] nn.Linear layout.
    /// Every `spk_emb_linear` in this checkpoint (enc_p's and each flow layer's) exports as a bare
    /// MatMul (not Gemm with transB), and PyTorch's nn.Linear-to-MatMul export pre-transposes the
    /// weight to [in,out] so the graph can do a straight `input @ weight` -- confirmed by reading
    /// real initializer dims. Getting this backwards silently produces plausible-shaped garbage;
    /// caught originally via a per-layer golden-cosine bisection (see MeloTextEncoder's history).
    /// </summary>
    public static float[] LinearVec(float[] input, float[] weight, float[] bias, int outDim, int inDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            for (int i = 0; i < inDim; i++) sum += weight[i * outDim + o] * input[i];
            output[o] = sum;
        }
        return output;
    }
}
