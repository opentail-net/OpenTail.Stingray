using System;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's VITS TextEncoder (enc_p): token embedding (scaled by sqrt(hidden)) -> N-layer
/// post-LN Transformer with windowed relative-position self-attention (Shaw et al. 2018 style,
/// NOT the Transformer-XL/ESPnet scheme used elsewhere in this codebase for Chatterbox/Kokoro) ->
/// proj to [mu, logs]. No absolute positional embedding -- VITS relies entirely on the relative-
/// position attention bias.
///
/// Relative-position attention: the reference (VITS attentions.py MultiHeadAttention) computes
/// this via a "skew/unskew" matrix trick purely for vectorized-PyTorch efficiency; mathematically
/// it is exactly a per-(query,key) lookup into a small, FIXED-size window table
/// (2*windowSize+1 entries) keyed by clamp(j-i, -windowSize, windowSize), with ZERO contribution
/// (not clamped-to-edge) for |j-i| &gt; windowSize (confirmed by the reference's `_get_relative_
/// embeddings` zero-padding the table before slicing when T &gt; windowSize+1). This class
/// implements that direct form -- correct for any T, no reshape games -- verified against real
/// ONNX golden output via cosine similarity.
/// </summary>
public static class PiperTextEncoder
{
    /// <summary>
    /// tokens are phoneme ids. Returns (encoderHidden, mu, logs), all channel-first [hidden, T].
    /// encoderHidden is the encoder stack's output BEFORE the final proj split -- VITS's
    /// SynthesizerTrn.infer feeds this (not mu) into the duration predictor and flow, per
    /// `x, m_p, logs_p, x_mask = self.enc_p(...)` followed by `self.dp(x, x_mask, ...)`.
    /// </summary>
    public static (float[] EncoderHidden, float[] Mu, float[] Logs) Forward(PiperOnnxWeights w, ReadOnlySpan<int> tokens)
    {
        int t = tokens.Length;
        int dim = w.HiddenDim;
        float embScale = MathF.Sqrt(dim);

        // Embed + scale, channel-first [dim, t].
        var x = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
        {
            int tok = tokens[ti];
            int rowBase = tok * dim;
            for (int c = 0; c < dim; c++)
                x[c * t + ti] = w.EmbeddingWeight[rowBase + c] * embScale;
        }

        foreach (var layer in w.Layers)
        {
            var attnOut = VitsAttentionKernels.RelPositionSelfAttention(
                x, t, dim, w.NumHeads, w.WindowSize,
                layer.ConvQWeight, layer.ConvQBias,
                layer.ConvKWeight, layer.ConvKBias,
                layer.ConvVWeight, layer.ConvVBias,
                layer.ConvOWeight, layer.ConvOBias,
                layer.EmbRelK, layer.EmbRelV);
            for (int i = 0; i < x.Length; i++) x[i] += attnOut[i];
            x = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, layer.Norm1Gamma, layer.Norm1Beta);

            var ffnOut = Ffn(x, t, w, layer);
            for (int i = 0; i < x.Length; i++) x[i] += ffnOut[i];
            x = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, layer.Norm2Gamma, layer.Norm2Beta);
        }

        // proj: Conv1d(hidden, 2*hidden, kernel=1) -> split into mu/logs.
        var stats = VitsAttentionKernels.Conv1x1(x, dim, t, w.ProjWeight, w.ProjBias, 2 * dim);
        var mu = new float[dim * t];
        var logs = new float[dim * t];
        Array.Copy(stats, 0, mu, 0, dim * t);
        Array.Copy(stats, dim * t, logs, 0, dim * t);
        return (x, mu, logs);
    }

    /// <summary>FFN: conv_1 (kernel=3, pad=1) -> ReLU -> conv_2 (kernel=3, pad=1).</summary>
    private static float[] Ffn(float[] x, int t, PiperOnnxWeights w, PiperEncoderLayerWeights lw)
    {
        int dim = w.HiddenDim;
        int ffnHidden = lw.Ffn1Bias.Length;
        var h = VitsAttentionKernels.Conv1dSamePad(x, dim, t, lw.Ffn1Weight, lw.Ffn1Bias, ffnHidden, w.FfnKernel);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f; // ReLU
        return VitsAttentionKernels.Conv1dSamePad(h, ffnHidden, t, lw.Ffn2Weight, lw.Ffn2Bias, dim, w.FfnKernel);
    }
}
