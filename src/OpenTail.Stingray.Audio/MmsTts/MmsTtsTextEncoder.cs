
namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// MMS-TTS's VITS TextEncoder (`text_encoder`), same math as
/// <see cref="OpenTail.Stingray.Audio.Piper.PiperTextEncoder"/> (both real VITS `TextEncoder`
/// implementations) -- direct port sharing the exact same <see cref="VitsAttentionKernels"/>
/// primitives, adapted only for this checkpoint's own weight field names
/// (<see cref="MmsTtsWeights"/>, HuggingFace `transformers.VitsModel` naming).
/// </summary>
public static class MmsTtsTextEncoder
{
    /// <summary>
    /// tokens are real vocab ids (see <see cref="MmsTtsTokenizer"/>). Returns (encoderHidden, mu,
    /// logs), all channel-first [hidden, T]. encoderHidden is the encoder stack's output BEFORE
    /// the final proj split -- fed into the duration predictor and the flow, not mu/logs directly.
    /// </summary>
    public static (float[] EncoderHidden, float[] Mu, float[] Logs) Forward(MmsTtsWeights w, ReadOnlySpan<int> tokens)
    {
        int t = tokens.Length;
        int dim = w.HiddenDim;
        float embScale = MathF.Sqrt(dim);

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
            System.Numerics.Tensors.TensorPrimitives.Add(x, attnOut, x);
            x = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, layer.Norm1Gamma, layer.Norm1Beta);

            var ffnOut = Ffn(x, t, w, layer);
            System.Numerics.Tensors.TensorPrimitives.Add(x, ffnOut, x);
            x = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, layer.Norm2Gamma, layer.Norm2Beta);
        }

        var stats = VitsAttentionKernels.Conv1x1(x, dim, t, w.ProjWeight, w.ProjBias, 2 * dim);
        var mu = new float[dim * t];
        var logs = new float[dim * t];
        Array.Copy(stats, 0, mu, 0, dim * t);
        Array.Copy(stats, dim * t, logs, 0, dim * t);
        return (x, mu, logs);
    }

    /// <summary>FFN: conv_1 (kernel=ffnKernel, "same" pad) -> ReLU -> conv_2 (same).</summary>
    private static float[] Ffn(float[] x, int t, MmsTtsWeights w, MmsEncoderLayerWeights lw)
    {
        int dim = w.HiddenDim;
        int ffnHidden = lw.Ffn1Bias.Length;
        var h = VitsAttentionKernels.Conv1dSamePad(x, dim, t, lw.Ffn1Weight, lw.Ffn1Bias, ffnHidden, w.FfnKernel);
        System.Numerics.Tensors.TensorPrimitives.Max(h, 0f, h); // ReLU
        return VitsAttentionKernels.Conv1dSamePad(h, ffnHidden, t, lw.Ffn2Weight, lw.Ffn2Bias, dim, w.FfnKernel);
    }
}
