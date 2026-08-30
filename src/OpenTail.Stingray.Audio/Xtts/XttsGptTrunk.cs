
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 bare GPT2 trunk (`gpt.gpt`, a plain standard HuggingFace `GPT2Model` -- see
/// <see cref="XttsGptWeights"/>'s doc comment). Standard pre-LN causal transformer decoder:
/// per layer, `ln_1 -&gt; causal self-attention -&gt; +residual -&gt; ln_2 -&gt; MLP(GELU) -&gt; +residual`,
/// then a final `ln_f`. Takes pre-computed input embeddings directly (XTTS's own text/mel
/// token+positional embeddings are computed by the caller, not this class -- matches the real
/// reference's `wte`/`wpe` deletion, see <see cref="XttsGptWeights"/>).
/// </summary>
public static class XttsGptTrunk
{
    /// <summary>inputEmbeds is channel-first [ModelDim, T]. Returns the trunk's final hidden state (after ln_f), channel-first [ModelDim, T].</summary>
    public static float[] Forward(XttsGptWeights w, float[] inputEmbeds, int t)
    {
        var x = inputEmbeds;
        foreach (var layer in w.Layers)
            x = LayerForward(x, t, layer);

        return VitsAttentionKernels.LayerNormChannelFirst(x, XttsGptWeights.ModelDim, t, w.FinalNormWeight, w.FinalNormBias);
    }

    private static float[] LayerForward(float[] x, int t, XttsGptLayerWeights lw)
    {
        int dim = XttsGptWeights.ModelDim;

        var normed1 = VitsAttentionKernels.LayerNormChannelFirst(x, dim, t, lw.Ln1Weight, lw.Ln1Bias);
        var attnOut = CausalSelfAttention(normed1, t, lw);
        var h1 = new float[dim * t];
        for (int i = 0; i < h1.Length; i++) h1[i] = x[i] + attnOut[i];

        var normed2 = VitsAttentionKernels.LayerNormChannelFirst(h1, dim, t, lw.Ln2Weight, lw.Ln2Bias);
        var mlpOut = Mlp(normed2, t, lw);
        var h2 = new float[dim * t];
        for (int i = 0; i < h2.Length; i++) h2[i] = h1[i] + mlpOut[i];

        return h2;
    }

    private static float[] CausalSelfAttention(float[] x, int t, XttsGptLayerWeights lw)
    {
        int dim = XttsGptWeights.ModelDim;
        int heads = XttsGptWeights.NumHeads;
        int headDim = XttsGptWeights.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        // c_attn: fused QKV projection, dim -> 3*dim (real HF GPT2 split order: q, then k, then v).
        var qkv = VitsAttentionKernels.Conv1x1(x, dim, t, lw.AttnCAttnWeight, lw.AttnCAttnBias, 3 * dim);

        var context = new float[dim * t];

        for (int h = 0; h < heads; h++)
        {
            int qOff = h * headDim;
            int kOff = dim + h * headDim;
            int vOff = 2 * dim + h * headDim;

            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                // Causal: only attend to positions j <= i.
                for (int j = 0; j <= i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += qkv[(qOff + d) * t + i] * qkv[(kOff + d) * t + j];
                    scores[j] = dot * scale;
                }
                SoftmaxPrefixInPlace(scores, i + 1);

                for (int j = 0; j <= i; j++)
                {
                    float p = scores[j];
                    if (p == 0f) continue;
                    for (int d = 0; d < headDim; d++)
                        context[(qOff + d) * t + i] += p * qkv[(vOff + d) * t + j];
                }
            }
        }

        return VitsAttentionKernels.Conv1x1(context, dim, t, lw.AttnCProjWeight, lw.AttnCProjBias, dim);
    }

    private static void SoftmaxPrefixInPlace(float[] scores, int len)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < len; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < len; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < len; i++) scores[i] *= invSum;
    }

    private static float[] Mlp(float[] x, int t, XttsGptLayerWeights lw)
    {
        int dim = XttsGptWeights.ModelDim;
        int ffn = XttsGptWeights.FfnDim;

        var h = VitsAttentionKernels.Conv1x1(x, dim, t, lw.MlpCFcWeight, lw.MlpCFcBias, ffn);
        for (int i = 0; i < h.Length; i++) h[i] = GeluNew(h[i]);
        return VitsAttentionKernels.Conv1x1(h, ffn, t, lw.MlpCProjWeight, lw.MlpCProjBias, dim);
    }

    /// <summary>HF GPT2Config's real default `activation_function="gelu_new"` (tanh approximation), NOT overridden by XTTS's construction.</summary>
    private static float GeluNew(float x)
    {
        const float c = 0.7978845608f; // sqrt(2/pi)
        return 0.5f * x * (1f + MathF.Tanh(c * (x + 0.044715f * x * x * x)));
    }
}
