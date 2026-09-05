using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 bare GPT2 trunk (`gpt.gpt`, a plain standard HuggingFace `GPT2Model` -- see
/// <see cref="XttsGptWeights"/>'s doc comment). Standard pre-LN causal transformer decoder:
/// per layer, `ln_1 -> causal self-attention -> +residual -> ln_2 -> MLP(GELU) -> +residual`,
/// then a final `ln_f`. Takes pre-computed input embeddings directly (XTTS's own text/mel
/// token+positional embeddings are computed by the caller, not this class -- matches the real
/// reference's `wte`/`wpe` deletion, see <see cref="XttsGptWeights"/>).
/// </summary>
public static class XttsGptTrunk
{
    /// <summary>
    /// Evaluates ONE step of the GPT2 trunk with KV cache at the current position.
    /// Zero heap allocations via pre-allocated scratch workspace.
    /// </summary>
    public static ReadOnlySpan<float> Step(XttsGptWeights w, XttsGptCache cache, ReadOnlySpan<float> inputVec, bool computeFinalNorm = true)
    {
        ReadOnlySpan<float> x = inputVec;
        bool trace = Environment.GetEnvironmentVariable("STINGRAY_XTTS_LAYER_TRACE") == "1";
        for (int i = 0; i < w.Layers.Length; i++)
        {
            LayerStep(x, w.Layers[i], cache, i);
            x = cache.Output;
            if (trace)
            {
                double sumsq = 0;
                foreach (var v in x) sumsq += (double)v * v;
                Console.Error.WriteLine($"[XttsLayerTrace] layer={i} rms={Math.Sqrt(sumsq / x.Length):F6} first3=[{x[0]:F4},{x[1]:F4},{x[2]:F4}]");
            }
        }

        if (!computeFinalNorm) return ReadOnlySpan<float>.Empty;

        LayerNorm(cache.Output, w.FinalNormWeight, w.FinalNormBias, cache.LastHidden);
        return cache.LastHidden;
    }

    private static void LayerStep(ReadOnlySpan<float> x, XttsGptLayerWeights lw, XttsGptCache cache, int layerIdx)
    {
        int dim = XttsGptWeights.ModelDim;
        int heads = XttsGptWeights.NumHeads;
        int headDim = XttsGptWeights.HeadDim;
        int ffnDim = XttsGptWeights.FfnDim;

        // 1. ln_1
        LayerNorm(x, lw.Ln1Weight, lw.Ln1Bias, cache.Normed);

        // 2. c_attn: fused QKV projection [3*dim, dim]
        LinearWithBias(cache.Normed, lw.AttnCAttnWeight, lw.AttnCAttnBias, 3 * dim, dim, cache.Qkv);

        int pos = cache.Counts[layerIdx];
        var kSlot = cache.K[layerIdx][pos];
        var vSlot = cache.V[layerIdx][pos];

        Array.Copy(cache.Qkv, dim, kSlot, 0, dim);
        Array.Copy(cache.Qkv, 2 * dim, vSlot, 0, dim);

        cache.Counts[layerIdx]++;
        int t = cache.Counts[layerIdx];

        // 3. Multi-head causal self-attention
        Array.Clear(cache.Context, 0, dim);
        float scale = 1f / MathF.Sqrt(headDim);
        var kLayer = cache.K[layerIdx];
        var vLayer = cache.V[layerIdx];

        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            var qSpan = cache.Qkv.AsSpan(hOff, headDim);
            for (int j = 0; j < t; j++)
            {
                float dot = System.Numerics.Tensors.TensorPrimitives.Dot(qSpan, kLayer[j].AsSpan(hOff, headDim));
                cache.Scores[j] = dot * scale;
            }
            SoftmaxPrefixInPlace(cache.Scores, t);

            var ctxSpan = cache.Context.AsSpan(hOff, headDim);
            for (int j = 0; j < t; j++)
            {
                float s = cache.Scores[j];
                if (s == 0f) continue;
                TensorPrimitives.MultiplyAdd(vLayer[j].AsSpan(hOff, headDim), s, ctxSpan, ctxSpan);
            }
        }

        // 4. c_proj
        LinearWithBias(cache.Context, lw.AttnCProjWeight, lw.AttnCProjBias, dim, dim, cache.AttnOut);

        // 5. Residual
        System.Numerics.Tensors.TensorPrimitives.Add(x, cache.AttnOut.AsSpan(0, dim), cache.H1.AsSpan(0, dim));

        // 6. ln_2
        LayerNorm(cache.H1, lw.Ln2Weight, lw.Ln2Bias, cache.FfnNormed);

        // 7. mlp.c_fc
        LinearWithBias(cache.FfnNormed, lw.MlpCFcWeight, lw.MlpCFcBias, ffnDim, dim, cache.FfnMid);

        // 8. GeluNew
        unsafe
        {
            fixed (float* fp = cache.FfnMid)
            {
                SimdKernels.GeluInPlace(fp, ffnDim);
            }
        }

        // 9. mlp.c_proj
        LinearWithBias(cache.FfnMid, lw.MlpCProjWeight, lw.MlpCProjBias, dim, ffnDim, cache.FfnOut);

        // 10. Residual
        System.Numerics.Tensors.TensorPrimitives.Add(cache.H1.AsSpan(0, dim), cache.FfnOut.AsSpan(0, dim), cache.Output.AsSpan(0, dim));
    }

    private static unsafe void LinearWithBias(float[] input, float[] weight, float[] bias, int outDim, int inDim, float[] output)
    {
        fixed (float* wp = weight, bp = bias, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, bp, xp, outDim, inDim);
        }
    }

    private static unsafe void LayerNorm(ReadOnlySpan<float> x, float[] gamma, float[] beta, float[] output, float eps = 1e-5f)
    {
        fixed (float* xp = x, gp = gamma, bp = beta, op = output)
        {
            SimdKernels.LayerNorm(op, xp, gp, bp, x.Length, eps);
        }
    }

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

    /// <summary>Explicit max-subtracted stable softmax, not <c>TensorPrimitives.SoftMax(x, x)</c> --
    /// same class of bug as Chatterbox's S3Gen CFM decoder (see
    /// <see cref="OpenTail.Stingray.Audio.Primitives.DenseKernels.SoftmaxInPlace"/>'s doc comment):
    /// confirmed via direct comparison against the real PyTorch reference's per-step logits that
    /// this call was producing a substantially wrong winning token (a token the reference doesn't
    /// even rank in its top 5) starting a few generation steps in, compounding into a stable
    /// degenerate "stuck repeating the same token" state that (under real sampling) essentially
    /// never reaches XTTS's stop_audio_token -- explaining the port's "always hits the 605-token
    /// hard cap" bug. Do not swap back without re-verifying against a real multi-step generation,
    /// not just a single-step golden test.</summary>
    private static void SoftmaxPrefixInPlace(float[] scores, int len)
    {
        var span = scores.AsSpan(0, len);
        float max = float.NegativeInfinity;
        for (int i = 0; i < len; i++) if (span[i] > max) max = span[i];
        float sum = 0f;
        for (int i = 0; i < len; i++)
        {
            float e = MathF.Exp(span[i] - max);
            span[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < len; i++) span[i] *= invSum;
    }

    private static unsafe float[] Mlp(float[] x, int t, XttsGptLayerWeights lw)
    {
        int dim = XttsGptWeights.ModelDim;
        int ffn = XttsGptWeights.FfnDim;

        var h = VitsAttentionKernels.Conv1x1(x, dim, t, lw.MlpCFcWeight, lw.MlpCFcBias, ffn);
        fixed (float* hp = h)
        {
            SimdKernels.GeluInPlace(hp, h.Length);
        }
        return VitsAttentionKernels.Conv1x1(h, ffn, t, lw.MlpCProjWeight, lw.MlpCProjBias, dim);
    }

    /// <summary>HF GPT2Config's real default `activation_function="gelu_new"` (tanh approximation), NOT overridden by XTTS's construction.</summary>
    private static float GeluNew(float x)
    {
        const float c = 0.7978845608f; // sqrt(2/pi)
        return 0.5f * x * (1f + MathF.Tanh(c * (x + 0.044715f * x * x * x)));
    }
}
