
namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real Parler-TTS decoder forward pass (MusicGen-style causal decoder), transcribed directly
/// from the real `parler_tts` Python package's `ParlerTTSDecoderLayer`/`ParlerTTSDecoder`
/// (`modeling_parler_tts.py`, fetched via `pip download parler-tts --no-deps`) -- see
/// <see cref="ParlerDecoderWeights"/>'s doc comment for the full config derivation.
///
/// <para>Standard pre-LN transformer decoder, genuinely simpler than every other pipeline
/// finished this session: plain LayerNorm (mean-subtract + bias, NOT RMSNorm), full MHA (no
/// GQA), NO RoPE (real precomputed sinusoidal positions, loaded as a table, not a formula), plain
/// GELU FFN (not gated). Per layer: `self_attn_layer_norm(x)` -&gt; causal self-attention -&gt;
/// residual -&gt; `encoder_attn_layer_norm(x)` -&gt; cross-attention to the T5 encoder's output
/// -&gt; residual -&gt; `final_layer_norm(x)` -&gt; `fc2(gelu(fc1(x)))` -&gt; residual.</para>
///
/// <para>Embedding: `sum(embed_tokens[cb][input_ids[cb]] for cb in 0..8)` (9 real codebook
/// streams summed, no scale) + the real precomputed sinusoidal position embedding for that
/// timestep. Output: 9 SEPARATE `lm_heads` projections (not tied), all predicted in parallel per
/// timestep -- unlike Fish Speech's sequential fast-AR codebook expansion, Parler's decoder
/// emits all 9 codebooks' logits directly from the same 24-layer trunk.</para>
/// </summary>
public static class ParlerDecoder
{
    private const int HeadDim = ParlerDecoderWeights.HiddenDim / ParlerDecoderWeights.NumHeads; // 64

    /// <summary>
    /// Composes the real input embedding for one PROMPT (transcript) token: a plain
    /// `embed_prompts` lookup + the same sinusoidal position table the audio codebook embeddings
    /// use, sharing one continuous position counter across the prepended prompt-token prefix and
    /// the audio tokens that follow it (real `ParlerTTSDecoder.forward`: positions are computed
    /// ONCE over the already-concatenated prompt+audio sequence, not two independent counters).
    /// </summary>
    public static float[] EmbedPromptToken(ParlerDecoderWeights w, int tokenId, int position)
    {
        var table = w.EmbedPrompts ?? throw new InvalidOperationException(
            "This ParlerDecoderWeights has no embed_prompts table (GGUF conversion gap) -- cannot embed a prompt token.");
        var emb = new float[ParlerDecoderWeights.HiddenDim];
        int row = tokenId * ParlerDecoderWeights.HiddenDim;
        int posRow = position * ParlerDecoderWeights.HiddenDim;
        System.Numerics.Tensors.TensorPrimitives.Add(
            table.AsSpan(row, ParlerDecoderWeights.HiddenDim),
            w.EmbedPositions.AsSpan(posRow, ParlerDecoderWeights.HiddenDim),
            emb);
        return emb;
    }

    /// <summary>Composes the real input embedding for one timestep: sum of 9 codebook-token lookups + the real sinusoidal position embedding.</summary>
    public static float[] EmbedStep(ParlerDecoderWeights w, int[] codebookTokenIds, int position)
    {
        var emb = new float[ParlerDecoderWeights.HiddenDim];
        int dim = ParlerDecoderWeights.HiddenDim;
        for (int cb = 0; cb < ParlerDecoderWeights.NumCodebooks; cb++)
        {
            int row = codebookTokenIds[cb] * dim;
            System.Numerics.Tensors.TensorPrimitives.Add(emb, w.EmbedTokens[cb].AsSpan(row, dim), emb);
        }
        int posRow = position * dim;
        System.Numerics.Tensors.TensorPrimitives.Add(emb, w.EmbedPositions.AsSpan(posRow, dim), emb);
        return emb;
    }

    /// <summary>Runs the full decoder trunk over a sequence of already-composed input embeddings, cross-attending to the real T5 encoder output. Returns per-position hidden states [T, HiddenDim] (post final LayerNorm).</summary>
    public static float[][] Forward(ParlerDecoderWeights w, float[][] inputEmbeds, float[][] encoderHidden)
    {
        var x = inputEmbeds;
        foreach (var layer in w.Layers)
            x = DecoderLayer(x, encoderHidden, layer);

        int t = x.Length;
        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = LayerNorm(x[i], w.FinalLayerNormWeight, w.FinalLayerNormBias));
        return output;
    }

    /// <summary>
    /// Real single-step decode with self-/cross-attention KV caching, equivalent to (but far
    /// cheaper for autoregressive generation than) calling <see cref="Forward"/> on the whole
    /// prefix each step. Self-attention K/V is recomputed for THIS position and appended to the
    /// cache; cross-attention K/V is projected from <paramref name="encoderHidden"/> once per
    /// layer (on the first call using this cache) and reused unchanged thereafter. Returns the
    /// post-final-LayerNorm hidden state for this one position.
    /// </summary>
    public static float[] ForwardStep(ParlerDecoderWeights w, ParlerDecoderKvCache cache, float[] inputEmbed, float[][] encoderHidden)
    {
        var x = inputEmbed;
        for (int layerIdx = 0; layerIdx < w.Layers.Length; layerIdx++)
            x = DecoderLayerStep(x, encoderHidden, w.Layers[layerIdx], cache, layerIdx);

        return LayerNorm(x, w.FinalLayerNormWeight, w.FinalLayerNormBias);
    }

    private static float[] DecoderLayerStep(float[] x, float[][] encoderHidden, ParlerDecoderLayerWeights lw, ParlerDecoderKvCache cache, int layerIdx)
    {
        int dim = ParlerDecoderWeights.HiddenDim;

        var normed1 = LayerNorm(x, lw.SelfAttnLayerNormWeight, lw.SelfAttnLayerNormBias);
        var selfAttnOut = SelfAttentionStep(normed1, lw, cache, layerIdx);

        var afterSelf = new float[dim];
        System.Numerics.Tensors.TensorPrimitives.Add(x, selfAttnOut, afterSelf);

        var normed2 = LayerNorm(afterSelf, lw.CrossAttnLayerNormWeight, lw.CrossAttnLayerNormBias);
        var crossAttnOut = CrossAttentionStep(normed2, encoderHidden, lw, cache, layerIdx);

        var afterCross = new float[dim];
        System.Numerics.Tensors.TensorPrimitives.Add(afterSelf, crossAttnOut, afterCross);

        var normed3 = LayerNorm(afterCross, lw.FinalLayerNormWeight, lw.FinalLayerNormBias);
        var ffnOut = FfnStep(normed3, lw);

        var output = new float[dim];
        System.Numerics.Tensors.TensorPrimitives.Add(afterCross, ffnOut, output);
        return output;
    }

    /// <summary>Real causal self-attention for one new position: project this step's Q/K/V, append K/V to the cache, attend over every cached position including this one.</summary>
    private static float[] SelfAttentionStep(float[] xNormed, ParlerDecoderLayerWeights lw, ParlerDecoderKvCache cache, int layerIdx)
    {
        int dim = ParlerDecoderWeights.HiddenDim;
        int heads = ParlerDecoderWeights.NumHeads;

        var qNew = LinearQ8_0(xNormed, lw.SelfAttnQWeight, dim, dim);
        var kNew = LinearQ8_0(xNormed, lw.SelfAttnKWeight, dim, dim);
        var vNew = LinearQ8_0(xNormed, lw.SelfAttnVWeight, dim, dim);

        cache.SelfK[layerIdx].Add(kNew);
        cache.SelfV[layerIdx].Add(vNew);
        var kCache = cache.SelfK[layerIdx];
        var vCache = cache.SelfV[layerIdx];
        int t = kCache.Count;

        var context = new float[dim];
        float scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, heads, h =>
        {
            int off = h * HeadDim;
            var scores = new float[t];
            for (int j = 0; j < t; j++) scores[j] = Dot(qNew, kCache[j], off, HeadDim) * scale;
            SoftmaxInPlace(scores, t);

            var ctxSpan = context.AsSpan(off, HeadDim);
            for (int j = 0; j < t; j++)
                TensorPrimitives.MultiplyAdd(vCache[j].AsSpan(off, HeadDim), scores[j], ctxSpan, ctxSpan);
        });

        return LinearQ8_0(context, lw.SelfAttnOutWeight, dim, dim);
    }

    /// <summary>Real cross-attention for one new position: project Q for this step; project encoder K/V into the cache ONLY the first time this layer sees a cache without them, otherwise reuse.</summary>
    private static float[] CrossAttentionStep(float[] xNormed, float[][] encoderHidden, ParlerDecoderLayerWeights lw, ParlerDecoderKvCache cache, int layerIdx)
    {
        int dim = ParlerDecoderWeights.HiddenDim;
        int heads = ParlerDecoderWeights.NumHeads;
        int tk = encoderHidden.Length;

        var q = LinearQ8_0(xNormed, lw.CrossAttnQWeight, dim, dim);

        if (cache.CrossK[layerIdx] is null)
        {
            var kCross = new float[tk][];
            var vCross = new float[tk][];
            Parallel.For(0, tk, j =>
            {
                kCross[j] = LinearQ8_0(encoderHidden[j], lw.CrossAttnKWeight, dim, dim);
                vCross[j] = LinearQ8_0(encoderHidden[j], lw.CrossAttnVWeight, dim, dim);
            });
            cache.CrossK[layerIdx] = kCross;
            cache.CrossV[layerIdx] = vCross;
        }
        var k = cache.CrossK[layerIdx]!;
        var v = cache.CrossV[layerIdx]!;

        var context = new float[dim];
        float scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, heads, h =>
        {
            int off = h * HeadDim;
            var scores = new float[tk];
            for (int j = 0; j < tk; j++) scores[j] = Dot(q, k[j], off, HeadDim) * scale;
            SoftmaxInPlace(scores, tk);

            var ctxSpan = context.AsSpan(off, HeadDim);
            for (int j = 0; j < tk; j++)
                TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, HeadDim), scores[j], ctxSpan, ctxSpan);
        });

        return LinearQ8_0(context, lw.CrossAttnOutWeight, dim, dim);
    }

    private static float[] FfnStep(float[] x, ParlerDecoderLayerWeights lw)
    {
        var h = LinearQ8_0(x, lw.Fc1Weight, ParlerDecoderWeights.HiddenDim, ParlerDecoderWeights.FfnDim);
        for (int d = 0; d < h.Length; d++) h[d] = Gelu(h[d]);
        return LinearQ8_0(h, lw.Fc2Weight, ParlerDecoderWeights.FfnDim, ParlerDecoderWeights.HiddenDim);
    }

    /// <summary>Projects the final hidden states through all 9 real, separate lm_heads. Returns [T][9][OutputVocabSize].</summary>
    public static float[][][] ComputeLogits(ParlerDecoderWeights w, float[][] hidden)
    {
        int t = hidden.Length;
        var result = new float[t][][];
        Parallel.For(0, t, i =>
        {
            result[i] = new float[ParlerDecoderWeights.NumCodebooks][];
            for (int cb = 0; cb < ParlerDecoderWeights.NumCodebooks; cb++)
                result[i][cb] = LinearNoBias(hidden[i], w.LmHeads[cb], ParlerDecoderWeights.HiddenDim, ParlerDecoderWeights.OutputVocabSize);
        });
        return result;
    }

    private static float[][] DecoderLayer(float[][] x, float[][] encoderHidden, ParlerDecoderLayerWeights lw)
    {
        int t = x.Length;

        var normed1 = new float[t][];
        Parallel.For(0, t, i => normed1[i] = LayerNorm(x[i], lw.SelfAttnLayerNormWeight, lw.SelfAttnLayerNormBias));
        var selfAttnOut = SelfAttentionCausal(normed1, lw);

        var afterSelf = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) row[d] = x[i][d] + selfAttnOut[i][d];
            afterSelf[i] = row;
        });

        var normed2 = new float[t][];
        Parallel.For(0, t, i => normed2[i] = LayerNorm(afterSelf[i], lw.CrossAttnLayerNormWeight, lw.CrossAttnLayerNormBias));
        var crossAttnOut = CrossAttention(normed2, encoderHidden, lw);

        var afterCross = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) row[d] = afterSelf[i][d] + crossAttnOut[i][d];
            afterCross[i] = row;
        });

        var normed3 = new float[t][];
        Parallel.For(0, t, i => normed3[i] = LayerNorm(afterCross[i], lw.FinalLayerNormWeight, lw.FinalLayerNormBias));
        var ffnOut = Ffn(normed3, lw);

        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++) row[d] = afterCross[i][d] + ffnOut[i][d];
            output[i] = row;
        });
        return output;
    }

    /// <summary>Real causal self-attention: full MHA (no GQA), standard `1/sqrt(headDim)` scaling, no RoPE.</summary>
    private static float[][] SelfAttentionCausal(float[][] x, ParlerDecoderLayerWeights lw)
    {
        int t = x.Length;
        int dim = ParlerDecoderWeights.HiddenDim;
        int heads = ParlerDecoderWeights.NumHeads;

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        Parallel.For(0, t, i =>
        {
            q[i] = LinearQ8_0(x[i], lw.SelfAttnQWeight, dim, dim);
            k[i] = LinearQ8_0(x[i], lw.SelfAttnKWeight, dim, dim);
            v[i] = LinearQ8_0(x[i], lw.SelfAttnVWeight, dim, dim);
        });

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[dim];

        float scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, heads, h =>
        {
            int off = h * HeadDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j <= i; j++) // causal
                    scores[j] = Dot(q[i], k[j], off, HeadDim) * scale;
                SoftmaxInPlace(scores, i + 1);

                var ctxSpan = context[i].AsSpan(off, HeadDim);
                for (int j = 0; j <= i; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, HeadDim), scores[j], ctxSpan, ctxSpan);
            }
        });

        var output = new float[t][];
        Parallel.For(0, t, i => output[i] = LinearQ8_0(context[i], lw.SelfAttnOutWeight, dim, dim));
        return output;
    }

    /// <summary>Real cross-attention to the T5 encoder's output: full MHA, standard scaling, non-causal (attends to all encoder positions).</summary>
    private static float[][] CrossAttention(float[][] x, float[][] encoderHidden, ParlerDecoderLayerWeights lw)
    {
        int tq = x.Length;
        int tk = encoderHidden.Length;
        int dim = ParlerDecoderWeights.HiddenDim;
        int heads = ParlerDecoderWeights.NumHeads;

        var q = new float[tq][];
        Parallel.For(0, tq, i => q[i] = LinearQ8_0(x[i], lw.CrossAttnQWeight, dim, dim));
        var k = new float[tk][];
        var v = new float[tk][];
        Parallel.For(0, tk, j =>
        {
            k[j] = LinearQ8_0(encoderHidden[j], lw.CrossAttnKWeight, dim, dim);
            v[j] = LinearQ8_0(encoderHidden[j], lw.CrossAttnVWeight, dim, dim);
        });

        var context = new float[tq][];
        for (int i = 0; i < tq; i++) context[i] = new float[dim];

        float scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, heads, h =>
        {
            int off = h * HeadDim;
            var scores = new float[tk];
            for (int i = 0; i < tq; i++)
            {
                for (int j = 0; j < tk; j++)
                    scores[j] = Dot(q[i], k[j], off, HeadDim) * scale;
                SoftmaxInPlace(scores, tk);

                var ctxSpan = context[i].AsSpan(off, HeadDim);
                for (int j = 0; j < tk; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, HeadDim), scores[j], ctxSpan, ctxSpan);
            }
        });

        var output = new float[tq][];
        Parallel.For(0, tq, i => output[i] = LinearQ8_0(context[i], lw.CrossAttnOutWeight, dim, dim));
        return output;
    }

    /// <summary>Real plain (non-gated) FFN: `fc2(gelu(fc1(x)))`, no bias.</summary>
    private static float[][] Ffn(float[][] x, ParlerDecoderLayerWeights lw)
    {
        int t = x.Length;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var h = LinearQ8_0(x[i], lw.Fc1Weight, ParlerDecoderWeights.HiddenDim, ParlerDecoderWeights.FfnDim);
            for (int d = 0; d < h.Length; d++) h[d] = Gelu(h[d]);
            output[i] = LinearQ8_0(h, lw.Fc2Weight, ParlerDecoderWeights.FfnDim, ParlerDecoderWeights.HiddenDim);
        });
        return output;
    }

    /// <summary>Real (exact, erf-based) GELU -- HF's default "gelu" activation, NOT the tanh approximation ("gelu_new") T5 uses.</summary>
    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    private static float Erf(float x)
    {
        // Abramowitz-Stegun 7.1.26 approximation, max error ~1.5e-7 -- sufficient for F32 activations.
        float sign = MathF.Sign(x);
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }

    private static float Dot(float[] a, float[] b, int offset, int len) =>
        System.Numerics.Tensors.TensorPrimitives.Dot(a.AsSpan(offset, len), b.AsSpan(offset, len));

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    /// <summary>Dtype-generic fused mat-vec (see IQuantWeightRef's doc comment) -- applied to the decoder's 8 big per-layer matrices. Works unchanged whether those matrices came from a Safetensors-sourced, re-quantized-to-Q8_0 loader or a GGUF-sourced loader reading its own real on-disk dtype directly.</summary>
    private static float[] LinearQ8_0(float[] input, IQuantWeightRef weight, int inDim, int outDim) =>
        weight.MatVec(input, inDim, outDim);

    /// <summary>Real `nn.LayerNorm`: mean-subtract, variance-normalize, scale + bias (NOT RMSNorm).</summary>
    private static unsafe float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = ParlerDecoderWeights.LayerNormEps)
    {
        var output = new float[x.Length];
        fixed (float* xp = x, wp = weight, bp = bias, op = output)
        {
            SimdKernels.LayerNorm(op, xp, wp, bp, x.Length, eps);
        }
        return output;
    }

    private static void SoftmaxInPlace(float[] scores, int count)
    {
        var span = scores.AsSpan(0, count);
        float max = System.Numerics.Tensors.TensorPrimitives.Max(span);
        float sum = 0f;
        for (int i = 0; i < count; i++)
        {
            float e = MathF.Exp(span[i] - max);
            span[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        System.Numerics.Tensors.TensorPrimitives.Multiply(span, invSum, span);
    }
}
