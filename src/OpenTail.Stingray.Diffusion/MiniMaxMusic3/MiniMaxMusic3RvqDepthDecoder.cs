using System.Numerics.Tensors;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 "local" language model (`MiniMaxMusic3RVQDepthDecoder`), transcribed
/// directly from the real, already-installed `diffusers==0.40.0` source
/// (`diffusers/models/transformers/minimax_music3_rvq_depth_decoder.py`) -- see
/// docs/066-minimax-music3-future-plan.md for the full archaeology.
///
/// <para>Within each audio frame, autoregressively predicts the seven residual RVQ codebooks
/// (c1..c7) from the global language model's hidden state plus the frame's semantic code, and
/// exposes its per-step hidden states -- these are the real "Local" hidden states this project's
/// original plan doc described as needing a fusion step with the Global model's own hidden states;
/// per the real source, that "fusion" is just concatenation of this class's per-step hidden states
/// with the Global model's, no separate module (see docs/066's "hidden-state fusion" finding).</para>
///
/// <para><b>Real, simple architecture, no RoPE</b>: unlike every other real transformer this
/// project has ported this session, this one uses LEARNED absolute positional embeddings
/// (`pos_embedding`, `max_position_embeddings=16` -- one call per depth step: the projected global
/// hidden state followed by up to 7 embedded residual codes) added at the INPUT, not rotary
/// embeddings. Standard causal full self-attention (no GQA -- `heads=16`, `head_dim=256`), standard
/// pre-norm RMSNorm blocks, standard SwiGLU MLP.</para>
/// </summary>
public sealed class MiniMaxMusic3RvqDepthDecoderWeights
{
    public required float[] AudioEmbeddingsWeight { get; init; } // [audioVocabSize*(numCodebooks-1), hidden] = [7168,4096]
    public required CfmLinearWeight ProjectionWeight { get; init; } // no bias
    public required float[] PosEmbeddingWeight { get; init; } // [maxPositionEmbeddings(16), hidden]
    public required DepthDecoderLayerWeights[] Layers { get; init; } // 4 real layers
    public required float[] NormWeight { get; init; }
    public required CfmLinearWeight[] AudioHeads { get; init; } // 7 real heads, one per residual codebook

    public static MiniMaxMusic3RvqDepthDecoderWeights Load(SafetensorsLoader loader)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        var layers = new DepthDecoderLayerWeights[MiniMaxMusic3Config.RvqDepthDecoderNumLayers];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = LoadLayer(loader, $"layers.{i}");

        int numResidualCodebooks = MiniMaxMusic3Config.RvqDepthDecoderNumCodebooks - 1; // 7
        int vocabSize = MiniMaxMusic3Config.RvqDepthDecoderAudioVocabSize;
        var audioHeads = new CfmLinearWeight[numResidualCodebooks];
        for (int i = 0; i < numResidualCodebooks; i++)
            audioHeads[i] = CfmLinearWeight.FromF32(loader.ReadF32($"audio_heads.{i}.weight"), outDim: vocabSize, inDim: hidden);

        return new MiniMaxMusic3RvqDepthDecoderWeights
        {
            AudioEmbeddingsWeight = loader.ReadF32("audio_embeddings.weight"),
            ProjectionWeight = CfmLinearWeight.FromF32(loader.ReadF32("projection.weight"), outDim: hidden, inDim: hidden),
            PosEmbeddingWeight = loader.ReadF32("pos_embedding.weight"),
            Layers = layers,
            NormWeight = loader.ReadF32("norm.weight"),
            AudioHeads = audioHeads,
        };
    }

    private static DepthDecoderLayerWeights LoadLayer(SafetensorsLoader loader, string p)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int ffn = MiniMaxMusic3Config.RvqDepthDecoderIntermediateSize;
        return new DepthDecoderLayerWeights
        {
            InputLayerNormWeight = loader.ReadF32($"{p}.input_layernorm.weight"),
            QWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.attn.to_q.weight"), outDim: hidden, inDim: hidden),
            KWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.attn.to_k.weight"), outDim: hidden, inDim: hidden),
            VWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.attn.to_v.weight"), outDim: hidden, inDim: hidden),
            OWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.attn.to_out.weight"), outDim: hidden, inDim: hidden),
            PostAttnLayerNormWeight = loader.ReadF32($"{p}.post_attention_layernorm.weight"),
            GateWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.gate_proj.weight"), outDim: ffn, inDim: hidden),
            UpWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.up_proj.weight"), outDim: ffn, inDim: hidden),
            DownWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.down_proj.weight"), outDim: hidden, inDim: ffn),
        };
    }
}

public sealed class DepthDecoderLayerWeights
{
    public required float[] InputLayerNormWeight { get; init; }
    public required CfmLinearWeight QWeight { get; init; }
    public required CfmLinearWeight KWeight { get; init; }
    public required CfmLinearWeight VWeight { get; init; }
    public required CfmLinearWeight OWeight { get; init; }
    public required float[] PostAttnLayerNormWeight { get; init; }
    public required CfmLinearWeight GateWeight { get; init; }
    public required CfmLinearWeight UpWeight { get; init; }
    public required CfmLinearWeight DownWeight { get; init; }
}

public static class MiniMaxMusic3RvqDepthDecoder
{
    /// <summary>Real forward: `hidden_states = inputs_embeds + pos_embedding(arange(steps))`, then `numLayers` real causal (no RoPE) pre-norm SwiGLU transformer layers, then a final RMSNorm. Returns `[steps][hidden]` -- the last step's row feeds the next codebook's output head.</summary>
    public static float[][] Forward(MiniMaxMusic3RvqDepthDecoderWeights w, float[][] inputsEmbeds)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int steps = inputsEmbeds.Length;

        var x = new float[steps][];
        for (int t = 0; t < steps; t++)
        {
            var row = new float[hidden];
            for (int i = 0; i < hidden; i++) row[i] = inputsEmbeds[t][i] + w.PosEmbeddingWeight[t * hidden + i];
            x[t] = row;
        }

        for (int li = 0; li < w.Layers.Length; li++)
            x = Layer(w.Layers[li], x, steps);

        var output = new float[steps][];
        for (int t = 0; t < steps; t++)
        {
            output[t] = new float[hidden];
            RmsNorm(x[t], w.NormWeight, output[t], 1e-6f);
        }
        return output;
    }

    /// <summary>
    /// Incremental forward for a single depth step: adds pos_embedding for stepIndex, evaluates each layer with KV caching,
    /// appends K and V to cache, and returns the single output hidden state [hidden].
    /// </summary>
    public static unsafe float[] ForwardStep(
        MiniMaxMusic3RvqDepthDecoderWeights w,
        float[] inputEmbed,
        int stepIndex,
        MiniMaxMusic3RvqDepthKvCache cache)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        var x = new float[hidden];
        for (int i = 0; i < hidden; i++)
            x[i] = inputEmbed[i] + w.PosEmbeddingWeight[stepIndex * hidden + i];

        for (int li = 0; li < w.Layers.Length; li++)
            x = LayerIncremental(w.Layers[li], x, stepIndex, cache.Keys[li], cache.Values[li]);

        var output = new float[hidden];
        RmsNorm(x, w.NormWeight, output, 1e-6f);
        return output;
    }

    /// <summary>Incremental forward for BOTH CFG branches (conditional + unconditional) of a single
    /// depth step in one call. Real, measured motivation: each of the 6 major per-layer matmuls
    /// (Q/K/V/O/gate/up[+down]) reads a full weight matrix (up to 64MB for the 4096x4096 attention
    /// projections -- far larger than this machine's L3 cache) off RAM; calling <see cref="ForwardStep"/>
    /// twice (once per branch) re-streams every one of those matrices from RAM a second time for no
    /// reason, since both branches read the identical weights. This calls
    /// <see cref="CfmLinearWeight.MatMulPairRowMajor"/> instead, which streams each weight row ONCE
    /// and applies it to both branches' inputs -- halving both the RAM traffic and the number of
    /// `Parallel.For` dispatches per step (a real, separate cost at this call frequency: 7 steps x 4
    /// layers x 200 frames). Attention is still computed per-branch (KV caches are branch-specific),
    /// but batched into one `Parallel.For` dispatch over both branches' heads together, mirroring
    /// <see cref="MiniMaxMusic3Transformer.ForwardPair"/>'s existing CFG-batching pattern for the
    /// Flow DiT. Numerically identical to two separate <see cref="ForwardStep"/> calls -- see
    /// `MiniMaxMusic3RvqDepthDecoderGoldenParityTests.ForwardStepPair_MatchesForwardStep_BitForBit`.
    /// </summary>
    public static unsafe (float[] Cond, float[] Uncond) ForwardStepPair(
        MiniMaxMusic3RvqDepthDecoderWeights w,
        float[] condInputEmbed,
        float[] uncondInputEmbed,
        int stepIndex,
        MiniMaxMusic3RvqDepthKvCache condCache,
        MiniMaxMusic3RvqDepthKvCache uncondCache)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        var xCond = new float[hidden];
        var xUncond = new float[hidden];
        for (int i = 0; i < hidden; i++)
        {
            float pos = w.PosEmbeddingWeight[stepIndex * hidden + i];
            xCond[i] = condInputEmbed[i] + pos;
            xUncond[i] = uncondInputEmbed[i] + pos;
        }

        for (int li = 0; li < w.Layers.Length; li++)
        {
            (xCond, xUncond) = LayerIncrementalPair(
                w.Layers[li], xCond, xUncond, stepIndex,
                condCache.Keys[li], condCache.Values[li],
                uncondCache.Keys[li], uncondCache.Values[li]);
        }

        var outCond = new float[hidden];
        var outUncond = new float[hidden];
        RmsNorm(xCond, w.NormWeight, outCond, 1e-6f);
        RmsNorm(xUncond, w.NormWeight, outUncond, 1e-6f);
        return (outCond, outUncond);
    }

    private static unsafe (float[] Cond, float[] Uncond) LayerIncrementalPair(
        DepthDecoderLayerWeights lw,
        float[] xCond, float[] xUncond,
        int stepIndex,
        List<float[]> condKeyCache, List<float[]> condValCache,
        List<float[]> uncondKeyCache, List<float[]> uncondValCache)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;

        var normed1Cond = new float[hidden];
        var normed1Uncond = new float[hidden];
        RmsNorm(xCond, lw.InputLayerNormWeight, normed1Cond, 1e-6f);
        RmsNorm(xUncond, lw.InputLayerNormWeight, normed1Uncond, 1e-6f);

        var (attnCond, attnUncond) = SelfAttentionIncrementalPair(
            lw, normed1Cond, normed1Uncond, stepIndex, condKeyCache, condValCache, uncondKeyCache, uncondValCache);

        var afterAttnCond = new float[hidden];
        var afterAttnUncond = new float[hidden];
        for (int i = 0; i < hidden; i++)
        {
            afterAttnCond[i] = xCond[i] + attnCond[i];
            afterAttnUncond[i] = xUncond[i] + attnUncond[i];
        }

        var normed2Cond = new float[hidden];
        var normed2Uncond = new float[hidden];
        RmsNorm(afterAttnCond, lw.PostAttnLayerNormWeight, normed2Cond, 1e-6f);
        RmsNorm(afterAttnUncond, lw.PostAttnLayerNormWeight, normed2Uncond, 1e-6f);

        var (mlpCond, mlpUncond) = MlpStepPair(lw, normed2Cond, normed2Uncond);

        var outCond = new float[hidden];
        var outUncond = new float[hidden];
        for (int i = 0; i < hidden; i++)
        {
            outCond[i] = afterAttnCond[i] + mlpCond[i];
            outUncond[i] = afterAttnUncond[i] + mlpUncond[i];
        }
        return (outCond, outUncond);
    }

    private static unsafe (float[] Cond, float[] Uncond) SelfAttentionIncrementalPair(
        DepthDecoderLayerWeights lw,
        float[] normedCond, float[] normedUncond,
        int stepIndex,
        List<float[]> condKeyCache, List<float[]> condValCache,
        List<float[]> uncondKeyCache, List<float[]> uncondValCache)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int numHeads = MiniMaxMusic3Config.RvqDepthDecoderNumHeads;
        int headDim = hidden / numHeads;
        float scale = MathF.Pow(headDim, -0.5f);

        var qCond = new float[hidden]; var qUncond = new float[hidden];
        var kCond = new float[hidden]; var kUncond = new float[hidden];
        var vCond = new float[hidden]; var vUncond = new float[hidden];
        fixed (float* npc = normedCond, npu = normedUncond,
                      qpc = qCond, qpu = qUncond, kpc = kCond, kpu = kUncond, vpc = vCond, vpu = vUncond)
        {
            lw.QWeight.MatMulPairRowMajor(npc, npu, qpc, qpu);
            lw.KWeight.MatMulPairRowMajor(npc, npu, kpc, kpu);
            lw.VWeight.MatMulPairRowMajor(npc, npu, vpc, vpu);
        }

        condKeyCache.Add(kCond); condValCache.Add(vCond);
        uncondKeyCache.Add(kUncond); uncondValCache.Add(vUncond);

        int totalLen = condKeyCache.Count; // same length as uncondKeyCache -- both branches step together
        var contextCond = new float[hidden];
        var contextUncond = new float[hidden];

        Parallel.For(0, 2 * numHeads, bh =>
        {
            bool cond = bh < numHeads;
            int h = cond ? bh : bh - numHeads;
            int off = h * headDim;
            var keyCache = cond ? condKeyCache : uncondKeyCache;
            var valCache = cond ? condValCache : uncondValCache;
            var q = cond ? qCond : qUncond;
            var context = cond ? contextCond : contextUncond;

            var scores = new float[totalLen];
            var qSpan = q.AsSpan(off, headDim);
            for (int j = 0; j < totalLen; j++)
            {
                var kSpan = keyCache[j].AsSpan(off, headDim);
                scores[j] = TensorPrimitives.Dot(qSpan, kSpan) * scale;
            }
            SoftmaxRange(scores, 0, totalLen);

            var ctxSpan = context.AsSpan(off, headDim);
            for (int j = 0; j < totalLen; j++)
            {
                float s = scores[j];
                var vSpan = valCache[j].AsSpan(off, headDim);
                TensorPrimitives.MultiplyAdd(vSpan, s, ctxSpan, ctxSpan);
            }
        });

        var outCond = new float[hidden];
        var outUncond = new float[hidden];
        fixed (float* cpc = contextCond, cpu = contextUncond, opc = outCond, opu = outUncond)
        {
            lw.OWeight.MatMulPairRowMajor(cpc, cpu, opc, opu);
        }
        return (outCond, outUncond);
    }

    private static unsafe (float[] Cond, float[] Uncond) MlpStepPair(DepthDecoderLayerWeights lw, float[] normedCond, float[] normedUncond)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int ffn = MiniMaxMusic3Config.RvqDepthDecoderIntermediateSize;

        var gateCond = new float[ffn]; var gateUncond = new float[ffn];
        var upCond = new float[ffn]; var upUncond = new float[ffn];
        fixed (float* npc = normedCond, npu = normedUncond,
                      gpc = gateCond, gpu = gateUncond, upc = upCond, upu = upUncond)
        {
            lw.GateWeight.MatMulPairRowMajor(npc, npu, gpc, gpu);
            lw.UpWeight.MatMulPairRowMajor(npc, npu, upc, upu);
        }
        for (int i = 0; i < ffn; i++)
        {
            gateCond[i] = Silu(gateCond[i]) * upCond[i];
            gateUncond[i] = Silu(gateUncond[i]) * upUncond[i];
        }

        var outCond = new float[hidden];
        var outUncond = new float[hidden];
        fixed (float* gpc = gateCond, gpu = gateUncond, opc = outCond, opu = outUncond)
        {
            lw.DownWeight.MatMulPairRowMajor(gpc, gpu, opc, opu);
        }
        return (outCond, outUncond);
    }

    /// <summary>Embeds a real residual code (real codebook index `codebookIdx` in `[0,6]`, real code value in `[0,audioVocabSize)`) via the real `audio_embeddings` table, real row offset `codebookIdx*audioVocabSize + code` (confirmed from the real embedding table's shape `[audioVocabSize*(numCodebooks-1), hidden]`).</summary>
    public static float[] EmbedResidualCode(MiniMaxMusic3RvqDepthDecoderWeights w, int codebookIdx, int code)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int vocabSize = MiniMaxMusic3Config.RvqDepthDecoderAudioVocabSize;
        var row = new float[hidden];
        Array.Copy(w.AudioEmbeddingsWeight, (long)(codebookIdx * vocabSize + code) * hidden, row, 0, hidden);
        return row;
    }

    /// <summary>Real per-step projection applied before feeding an embedded step into the depth decoder: `projection(x)`, no bias.</summary>
    public static unsafe float[] Project(MiniMaxMusic3RvqDepthDecoderWeights w, float[] x)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        var output = new float[hidden];
        fixed (float* xp = x, op = output)
            w.ProjectionWeight.MatMul(xp, 1, op);
        return output;
    }

    /// <summary>Real logits for residual codebook `codebookIdx` (`[0,6]`) from the depth decoder's LAST step hidden state.</summary>
    public static unsafe float[] CodebookLogits(MiniMaxMusic3RvqDepthDecoderWeights w, float[] lastStepHidden, int codebookIdx)
    {
        int vocabSize = MiniMaxMusic3Config.RvqDepthDecoderAudioVocabSize;
        var logits = new float[vocabSize];
        fixed (float* xp = lastStepHidden, op = logits)
            w.AudioHeads[codebookIdx].MatMul(xp, 1, op);
        return logits;
    }

    private static unsafe float[][] Layer(DepthDecoderLayerWeights lw, float[][] x, int seqLen)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;

        var normed1 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { normed1[t] = new float[hidden]; RmsNorm(x[t], lw.InputLayerNormWeight, normed1[t], 1e-6f); }

        var attnOut = SelfAttentionCausal(lw, normed1, seqLen);
        var afterAttn = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            afterAttn[t] = new float[hidden];
            for (int i = 0; i < hidden; i++) afterAttn[t][i] = x[t][i] + attnOut[t][i];
        }

        var normed2 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { normed2[t] = new float[hidden]; RmsNorm(afterAttn[t], lw.PostAttnLayerNormWeight, normed2[t], 1e-6f); }

        var mlpOut = Mlp(lw, normed2, seqLen);
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            output[t] = new float[hidden];
            for (int i = 0; i < hidden; i++) output[t][i] = afterAttn[t][i] + mlpOut[t][i];
        }
        return output;
    }

    private static unsafe float[][] SelfAttentionCausal(DepthDecoderLayerWeights lw, float[][] normed, int seqLen)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int numHeads = MiniMaxMusic3Config.RvqDepthDecoderNumHeads;
        int headDim = hidden / numHeads;
        float scale = MathF.Pow(headDim, -0.5f);

        var flat = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++) Array.Copy(normed[t], 0, flat, t * hidden, hidden);

        var q = new float[seqLen * hidden];
        var k = new float[seqLen * hidden];
        var v = new float[seqLen * hidden];
        fixed (float* fp = flat, qp = q, kp = k, vp = v)
        {
            lw.QWeight.MatMul(fp, seqLen, qp);
            lw.KWeight.MatMul(fp, seqLen, kp);
            lw.VWeight.MatMul(fp, seqLen, vp);
        }

        var context = new float[seqLen * hidden];
        Parallel.For(0, numHeads, h =>
        {
            int off = h * headDim;
            var scores = new float[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                var qSpan = q.AsSpan(i * hidden + off, headDim);
                for (int j = 0; j <= i; j++) // real causal mask
                {
                    var kSpan = k.AsSpan(j * hidden + off, headDim);
                    scores[j] = TensorPrimitives.Dot(qSpan, kSpan) * scale;
                }
                SoftmaxRange(scores, 0, i + 1);

                var ctxSpan = context.AsSpan(i * hidden + off, headDim);
                for (int j = 0; j <= i; j++)
                {
                    float s = scores[j];
                    var vSpan = v.AsSpan(j * hidden + off, headDim);
                    TensorPrimitives.MultiplyAdd(vSpan, s, ctxSpan, ctxSpan);
                }
            }
        });

        var output = new float[seqLen * hidden];
        fixed (float* cp = context, op = output)
            lw.OWeight.MatMul(cp, seqLen, op);

        var rows = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { rows[t] = new float[hidden]; Array.Copy(output, t * hidden, rows[t], 0, hidden); }
        return rows;
    }

    private static unsafe float[] LayerIncremental(
        DepthDecoderLayerWeights lw,
        float[] x,
        int stepIndex,
        List<float[]> keyCache,
        List<float[]> valCache)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;

        var normed1 = new float[hidden];
        RmsNorm(x, lw.InputLayerNormWeight, normed1, 1e-6f);

        var attnOut = SelfAttentionIncremental(lw, normed1, stepIndex, keyCache, valCache);
        var afterAttn = new float[hidden];
        for (int i = 0; i < hidden; i++) afterAttn[i] = x[i] + attnOut[i];

        var normed2 = new float[hidden];
        RmsNorm(afterAttn, lw.PostAttnLayerNormWeight, normed2, 1e-6f);

        var mlpOut = MlpStep(lw, normed2);
        var output = new float[hidden];
        for (int i = 0; i < hidden; i++) output[i] = afterAttn[i] + mlpOut[i];
        return output;
    }

    private static unsafe float[] SelfAttentionIncremental(
        DepthDecoderLayerWeights lw,
        float[] normed,
        int stepIndex,
        List<float[]> keyCache,
        List<float[]> valCache)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int numHeads = MiniMaxMusic3Config.RvqDepthDecoderNumHeads;
        int headDim = hidden / numHeads;
        float scale = MathF.Pow(headDim, -0.5f);

        var q = new float[hidden];
        var k = new float[hidden];
        var v = new float[hidden];
        fixed (float* np = normed, qp = q, kp = k, vp = v)
        {
            lw.QWeight.MatMul(np, 1, qp);
            lw.KWeight.MatMul(np, 1, kp);
            lw.VWeight.MatMul(np, 1, vp);
        }

        keyCache.Add(k);
        valCache.Add(v);

        int totalLen = keyCache.Count;
        var context = new float[hidden];

        Parallel.For(0, numHeads, h =>
        {
            int off = h * headDim;
            var scores = new float[totalLen];
            var qSpan = q.AsSpan(off, headDim);

            for (int j = 0; j < totalLen; j++)
            {
                var kSpan = keyCache[j].AsSpan(off, headDim);
                scores[j] = TensorPrimitives.Dot(qSpan, kSpan) * scale;
            }
            SoftmaxRange(scores, 0, totalLen);

            var ctxSpan = context.AsSpan(off, headDim);
            for (int j = 0; j < totalLen; j++)
            {
                float s = scores[j];
                var vSpan = valCache[j].AsSpan(off, headDim);
                TensorPrimitives.MultiplyAdd(vSpan, s, ctxSpan, ctxSpan);
            }
        });

        var output = new float[hidden];
        fixed (float* cp = context, op = output)
            lw.OWeight.MatMul(cp, 1, op);

        return output;
    }

    private static unsafe float[] MlpStep(DepthDecoderLayerWeights lw, float[] normed)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int ffn = MiniMaxMusic3Config.RvqDepthDecoderIntermediateSize;

        var gate = new float[ffn];
        var up = new float[ffn];
        fixed (float* np = normed, gp = gate, up_ = up)
        {
            lw.GateWeight.MatMul(np, 1, gp);
            lw.UpWeight.MatMul(np, 1, up_);
        }
        for (int i = 0; i < gate.Length; i++) gate[i] = Silu(gate[i]) * up[i];

        var output = new float[hidden];
        fixed (float* gp = gate, op = output)
            lw.DownWeight.MatMul(gp, 1, op);

        return output;
    }

    private static unsafe float[][] Mlp(DepthDecoderLayerWeights lw, float[][] normed, int seqLen)
    {
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        int ffn = MiniMaxMusic3Config.RvqDepthDecoderIntermediateSize;

        var flat = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++) Array.Copy(normed[t], 0, flat, t * hidden, hidden);

        var gate = new float[seqLen * ffn];
        var up = new float[seqLen * ffn];
        fixed (float* fp = flat, gp = gate, up_ = up)
        {
            lw.GateWeight.MatMul(fp, seqLen, gp);
            lw.UpWeight.MatMul(fp, seqLen, up_);
        }
        for (int i = 0; i < gate.Length; i++) gate[i] = Silu(gate[i]) * up[i];

        var output = new float[seqLen * hidden];
        fixed (float* gp = gate, op = output)
            lw.DownWeight.MatMul(gp, seqLen, op);

        var rows = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { rows[t] = new float[hidden]; Array.Copy(output, t * hidden, rows[t], 0, hidden); }
        return rows;
    }

    private static void RmsNorm(ReadOnlySpan<float> x, float[] weight, Span<float> output, float eps)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    private static void SoftmaxRange(float[] scores, int start, int end)
    {
        float max = float.NegativeInfinity;
        for (int i = start; i < end; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = start; i < end; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = start; i < end; i++) scores[i] *= invSum;
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));
}
