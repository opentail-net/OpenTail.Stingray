using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Diffusion.AceStep.Conditioning;

/// <summary>
/// Real ACE-Step condition encoder (`AceStepConditionEncoder`/`AceStepLyricEncoder`), transcribed
/// directly from the real `modeling_acestep_v15_turbo.py` -- see
/// docs/064-acestep-implementation-plan.md.
///
/// <para><b>V1 scope</b> (matches the plan's own V1 cut): text + lyrics only, NO timbre/reference-
/// audio conditioning. Real `AceStepConditionEncoder.forward` packs
/// `[lyric_hidden, timbre_hidden, text_hidden]` via a real sort-by-mask `pack_sequences` helper
/// (reorders valid-before-padding tokens for a batched, padded scenario) -- for V1's single,
/// unpadded prompt (no batching), that sort is a no-op (mask is all-1s throughout, stable sort
/// preserves order), so this class just concatenates `[lyricHidden, textHidden]` directly rather
/// than re-implementing the padding-aware sort for a case that never triggers it. Revisit if/when
/// real batching or padding is added.</para>
///
/// <para><b>Lyric encoder is NOT the same as the DiT's AdaLN layers</b>: real
/// `AceStepEncoderLayer` is a STANDARD pre-norm transformer block (`input_layernorm -&gt; self_attn
/// -&gt; +residual`, `post_attention_layernorm -&gt; mlp -&gt; +residual`), no timestep modulation at
/// all -- do not reuse `AceStepDiT`'s AdaLN math here. Same GQA/RoPE/per-head-QK-norm attention
/// primitives as the DiT (byte-identical formulas, real duplication -- worth extracting to a
/// shared kernel in a later DRY pass per CLAUDE.md rule 7, not done speculatively here with only
/// two real callers whose per-layer glue still differs, matching how MusicGen/AudioGen's
/// generation loop was left un-merged for the same reason).</para>
///
/// <para><b>Real lyric embedding path</b>: lyrics are embedded via Qwen3's raw token-embedding
/// LOOKUP (NOT the full Qwen3 forward pass -- confirmed from the real `diffusers` ACE-Step
/// pipeline, see docs/064's "Corrections and confirmations"), then projected
/// `text_hidden_dim(1024) -&gt; hidden_size(2048)` via `embed_tokens` (a `nn.Linear` WITH bias,
/// confusingly named the same as an embedding layer but it is not one), then run through 8 real
/// bidirectional (sliding/full alternating, same `layer_types` config as the DiT) transformer
/// layers.</para>
/// </summary>
public sealed class AceStepConditionEncoderWeights
{
    public required CfmLinearWeight TextProjectorWeight { get; init; } // no bias, real config

    public required CfmLinearWeight LyricEmbedWeight { get; init; } // WITH bias -- real nn.Linear, not an embedding table
    public required float[] LyricEmbedBias { get; init; }
    public required AceStepEncoderLayerWeights[] LyricLayers { get; init; } // 8 real layers
    public required float[] LyricNormWeight { get; init; }

    public static AceStepConditionEncoderWeights Load(SafetensorsLoader loader)
    {
        int hidden = AceStepConfig.HiddenSize;
        int textDim = AceStepConfig.TextHiddenDim;

        var lyricLayers = new AceStepEncoderLayerWeights[AceStepConfig.NumLyricEncoderHiddenLayers];
        for (int i = 0; i < lyricLayers.Length; i++)
            lyricLayers[i] = LoadEncoderLayer(loader, $"encoder.lyric_encoder.layers.{i}");

        return new AceStepConditionEncoderWeights
        {
            TextProjectorWeight = CfmLinearWeight.FromF32(loader.ReadF32("encoder.text_projector.weight"), outDim: hidden, inDim: textDim),
            LyricEmbedWeight = CfmLinearWeight.FromF32(loader.ReadF32("encoder.lyric_encoder.embed_tokens.weight"), outDim: hidden, inDim: textDim),
            LyricEmbedBias = loader.ReadF32("encoder.lyric_encoder.embed_tokens.bias"),
            LyricLayers = lyricLayers,
            LyricNormWeight = loader.ReadF32("encoder.lyric_encoder.norm.weight"),
        };
    }

    private static AceStepEncoderLayerWeights LoadEncoderLayer(SafetensorsLoader loader, string p)
    {
        int hidden = AceStepConfig.HiddenSize;
        int qDim = AceStepConfig.NumAttentionHeads * AceStepConfig.HeadDim;
        int kvDim = AceStepConfig.NumKeyValueHeads * AceStepConfig.HeadDim;
        int ffn = AceStepConfig.IntermediateSize;

        return new AceStepEncoderLayerWeights
        {
            InputLayerNormWeight = loader.ReadF32($"{p}.input_layernorm.weight"),
            QWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.q_proj.weight"), outDim: qDim, inDim: hidden),
            KWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.k_proj.weight"), outDim: kvDim, inDim: hidden),
            VWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.v_proj.weight"), outDim: kvDim, inDim: hidden),
            OWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.o_proj.weight"), outDim: hidden, inDim: qDim),
            QNormWeight = loader.ReadF32($"{p}.self_attn.q_norm.weight"),
            KNormWeight = loader.ReadF32($"{p}.self_attn.k_norm.weight"),
            PostAttnLayerNormWeight = loader.ReadF32($"{p}.post_attention_layernorm.weight"),
            MlpGateWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.gate_proj.weight"), outDim: ffn, inDim: hidden),
            MlpUpWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.up_proj.weight"), outDim: ffn, inDim: hidden),
            MlpDownWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.down_proj.weight"), outDim: hidden, inDim: ffn),
        };
    }
}

public sealed class AceStepEncoderLayerWeights
{
    public required float[] InputLayerNormWeight { get; init; }
    public required CfmLinearWeight QWeight { get; init; }
    public required CfmLinearWeight KWeight { get; init; }
    public required CfmLinearWeight VWeight { get; init; }
    public required CfmLinearWeight OWeight { get; init; }
    public required float[] QNormWeight { get; init; }
    public required float[] KNormWeight { get; init; }
    public required float[] PostAttnLayerNormWeight { get; init; }
    public required CfmLinearWeight MlpGateWeight { get; init; }
    public required CfmLinearWeight MlpUpWeight { get; init; }
    public required CfmLinearWeight MlpDownWeight { get; init; }
}

public static class AceStepConditionEncoder
{
    /// <summary>Real V1-scoped forward: project text hidden states, encode lyrics through the real 8-layer bidirectional encoder, concatenate `[lyric, text]` (real `pack_sequences`, simplified for V1's unpadded single-sequence case -- see class doc comment). Returns the packed condition sequence for the DiT's cross-attention.</summary>
    public static unsafe float[][] Forward(AceStepConditionEncoderWeights w, float[][] textHiddenStates, int[] lyricTokenIds, float[] qwen3TokenEmbeddingTable)
    {
        int hidden = AceStepConfig.HiddenSize;
        int textDim = AceStepConfig.TextHiddenDim;

        // Real: text_hidden_states -> text_projector (no bias).
        var textProjected = new float[textHiddenStates.Length][];
        for (int i = 0; i < textHiddenStates.Length; i++)
        {
            var row = new float[hidden];
            fixed (float* xp = textHiddenStates[i], rp = row)
                w.TextProjectorWeight.MatMul(xp, 1, rp);
            textProjected[i] = row;
        }

        // Real: lyric token ids -> raw Qwen3 embedding lookup -> embed_tokens (Linear, WITH bias) -> 8 bidirectional layers -> final norm.
        var lyricEmbeds = new float[lyricTokenIds.Length][];
        for (int i = 0; i < lyricTokenIds.Length; i++)
        {
            var raw = new float[textDim];
            Array.Copy(qwen3TokenEmbeddingTable, (long)lyricTokenIds[i] * textDim, raw, 0, textDim);
            var projected = new float[hidden];
            fixed (float* rp = raw, pp = projected, bp = w.LyricEmbedBias)
                w.LyricEmbedWeight.MatMul(rp, 1, pp, bp);
            lyricEmbeds[i] = projected;
        }

        var lyricHidden = EncodeLyrics(w, lyricEmbeds);

        var packed = new float[lyricHidden.Length + textProjected.Length][];
        Array.Copy(lyricHidden, packed, lyricHidden.Length);
        Array.Copy(textProjected, 0, packed, lyricHidden.Length, textProjected.Length);
        return packed;
    }

    private static float[][] EncodeLyrics(AceStepConditionEncoderWeights w, float[][] embeds)
    {
        int seqLen = embeds.Length;
        var (cos, sin) = BuildRope(seqLen, AceStepConfig.HeadDim, AceStepConfig.RopeTheta);

        var x = embeds;
        for (int li = 0; li < w.LyricLayers.Length; li++)
            x = EncoderLayer(w.LyricLayers[li], x, seqLen, AceStepConfig.IsSlidingLayer(li), cos, sin);

        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            output[t] = new float[AceStepConfig.HiddenSize];
            RmsNorm(x[t], w.LyricNormWeight, output[t], 1e-6f);
        }
        return output;
    }

    private static unsafe float[][] EncoderLayer(AceStepEncoderLayerWeights lw, float[][] x, int seqLen, bool sliding, float[] cos, float[] sin)
    {
        int hidden = AceStepConfig.HiddenSize;

        var normed1 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { normed1[t] = new float[hidden]; RmsNorm(x[t], lw.InputLayerNormWeight, normed1[t], 1e-6f); }

        var attnOut = SelfAttention(lw, normed1, seqLen, sliding, cos, sin);
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

    private static unsafe float[][] SelfAttention(AceStepEncoderLayerWeights lw, float[][] normed, int seqLen, bool sliding, float[] cos, float[] sin)
    {
        int hidden = AceStepConfig.HiddenSize;
        int numHeads = AceStepConfig.NumAttentionHeads;
        int numKvHeads = AceStepConfig.NumKeyValueHeads;
        int headDim = AceStepConfig.HeadDim;
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int groups = numHeads / numKvHeads;
        int window = AceStepConfig.SlidingWindow;
        float scale = MathF.Pow(headDim, -0.5f);

        var flat = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++) Array.Copy(normed[t], 0, flat, t * hidden, hidden);

        var q = new float[seqLen * qDim];
        var k = new float[seqLen * kvDim];
        var v = new float[seqLen * kvDim];
        fixed (float* fp = flat, qp = q, kp = k, vp = v)
        {
            lw.QWeight.MatMul(fp, seqLen, qp);
            lw.KWeight.MatMul(fp, seqLen, kp);
            lw.VWeight.MatMul(fp, seqLen, vp);
        }
        RmsNormPerHead(q, seqLen, numHeads, headDim, lw.QNormWeight);
        RmsNormPerHead(k, seqLen, numKvHeads, headDim, lw.KNormWeight);
        ApplyRope(q, seqLen, numHeads, headDim, cos, sin);
        ApplyRope(k, seqLen, numKvHeads, headDim, cos, sin);

        var context = new float[seqLen * qDim];
        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / groups;
            int qOff = h * headDim;
            int kvOff = kvHead * headDim;
            var scores = new float[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                int jStart = sliding ? Math.Max(0, i - window) : 0;
                int jEnd = sliding ? Math.Min(seqLen, i + window + 1) : seqLen;
                for (int j = jStart; j < jEnd; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i * qDim + qOff + d] * k[j * kvDim + kvOff + d];
                    scores[j] = dot * scale;
                }
                SoftmaxRange(scores, jStart, jEnd);

                var ctxSpan = context.AsSpan(i * qDim + qOff, headDim);
                for (int j = jStart; j < jEnd; j++)
                {
                    float s = scores[j];
                    var vSpan = v.AsSpan(j * kvDim + kvOff, headDim);
                    for (int d = 0; d < headDim; d++) ctxSpan[d] += s * vSpan[d];
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

    private static unsafe float[][] Mlp(AceStepEncoderLayerWeights lw, float[][] normed, int seqLen)
    {
        int hidden = AceStepConfig.HiddenSize;
        int ffn = AceStepConfig.IntermediateSize;

        var flat = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++) Array.Copy(normed[t], 0, flat, t * hidden, hidden);

        var gate = new float[seqLen * ffn];
        var up = new float[seqLen * ffn];
        fixed (float* fp = flat, gp = gate, up_ = up)
        {
            lw.MlpGateWeight.MatMul(fp, seqLen, gp);
            lw.MlpUpWeight.MatMul(fp, seqLen, up_);
        }
        for (int i = 0; i < gate.Length; i++) gate[i] = Silu(gate[i]) * up[i];

        var output = new float[seqLen * hidden];
        fixed (float* gp = gate, op = output)
            lw.MlpDownWeight.MatMul(gp, seqLen, op);

        var rows = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { rows[t] = new float[hidden]; Array.Copy(output, t * hidden, rows[t], 0, hidden); }
        return rows;
    }

    private static (float[] Cos, float[] Sin) BuildRope(int seqLen, int headDim, float theta)
    {
        int half = headDim / 2;
        var cos = new float[seqLen * headDim];
        var sin = new float[seqLen * headDim];
        for (int p = 0; p < seqLen; p++)
        {
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Pow(theta, -2f * i / headDim);
                float angle = p * invFreq;
                float c = MathF.Cos(angle), s = MathF.Sin(angle);
                cos[p * headDim + i] = c; cos[p * headDim + half + i] = c;
                sin[p * headDim + i] = s; sin[p * headDim + half + i] = s;
            }
        }
        return (cos, sin);
    }

    private static void ApplyRope(float[] qOrK, int seqLen, int numHeads, int headDim, float[] cos, float[] sin)
    {
        int half = headDim / 2;
        int rowDim = numHeads * headDim;
        for (int t = 0; t < seqLen; t++)
        {
            int cosBase = t * headDim;
            for (int h = 0; h < numHeads; h++)
            {
                int off = t * rowDim + h * headDim;
                for (int i = 0; i < half; i++)
                {
                    float x1 = qOrK[off + i];
                    float x2 = qOrK[off + half + i];
                    float c1 = cos[cosBase + i], s1 = sin[cosBase + i];
                    float c2 = cos[cosBase + half + i], s2 = sin[cosBase + half + i];
                    qOrK[off + i] = x1 * c1 - x2 * s1;
                    qOrK[off + half + i] = x2 * c2 + x1 * s2;
                }
            }
        }
    }

    private static void RmsNormPerHead(float[] qOrK, int seqLen, int numHeads, int headDim, float[] weight, float eps = 1e-6f)
    {
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = t * numHeads * headDim + h * headDim;
                var span = qOrK.AsSpan(off, headDim);
                float sumSq = 0f;
                for (int i = 0; i < headDim; i++) sumSq += span[i] * span[i];
                float invRms = 1f / MathF.Sqrt(sumSq / headDim + eps);
                for (int i = 0; i < headDim; i++) span[i] = span[i] * invRms * weight[i];
            }
        }
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
