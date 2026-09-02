using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Diffusion.AceStep.Transformer;

/// <summary>
/// Real ACE-Step Turbo DiT forward pass (`AceStepDiTModel`), transcribed directly from the real
/// `modeling_acestep_v15_turbo.py` bundled in the `ACE-Step/Ace-Step1.5` HF repo as `custom_code`
/// (read directly, not reconstructed) -- see docs/064-acestep-implementation-plan.md.
///
/// <para><b>Real, easy-to-get-wrong structural facts, confirmed from source</b>: (1) this is a
/// BIDIRECTIONAL (non-causal) transformer -- the whole latent sequence is processed in one shot
/// every denoising step, not autoregressively; self-attention layers alternate
/// sliding-window/full per <see cref="AceStepConfig.IsSlidingLayer"/>, both bidirectional (real
/// `create_4d_mask(..., is_causal=False)`). (2) Self-attention gets RoPE + a gated residual
/// (`hidden + attn_output * gate_msa`); cross-attention gets NO RoPE and a PLAIN (ungated)
/// residual (`hidden + attn_output`, no `gate` multiply at all -- easy to miss since self-attn and
/// MLP both DO gate). (3) Q/K RMSNorm is applied per-head (over `head_dim`, after the
/// proj+reshape-to-heads, BEFORE RoPE) to BOTH self- and cross-attention. (4) AdaLN modulation:
/// each layer's `scale_shift_table` (`[6,hidden]`) is added to the SAME shared `timestep_proj`
/// (`[6,hidden]`, computed ONCE in <see cref="Forward"/> and passed unchanged to every layer, not
/// per-layer-distinct) and split into 6 chunks (`shift_msa,scale_msa,gate_msa,c_shift_msa,
/// c_scale_msa,c_gate_msa`) -- self-attn and MLP each get their own adaptive norm
/// (`norm(x)*(1+scale)+shift`) from this same split. (5) `timestep_r` in real Turbo inference
/// (`generate_audio`) is always called with `timestep_r=timestep` (same value), so `t - r = 0`
/// always -- `TimeEmbedR`'s contribution is a constant (sinusoidal embedding of 0), only
/// meaningful for the real meanflow TRAINING objective, not inference; still computed for
/// correctness rather than special-cased away, since it's cheap and matches the real reference
/// exactly rather than assuming.</para>
///
/// <para><b>Cross-attention K/V caching across denoising steps</b> (a real, deliberate
/// optimization in the reference worth replicating, not just a nicety): conditioning
/// (`encoder_hidden_states`) never changes across the real 8-step Euler loop, so cross-attention
/// K/V is computed ONCE via <see cref="PrepareCrossAttention"/> and reused every step -- same
/// shape as MusicGen/AudioGen's `PrepareCrossAttention` pattern in this codebase, applied here to
/// a bidirectional DiT instead of an autoregressive LM.</para>
/// </summary>
public static class AceStepDiT
{
    /// <summary>Precomputed, reusable-across-steps state: cross-attention K/V (from the fixed condition sequence) plus RoPE tables for the current sequence length.</summary>
    public sealed class Context
    {
        public required float[][] CrossK { get; init; } // [layer][condLen * kvDim]
        public required float[][] CrossV { get; init; }
        public required int CondLen { get; init; }
        public required float[] Cos { get; init; } // [seqLen * headDim]
        public required float[] Sin { get; init; }
        public required int SeqLen { get; init; }
    }

    /// <summary>Precomputes cross-attention K/V (per-head Q/K-RMSNorm applied, NO RoPE) from the condition-encoder's packed output, plus the self-attention RoPE cos/sin tables for a given patchified sequence length. Both are constant across all real Euler steps.</summary>
    public static unsafe Context PrepareCrossAttention(AceStepDiTWeights w, float[][] encoderHiddenStates, int seqLen)
    {
        int hidden = AceStepConfig.HiddenSize;
        int condLen = encoderHiddenStates.Length;
        int kvDim = AceStepConfig.NumKeyValueHeads * AceStepConfig.HeadDim;

        var flat = new float[condLen * hidden];
        for (int i = 0; i < condLen; i++) Array.Copy(encoderHiddenStates[i], 0, flat, i * hidden, hidden);

        var crossK = new float[w.Layers.Length][];
        var crossV = new float[w.Layers.Length][];
        fixed (float* fp = flat)
        {
            for (int l = 0; l < w.Layers.Length; l++)
            {
                var attn = w.Layers[l].CrossAttn;
                var k = new float[condLen * kvDim];
                var v = new float[condLen * kvDim];
                fixed (float* kp = k, vp = v)
                {
                    attn.KWeight.MatMul(fp, condLen, kp);
                    attn.VWeight.MatMul(fp, condLen, vp);
                }
                RmsNormPerHead(k, condLen, AceStepConfig.NumKeyValueHeads, AceStepConfig.HeadDim, attn.KNormWeight);
                crossK[l] = k;
                crossV[l] = v;
            }
        }

        var (cos, sin) = BuildRope(seqLen, AceStepConfig.HeadDim, AceStepConfig.RopeTheta);
        return new Context { CrossK = crossK, CrossV = crossV, CondLen = condLen, Cos = cos, Sin = sin, SeqLen = seqLen };
    }

    /// <summary>Runs one full DiT forward pass over the whole (already patchified) latent sequence: `hidden[seqLen][hidden]` (== `proj_in` output) -&gt; 24 layers -&gt; final AdaLN + norm. Returns the same-shaped `[seqLen][hidden]` pre-`proj_out` hidden states (patch-level, not yet de-patchified -- patchify/de-patchify via `proj_in`/`proj_out` is the caller's responsibility, not done in this method).</summary>
    public static float[][] Forward(AceStepDiTWeights w, float[][] hidden, float timestep, float timestepR, Context ctx)
    {
        int seqLen = hidden.Length;
        int hiddenSize = AceStepConfig.HiddenSize;

        var (tembT, timestepProjT) = TimestepEmbed(w.TimeEmbed, timestep);
        var (tembR, timestepProjR) = TimestepEmbed(w.TimeEmbedR, timestep - timestepR);
        var temb = new float[hiddenSize];
        var timestepProj = new float[6 * hiddenSize];
        for (int i = 0; i < hiddenSize; i++) temb[i] = tembT[i] + tembR[i];
        for (int i = 0; i < timestepProj.Length; i++) timestepProj[i] = timestepProjT[i] + timestepProjR[i];

        var x = hidden;
        for (int li = 0; li < w.Layers.Length; li++)
            x = Layer(w.Layers[li], x, seqLen, li, timestepProj, ctx);

        // Final AdaLN + norm (per-token, `temb` unsqueezed to broadcast over the sequence).
        var shift = new float[hiddenSize];
        var scale = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            shift[i] = w.ScaleShiftTable[i] + temb[i];             // ScaleShiftTable[0,:] is "shift"
            scale[i] = w.ScaleShiftTable[hiddenSize + i] + temb[i]; // ScaleShiftTable[1,:] is "scale"
        }

        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var normed = new float[hiddenSize];
            RmsNorm(x[t], w.NormOutWeight, normed, 1e-6f); // Qwen3RMSNorm eps default
            var row = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) row[i] = normed[i] * (1f + scale[i]) + shift[i];
            output[t] = row;
        }
        return output;
    }

    private static unsafe float[][] Layer(AceStepDiTLayerWeights lw, float[][] x, int seqLen, int layerIndex, float[] timestepProj, Context ctx)
    {
        int hidden = AceStepConfig.HiddenSize;

        // scale_shift_table[6,hidden] + timestepProj[6,hidden] (broadcast over the sequence -- both are per-token-constant), chunk into 6.
        var mod = new float[6 * hidden];
        for (int i = 0; i < mod.Length; i++) mod[i] = lw.ScaleShiftTable[i] + timestepProj[i];
        var shiftMsa = mod.AsSpan(0, hidden);
        var scaleMsa = mod.AsSpan(hidden, hidden);
        var gateMsa = mod.AsSpan(2 * hidden, hidden);
        var cShiftMsa = mod.AsSpan(3 * hidden, hidden);
        var cScaleMsa = mod.AsSpan(4 * hidden, hidden);
        var cGateMsa = mod.AsSpan(5 * hidden, hidden);

        // Step 1: self-attention with AdaLN, gated residual.
        var normed1 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var n = new float[hidden];
            RmsNorm(x[t], lw.SelfAttnNormWeight, n, 1e-6f);
            for (int i = 0; i < hidden; i++) n[i] = n[i] * (1f + scaleMsa[i]) + shiftMsa[i];
            normed1[t] = n;
        }
        bool sliding = AceStepConfig.IsSlidingLayer(layerIndex);
        var selfOut = SelfAttention(lw.SelfAttn, normed1, seqLen, ctx, sliding);
        var afterSelf = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var row = new float[hidden];
            for (int i = 0; i < hidden; i++) row[i] = x[t][i] + selfOut[t][i] * gateMsa[i];
            afterSelf[t] = row;
        }

        // Step 2: cross-attention, PLAIN (ungated) residual -- real reference has no gate here.
        var normed2 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var n = new float[hidden];
            RmsNorm(afterSelf[t], lw.CrossAttnNormWeight, n, 1e-6f);
            normed2[t] = n;
        }
        var crossOut = CrossAttention(lw.CrossAttn, normed2, seqLen, layerIndex, ctx);
        var afterCross = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var row = new float[hidden];
            for (int i = 0; i < hidden; i++) row[i] = afterSelf[t][i] + crossOut[t][i];
            afterCross[t] = row;
        }

        // Step 3: MLP with AdaLN, gated residual.
        var normed3 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var n = new float[hidden];
            RmsNorm(afterCross[t], lw.MlpNormWeight, n, 1e-6f);
            for (int i = 0; i < hidden; i++) n[i] = n[i] * (1f + cScaleMsa[i]) + cShiftMsa[i];
            normed3[t] = n;
        }
        var mlpOut = Mlp(lw, normed3, seqLen);
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var row = new float[hidden];
            for (int i = 0; i < hidden; i++) row[i] = afterCross[t][i] + mlpOut[t][i] * cGateMsa[i];
            output[t] = row;
        }
        return output;
    }

    private static unsafe float[][] SelfAttention(AceStepAttentionWeights attn, float[][] normed, int seqLen, Context ctx, bool sliding)
    {
        int hidden = AceStepConfig.HiddenSize;
        int numHeads = AceStepConfig.NumAttentionHeads;
        int numKvHeads = AceStepConfig.NumKeyValueHeads;
        int headDim = AceStepConfig.HeadDim;
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int groups = numHeads / numKvHeads;
        float scale = MathF.Pow(headDim, -0.5f);

        var flat = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++) Array.Copy(normed[t], 0, flat, t * hidden, hidden);

        var q = new float[seqLen * qDim];
        var k = new float[seqLen * kvDim];
        var v = new float[seqLen * kvDim];
        fixed (float* fp = flat, qp = q, kp = k, vp = v)
        {
            attn.QWeight.MatMul(fp, seqLen, qp);
            attn.KWeight.MatMul(fp, seqLen, kp);
            attn.VWeight.MatMul(fp, seqLen, vp);
        }
        RmsNormPerHead(q, seqLen, numHeads, headDim, attn.QNormWeight);
        RmsNormPerHead(k, seqLen, numKvHeads, headDim, attn.KNormWeight);
        ApplyRope(q, seqLen, numHeads, headDim, ctx.Cos, ctx.Sin);
        ApplyRope(k, seqLen, numKvHeads, headDim, ctx.Cos, ctx.Sin);

        var context = new float[seqLen * qDim];
        int window = AceStepConfig.SlidingWindow;
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
            attn.OWeight.MatMul(cp, seqLen, op);

        var rows = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { rows[t] = new float[hidden]; Array.Copy(output, t * hidden, rows[t], 0, hidden); }
        return rows;
    }

    private static unsafe float[][] CrossAttention(AceStepAttentionWeights attn, float[][] normed, int seqLen, int layerIndex, Context ctx)
    {
        int hidden = AceStepConfig.HiddenSize;
        int numHeads = AceStepConfig.NumAttentionHeads;
        int numKvHeads = AceStepConfig.NumKeyValueHeads;
        int headDim = AceStepConfig.HeadDim;
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int groups = numHeads / numKvHeads;
        float scale = MathF.Pow(headDim, -0.5f);
        int condLen = ctx.CondLen;

        var flat = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++) Array.Copy(normed[t], 0, flat, t * hidden, hidden);

        var q = new float[seqLen * qDim];
        fixed (float* fp = flat, qp = q)
            attn.QWeight.MatMul(fp, seqLen, qp);
        RmsNormPerHead(q, seqLen, numHeads, headDim, attn.QNormWeight);
        // Real reference applies NO RoPE to cross-attention.

        var crossK = ctx.CrossK[layerIndex];
        var crossV = ctx.CrossV[layerIndex];

        var context = new float[seqLen * qDim];
        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / groups;
            int qOff = h * headDim;
            int kvOff = kvHead * headDim;
            var scores = new float[condLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < condLen; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i * qDim + qOff + d] * crossK[j * kvDim + kvOff + d];
                    scores[j] = dot * scale;
                }
                SoftmaxRange(scores, 0, condLen);

                var ctxSpan = context.AsSpan(i * qDim + qOff, headDim);
                for (int j = 0; j < condLen; j++)
                {
                    float s = scores[j];
                    var vSpan = crossV.AsSpan(j * kvDim + kvOff, headDim);
                    for (int d = 0; d < headDim; d++) ctxSpan[d] += s * vSpan[d];
                }
            }
        });

        var output = new float[seqLen * hidden];
        fixed (float* cp = context, op = output)
            attn.OWeight.MatMul(cp, seqLen, op);

        var rows = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { rows[t] = new float[hidden]; Array.Copy(output, t * hidden, rows[t], 0, hidden); }
        return rows;
    }

    /// <summary>Real Qwen3MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.</summary>
    private static unsafe float[][] Mlp(AceStepDiTLayerWeights lw, float[][] normed, int seqLen)
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

    /// <summary>Real `TimestepEmbedding.forward`: sinusoidal(dim=256, scale=1000) -&gt; linear_1 -&gt; SiLU -&gt; linear_2 (== `temb`, NO activation after) -&gt; `time_proj(SiLU(temb))` reshaped to `[6,hidden]` (== `timestep_proj`).</summary>
    private static unsafe (float[] Temb, float[] TimestepProj) TimestepEmbed(AceStepTimestepEmbeddingWeights w, float t)
    {
        var freq = SinusoidalTimestepEmbedding(t, dim: 256, scale: 1000f);
        var mid = new float[AceStepConfig.HiddenSize];
        fixed (float* fp = freq, mp = mid, b1 = w.Linear1Bias)
            w.Linear1Weight.MatMul(fp, 1, mp, b1);
        for (int i = 0; i < mid.Length; i++) mid[i] = Silu(mid[i]);

        var temb = new float[AceStepConfig.HiddenSize];
        fixed (float* mp = mid, tp = temb, b2 = w.Linear2Bias)
            w.Linear2Weight.MatMul(mp, 1, tp, b2);

        var act2 = new float[AceStepConfig.HiddenSize];
        for (int i = 0; i < temb.Length; i++) act2[i] = Silu(temb[i]);

        var timestepProj = new float[6 * AceStepConfig.HiddenSize];
        fixed (float* ap = act2, pp = timestepProj, b3 = w.TimeProjBias)
            w.TimeProjWeight.MatMul(ap, 1, pp, b3);

        return (temb, timestepProj);
    }

    /// <summary>Real `TimestepEmbedding.timestep_embedding`: `t *= scale`; `freqs = exp(-log(10000) * arange(half)/half)`; `cat([cos(t*freqs), sin(t*freqs)])`.</summary>
    private static float[] SinusoidalTimestepEmbedding(float t, int dim, float scale)
    {
        t *= scale;
        int half = dim / 2;
        var result = new float[dim];
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000f) * i / half);
            float arg = t * freq;
            result[i] = MathF.Cos(arg);
            result[half + i] = MathF.Sin(arg);
        }
        return result;
    }

    /// <summary>Real Qwen3-style RoPE table: `inv_freq[i] = theta^(-2i/headDim)`, position `p` in `[0,seqLen)`.</summary>
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
                // Real HF convention: cos/sin tables are duplicated across both halves of head_dim.
                cos[p * headDim + i] = c; cos[p * headDim + half + i] = c;
                sin[p * headDim + i] = s; sin[p * headDim + half + i] = s;
            }
        }
        return (cos, sin);
    }

    /// <summary>Real `apply_rotary_pos_emb` (rotate_half convention): `q_embed = q*cos + rotate_half(q)*sin`, `rotate_half(x) = cat(-x[half:], x[:half])`.</summary>
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

    /// <summary>Real per-head Q/K RMSNorm (`Qwen3RMSNorm(head_dim)`), applied independently to each head's `head_dim`-wide slice.</summary>
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

    /// <summary>
    /// Real `proj_in`: `torch.cat([context_latents, hidden_states], dim=-1)` (channel-concat,
    /// `contextLatents` and `noisyLatent` are both `[t][64]`, giving `[t][192]`), pad the TIME
    /// axis up to a multiple of `patch_size`(2) with zeros if needed, then a stride-2/kernel-2
    /// Conv1d (== non-overlapping patch pooling, no im2col needed since kernel==stride==padding-
    /// free). Returns `[t/2][hidden]` plus the real, unpadded `originalSeqLen` the caller needs
    /// to crop `ProjOut`'s output back to.
    /// </summary>
    public static unsafe (float[][] Patches, int OriginalSeqLen) ProjIn(AceStepDiTWeights w, float[][] contextLatents, float[][] noisyLatent)
    {
        int t = noisyLatent.Length;
        int inCh = AceStepConfig.InChannels; // 192 = contextLatents(128, == src_latents(64)+chunk_masks(64) already concatenated by the caller) + noisyLatent(64)
        int ctxCh = contextLatents[0].Length; // 128
        int hiddenCh = noisyLatent[0].Length; // 64

        int patch = AceStepConfig.PatchSize;
        int padded = t % patch == 0 ? t : t + (patch - t % patch);
        int outLen = padded / patch;
        int hidden = AceStepConfig.HiddenSize;

        var patches = new float[outLen][];
        for (int p = 0; p < outLen; p++)
        {
            // Gather this patch's `patch`-wide window of the concatenated [ctx|noisy] channels (zero past t).
            var window = new float[patch * inCh];
            for (int k = 0; k < patch; k++)
            {
                int ti = p * patch + k;
                if (ti >= t) continue; // zero-padding
                Array.Copy(contextLatents[ti], 0, window, k * inCh, ctxCh);
                Array.Copy(noisyLatent[ti], 0, window, k * inCh + ctxCh, hiddenCh);
            }

            // Conv1d weight layout [outCh, inCh, kernel]; window is [kernel, inCh] time-major -- reorder to [inCh,kernel] per output channel dot product.
            var row = new float[hidden];
            for (int oc = 0; oc < hidden; oc++)
            {
                float sum = w.ProjInBias[oc];
                int wBase = oc * inCh * patch;
                for (int ic = 0; ic < inCh; ic++)
                    for (int k = 0; k < patch; k++)
                        sum += w.ProjInWeight[wBase + ic * patch + k] * window[k * inCh + ic];
                row[oc] = sum;
            }
            patches[p] = row;
        }
        return (patches, t);
    }

    /// <summary>Real `proj_out`: ConvTranspose1d (kernel=stride=2, no overlap -- a plain per-patch expansion, no scatter-accumulate needed) de-patchifying `[t/2][hidden]` back to `[t][audioAcousticHiddenDim(64)]`, then cropped to `originalSeqLen`.</summary>
    public static float[][] ProjOut(AceStepDiTWeights w, float[][] patches, int originalSeqLen)
    {
        int patch = AceStepConfig.PatchSize;
        int hidden = AceStepConfig.HiddenSize;
        int outCh = AceStepConfig.AudioAcousticHiddenDim;

        var full = new float[patches.Length * patch][];
        for (int p = 0; p < patches.Length; p++)
        {
            for (int k = 0; k < patch; k++)
            {
                var row = new float[outCh];
                for (int oc = 0; oc < outCh; oc++)
                {
                    float sum = w.ProjOutBias[oc];
                    for (int ic = 0; ic < hidden; ic++)
                        // ConvTranspose1d weight layout [inCh(hidden), outCh, kernel]; kernel==stride so each output
                        // patch position k only ever receives contribution from this one input patch (no overlap-add).
                        sum += w.ProjOutWeight[(ic * outCh + oc) * patch + k] * patches[p][ic];
                    row[oc] = sum;
                }
                full[p * patch + k] = row;
            }
        }

        if (full.Length == originalSeqLen) return full;
        var cropped = new float[originalSeqLen][];
        Array.Copy(full, cropped, originalSeqLen);
        return cropped;
    }
}
