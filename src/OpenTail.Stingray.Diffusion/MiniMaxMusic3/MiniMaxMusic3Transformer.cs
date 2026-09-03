namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 flow-matching DiT (`MiniMaxMusic3Transformer1DModel`), transcribed directly
/// from the real, already-installed `diffusers==0.40.0` source
/// (`diffusers/models/transformers/transformer_minimax_music3.py`) -- see
/// docs/066-minimax-music3-future-plan.md.
///
/// <para><b>Status: code written and real tensor names cross-checked against the checkpoint's own
/// `diffusion_pytorch_model.safetensors.index.json` (a free, small file -- no real weight download
/// needed for this), but NOT YET real-weight-tested</b>: the two big real downloads
/// (`language_model/` ~17.2GB, `transformer/` ~9.7GB) are blocked on this session's current disk
/// space. Written now so it is ready to golden-verify immediately once space is available, matching
/// this session's established "write structure from real source, verify once weights land" pattern
/// (used for `StableAudioMediumDiT` before its checkpoint finished downloading).</para>
///
/// <para><b>Real, remarkably simple mechanism -- NO AdaLN</b> (unlike every other real DiT this
/// project has ported this session, all of which use adaptive-norm timestep conditioning): the
/// timestep embedding is PREPENDED as one extra sequence token before the transformer blocks and
/// STRIPPED OFF after them -- a real, plain "timestep-as-a-token" scheme. Standard `LayerNorm`
/// pre-norm blocks (not RMSNorm), full bidirectional self-attention (no causal mask, no windowing),
/// partial GPT-J-style RoPE (`rotary_dim=32` of `head_dim=64`), and a real GLU-style FF (`ff_in`
/// produces `2*ff_inner_dim`, split into `[gate_states, gate]`, `gate_states * silu(gate)`).</para>
///
/// <para><b>Real input assembly</b>: `concat([noisy_latent(128), zeros(128),
/// condition.transpose(128)])` along the channel axis (`2*in_channels + condition_dim = 2304`
/// channels total) -&gt; `preprocess_conv` (real `Conv1d(k=1)`, no bias) added as a RESIDUAL (`conv(x)
/// + x`, not just `conv(x)`) -&gt; `proj_in` (Linear, no bias) to the real inner dim (`2048`). Output
/// side mirrors this: `proj_out` (Linear, no bias) back to `in_channels`(128) -&gt; `postprocess_conv`
/// (`Conv1d(k=1)`, no bias) added as a residual again.</para>
/// </summary>
public sealed class MiniMaxMusic3TransformerWeights
{
    public required float[] TimeProjWeight { get; init; } // MiniMaxMusic3FourierEmbedding: [fourierDim/2, 1], no bias
    public required TimestepEmbeddingWeights TimeEmbed { get; init; }
    public required float[] PreprocessConvWeight { get; init; } // Conv1d k=1, no bias, [concatChannels, concatChannels, 1]
    public required float[] ProjInWeight { get; init; } // Linear [innerDim, concatChannels], no bias
    public required MiniMaxMusic3TransformerBlockWeights[] Blocks { get; init; } // 36 real layers
    public required float[] ProjOutWeight { get; init; } // Linear [inChannels, innerDim], no bias
    public required float[] PostprocessConvWeight { get; init; } // Conv1d k=1, no bias, [inChannels, inChannels, 1]

    public static MiniMaxMusic3TransformerWeights Load(SafetensorsLoader loader)
    {
        var blocks = new MiniMaxMusic3TransformerBlockWeights[MiniMaxMusic3Config.TransformerNumLayers];
        for (int i = 0; i < blocks.Length; i++)
            blocks[i] = LoadBlock(loader, $"transformer_blocks.{i}");

        return new MiniMaxMusic3TransformerWeights
        {
            TimeProjWeight = loader.ReadF32("time_proj.weight"),
            TimeEmbed = new TimestepEmbeddingWeights
            {
                Linear1Weight = loader.ReadF32("time_embed.linear_1.weight"),
                Linear1Bias = loader.ReadF32("time_embed.linear_1.bias"),
                Linear2Weight = loader.ReadF32("time_embed.linear_2.weight"),
                Linear2Bias = loader.ReadF32("time_embed.linear_2.bias"),
            },
            PreprocessConvWeight = loader.ReadF32("preprocess_conv.weight"),
            ProjInWeight = loader.ReadF32("proj_in.weight"),
            Blocks = blocks,
            ProjOutWeight = loader.ReadF32("proj_out.weight"),
            PostprocessConvWeight = loader.ReadF32("postprocess_conv.weight"),
        };
    }

    private static MiniMaxMusic3TransformerBlockWeights LoadBlock(SafetensorsLoader loader, string p)
    {
        return new MiniMaxMusic3TransformerBlockWeights
        {
            Norm1Weight = loader.ReadF32($"{p}.norm1.weight"),
            Norm1Bias = loader.ReadF32($"{p}.norm1.bias"),
            QWeight = loader.ReadF32($"{p}.attn.to_q.weight"),
            KWeight = loader.ReadF32($"{p}.attn.to_k.weight"),
            VWeight = loader.ReadF32($"{p}.attn.to_v.weight"),
            OWeight = loader.ReadF32($"{p}.attn.to_out.0.weight"),
            Norm2Weight = loader.ReadF32($"{p}.norm2.weight"),
            Norm2Bias = loader.ReadF32($"{p}.norm2.bias"),
            FfInWeight = loader.ReadF32($"{p}.ff_in.weight"), // [2*ffn, inner]
            FfInBias = loader.ReadF32($"{p}.ff_in.bias"),
            FfOutWeight = loader.ReadF32($"{p}.ff_out.weight"), // [inner, ffn]
            FfOutBias = loader.ReadF32($"{p}.ff_out.bias"),
        };
    }
}

public sealed class TimestepEmbeddingWeights
{
    public required float[] Linear1Weight { get; init; }
    public required float[] Linear1Bias { get; init; }
    public required float[] Linear2Weight { get; init; }
    public required float[] Linear2Bias { get; init; }
}

public sealed class MiniMaxMusic3TransformerBlockWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] QWeight { get; init; }
    public required float[] KWeight { get; init; }
    public required float[] VWeight { get; init; }
    public required float[] OWeight { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] FfInWeight { get; init; }
    public required float[] FfInBias { get; init; }
    public required float[] FfOutWeight { get; init; }
    public required float[] FfOutBias { get; init; }
}

public static class MiniMaxMusic3Transformer
{
    /// <summary>Real forward: predicts the flow-matching velocity. `latent`/`condition` are
    /// time-major (`[length][channels]`); `condition` must already be resampled onto the latent
    /// timeline (real <see cref="MiniMaxMusic3ConditionEncoder"/> output). Pass an all-zero
    /// `condition` for the real unconditional CFG branch. Returns `[length][inChannels(128)]`.</summary>
    public static float[][] Forward(MiniMaxMusic3TransformerWeights w, float[][] latent, float[][] condition, float timestep)
    {
        int inChannels = MiniMaxMusic3Config.TransformerInChannels; // 128
        int condDim = MiniMaxMusic3Config.TransformerConditionDim; // 2048
        int concatChannels = 2 * inChannels + condDim; // 2304
        int inner = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim;
        int length = latent.Length;

        // Real: concat([latent, zeros, condition]) along channels, preprocess_conv(k=1) + residual.
        var concatRows = new float[length][];
        for (int t = 0; t < length; t++)
        {
            var row = new float[concatChannels];
            Array.Copy(latent[t], 0, row, 0, inChannels);
            // zeros already default-initialized in [inChannels, 2*inChannels)
            Array.Copy(condition[t], 0, row, 2 * inChannels, condDim);
            concatRows[t] = row;
        }
        var conv1x1 = Conv1x1NoBias(concatRows, length, concatChannels, concatChannels, w.PreprocessConvWeight);
        var preNet = new float[length][];
        for (int t = 0; t < length; t++)
        {
            preNet[t] = new float[concatChannels];
            for (int c = 0; c < concatChannels; c++) preNet[t][c] = conv1x1[t][c] + concatRows[t][c];
        }

        var temb = TimestepEmbed(w, timestep);

        // proj_in, then prepend the timestep token.
        var projIn = LinearNoBias(preNet, length, concatChannels, inner, w.ProjInWeight);
        int seqLen = length + 1;
        var x = new float[seqLen][];
        x[0] = temb;
        for (int t = 0; t < length; t++) x[t + 1] = projIn[t];

        var (cos, sin) = BuildPartialRope(seqLen);
        for (int li = 0; li < w.Blocks.Length; li++)
            x = Block(w.Blocks[li], x, seqLen, cos, sin);

        // Strip the timestep token, proj_out, postprocess_conv + residual.
        var stripped = new float[length][];
        for (int t = 0; t < length; t++) stripped[t] = x[t + 1];
        var projOut = LinearNoBias(stripped, length, inner, inChannels, w.ProjOutWeight);
        var postConv = Conv1x1NoBias(projOut, length, inChannels, inChannels, w.PostprocessConvWeight);

        var output = new float[length][];
        for (int t = 0; t < length; t++)
        {
            output[t] = new float[inChannels];
            for (int c = 0; c < inChannels; c++) output[t][c] = postConv[t][c] + projOut[t][c];
        }
        return output;
    }

    private static float[] TimestepEmbed(MiniMaxMusic3TransformerWeights w, float timestep)
    {
        int fourierDim = MiniMaxMusic3Config.TransformerFourierEmbeddingDim;
        int half = fourierDim / 2;
        int inner = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim;

        // Real MiniMaxMusic3FourierEmbedding: angles = 2*pi*t*weight; cat(cos, sin).
        var freq = new float[fourierDim];
        for (int i = 0; i < half; i++)
        {
            float angle = 2f * MathF.PI * timestep * w.TimeProjWeight[i];
            freq[i] = MathF.Cos(angle);
            freq[half + i] = MathF.Sin(angle);
        }

        var mid = LinearWithBias(freq, fourierDim, inner, w.TimeEmbed.Linear1Weight, w.TimeEmbed.Linear1Bias);
        for (int i = 0; i < mid.Length; i++) mid[i] = Silu(mid[i]);
        return LinearWithBias(mid, inner, inner, w.TimeEmbed.Linear2Weight, w.TimeEmbed.Linear2Bias);
    }

    private static float[][] Block(MiniMaxMusic3TransformerBlockWeights lw, float[][] x, int seqLen, float[] cos, float[] sin)
    {
        int inner = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim;

        var normed1 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { normed1[t] = new float[inner]; LayerNorm(x[t], lw.Norm1Weight, lw.Norm1Bias, normed1[t]); }

        var attnOut = SelfAttention(lw, normed1, seqLen, cos, sin);
        var afterAttn = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            afterAttn[t] = new float[inner];
            for (int i = 0; i < inner; i++) afterAttn[t][i] = x[t][i] + attnOut[t][i];
        }

        var normed2 = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { normed2[t] = new float[inner]; LayerNorm(afterAttn[t], lw.Norm2Weight, lw.Norm2Bias, normed2[t]); }

        var ffOut = FeedForward(lw, normed2, seqLen);
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            output[t] = new float[inner];
            for (int i = 0; i < inner; i++) output[t][i] = afterAttn[t][i] + ffOut[t][i];
        }
        return output;
    }

    private static float[][] SelfAttention(MiniMaxMusic3TransformerBlockWeights lw, float[][] normed, int seqLen, float[] cos, float[] sin)
    {
        int heads = MiniMaxMusic3Config.TransformerNumAttentionHeads;
        int headDim = MiniMaxMusic3Config.TransformerAttentionHeadDim;
        int inner = heads * headDim;

        var q = LinearNoBias(normed, seqLen, inner, inner, lw.QWeight);
        var k = LinearNoBias(normed, seqLen, inner, inner, lw.KWeight);
        var v = LinearNoBias(normed, seqLen, inner, inner, lw.VWeight);

        var qFlat = Flatten(q, seqLen, inner);
        var kFlat = Flatten(k, seqLen, inner);
        ApplyPartialRope(qFlat, seqLen, heads, headDim, cos, sin);
        ApplyPartialRope(kFlat, seqLen, heads, headDim, cos, sin);

        var vFlat = Flatten(v, seqLen, inner);
        float scale = 1f / MathF.Sqrt(headDim);
        var context = new float[seqLen * inner];
        Parallel.For(0, heads, h =>
        {
            int off = h * headDim;
            var scores = new float[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qFlat[i * inner + off + d] * kFlat[j * inner + off + d];
                    scores[j] = dot * scale;
                }
                SoftmaxRange(scores, 0, seqLen);

                var ctxSpan = context.AsSpan(i * inner + off, headDim);
                for (int j = 0; j < seqLen; j++)
                {
                    float s = scores[j];
                    var vSpan = vFlat.AsSpan(j * inner + off, headDim);
                    for (int d = 0; d < headDim; d++) ctxSpan[d] += s * vSpan[d];
                }
            }
        });

        var contextRows = Unflatten(context, seqLen, inner);
        return LinearNoBias(contextRows, seqLen, inner, inner, lw.OWeight);
    }

    private static float[][] FeedForward(MiniMaxMusic3TransformerBlockWeights lw, float[][] normed, int seqLen)
    {
        int inner = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim;
        int ffn = MiniMaxMusic3Config.TransformerFfInnerDim;

        var proj = LinearWithBias(normed, seqLen, inner, 2 * ffn, lw.FfInWeight, lw.FfInBias);
        var gated = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var row = new float[ffn];
            var gateStates = proj[t].AsSpan(0, ffn);
            var gate = proj[t].AsSpan(ffn, ffn);
            for (int i = 0; i < ffn; i++) row[i] = gateStates[i] * Silu(gate[i]);
            gated[t] = row;
        }

        return LinearWithBias(gated, seqLen, ffn, inner, lw.FfOutWeight, lw.FfOutBias);
    }

    // ── Shared small helpers (real Conv1d k=1 no-bias, Linear no-bias/with-bias, LayerNorm, partial RoPE) ──

    private static float[][] Conv1x1NoBias(float[][] x, int seqLen, int inCh, int outCh, float[] weight)
    {
        // Real Conv1d(kernel=1): output[t,oc] = sum_ic weight[oc,ic,0] * x[t,ic] (no bias).
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var row = new float[outCh];
            for (int oc = 0; oc < outCh; oc++)
            {
                float acc = 0f;
                int wBase = oc * inCh;
                for (int ic = 0; ic < inCh; ic++) acc += weight[wBase + ic] * x[t][ic];
                row[oc] = acc;
            }
            output[t] = row;
        }
        return output;
    }

    private static float[][] LinearNoBias(float[][] x, int seqLen, int inDim, int outDim, float[] weight)
    {
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var row = new float[outDim];
            for (int oc = 0; oc < outDim; oc++)
            {
                float acc = 0f;
                int wBase = oc * inDim;
                for (int ic = 0; ic < inDim; ic++) acc += weight[wBase + ic] * x[t][ic];
                row[oc] = acc;
            }
            output[t] = row;
        }
        return output;
    }

    private static float[] LinearWithBias(float[] x, int inDim, int outDim, float[] weight, float[] bias)
    {
        var output = new float[outDim];
        for (int oc = 0; oc < outDim; oc++)
        {
            float acc = bias[oc];
            int wBase = oc * inDim;
            for (int ic = 0; ic < inDim; ic++) acc += weight[wBase + ic] * x[ic];
            output[oc] = acc;
        }
        return output;
    }

    private static float[][] LinearWithBias(float[][] x, int seqLen, int inDim, int outDim, float[] weight, float[] bias)
    {
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) output[t] = LinearWithBias(x[t], inDim, outDim, weight, bias);
        return output;
    }

    private static void LayerNorm(float[] x, float[] weight, float[] bias, float[] output)
    {
        int n = x.Length;
        float mean = 0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        float varSum = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; varSum += d * d; }
        float invStd = 1f / MathF.Sqrt(varSum / n + 1e-5f);
        for (int i = 0; i < n; i++) output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
    }

    private static float[] Flatten(float[][] rows, int seqLen, int dim)
    {
        var flat = new float[seqLen * dim];
        for (int t = 0; t < seqLen; t++) Array.Copy(rows[t], 0, flat, t * dim, dim);
        return flat;
    }

    private static float[][] Unflatten(float[] flat, int seqLen, int dim)
    {
        var rows = new float[seqLen][];
        for (int t = 0; t < seqLen; t++) { rows[t] = new float[dim]; Array.Copy(flat, t * dim, rows[t], 0, dim); }
        return rows;
    }

    /// <summary>Real `MiniMaxMusic3RotaryEmbedding`: standard GPT-J-style partial rotary
    /// (`rotary_dim=32` of `headDim=64`), theta=10000, `freqs = cat([freqs,freqs])` (full-width
    /// cos/sin table duplicated across both halves of `rotary_dim`, standard HF convention).</summary>
    private static (float[] cos, float[] sin) BuildPartialRope(int seqLen)
    {
        int rotaryDim = MiniMaxMusic3Config.TransformerRotaryDim; // 32
        int half = rotaryDim / 2; // 16
        float theta = 10000f;
        var cos = new float[seqLen * rotaryDim];
        var sin = new float[seqLen * rotaryDim];
        for (int s = 0; s < seqLen; s++)
        {
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Pow(theta, -2f * i / rotaryDim);
                float angle = s * invFreq;
                float c = MathF.Cos(angle), sn = MathF.Sin(angle);
                cos[s * rotaryDim + i] = c; cos[s * rotaryDim + half + i] = c;
                sin[s * rotaryDim + i] = sn; sin[s * rotaryDim + half + i] = sn;
            }
        }
        return (cos, sin);
    }

    /// <summary>Real `_apply_partial_rotary_emb` (rotate_half convention over just the leading `rotaryDim` channels; the remaining `headDim-rotaryDim` channels pass through untouched).</summary>
    private static void ApplyPartialRope(float[] qOrK, int seqLen, int numHeads, int headDim, float[] cos, float[] sin)
    {
        int rotaryDim = MiniMaxMusic3Config.TransformerRotaryDim;
        int half = rotaryDim / 2;
        int rowDim = numHeads * headDim;
        for (int t = 0; t < seqLen; t++)
        {
            int cosBase = t * rotaryDim;
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
                // channels [rotaryDim, headDim) left untouched -- real partial-rotary behavior.
            }
        }
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
