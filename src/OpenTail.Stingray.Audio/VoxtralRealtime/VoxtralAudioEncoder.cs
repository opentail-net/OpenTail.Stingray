
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real Voxtral Realtime audio-tower forward pass, transcribed directly from
/// `examples/audio.cpp/src/models/voxtral_realtime/audio_encoder.cpp`'s
/// `build_audio_attention`/`build_audio_mlp`/`AudioGraph` (non-streaming/offline path, not the
/// KV-cache streaming path -- this port processes a full clip in one shot) -- see
/// `docs/audio-review-progress.md`'s Voxtral section for the architecture derivation. NOT a
/// Whisper-shaped encoder: a real Llama/Mistral-style transformer (RoPE-NEOX + full MHA + SwiGLU +
/// RMSNorm) applied to conv-downsampled mel features, with a local causal sliding-window mask.
/// </summary>
public static class VoxtralAudioEncoder
{
    /// <summary>Returns [tokens][TextHiddenSize] real audio embeddings ready to splice into the
    /// text decoder's token stream, matching the reference's
    /// `VoxtralRealtimeAudioEmbeddings`.</summary>
    public static float[][] Forward(VoxtralAudioEncoderWeights w, float[] melChannelMajor, int melFrames)
    {
        int melBins = VoxtralAudioEncoderWeights.NumMelBins;
        int hidden = VoxtralAudioEncoderWeights.HiddenSize;

        // Conv1d(melBins->hidden, k=3, s=1, pad=1) + GELU(erf).
        var stage1 = Conv1dSamePad(melChannelMajor, melBins, melFrames, w.Conv1Weight, w.Conv1Bias, hidden, kernel: 3, stride: 1);
        GeluErfInPlace(stage1);
        // Conv1d(hidden->hidden, k=3, s=2, pad=1) + GELU(erf) -- 2x downsample.
        int steps = (melFrames + 2 * 1 - 3) / 2 + 1;
        var stage2 = Conv1dSamePad(stage1, hidden, melFrames, w.Conv2Weight, w.Conv2Bias, hidden, kernel: 3, stride: 2);
        GeluErfInPlace(stage2);

        // Transpose channel-major [hidden, steps] -> frame-major [steps][hidden].
        var x = new float[steps][];
        Parallel.For(0, steps, t =>
        {
            var row = new float[hidden];
            for (int c = 0; c < hidden; c++) row[c] = stage2[c * steps + t];
            x[t] = row;
        });

        for (int layer = 0; layer < VoxtralAudioEncoderWeights.NumLayers; layer++)
        {
            var lw = w.Layers[layer];
            var attnIn = RmsNormRows(x, hidden, lw.AttnNorm);
            var attn = SelfAttention(attnIn, hidden, lw);
            AddRows(x, attn);
            var mlpIn = RmsNormRows(x, hidden, lw.FinalNorm);
            var mlp = Mlp(mlpIn, hidden, lw);
            AddRows(x, mlp);
        }

        x = RmsNormRows(x, hidden, w.NormWeight);

        // Downsample by DownsampleFactor: reshape 4 adjacent hidden-dim steps into one vector.
        int factor = VoxtralAudioEncoderWeights.DownsampleFactor;
        int tokens = steps / factor;
        int textHidden = VoxtralAudioEncoderWeights.TextHiddenSize;
        var downsampled = new float[tokens][];
        Parallel.For(0, tokens, t =>
        {
            var row = new float[hidden * factor];
            for (int f = 0; f < factor; f++)
                Array.Copy(x[t * factor + f], 0, row, f * hidden, hidden);
            downsampled[t] = row;
        });

        var output = new float[tokens][];
        Parallel.For(0, tokens, t =>
        {
            var proj1 = new float[textHidden];
            LinearNoBiasDirect(downsampled[t], w.Projector1Weight, proj1, hidden * factor, textHidden);
            GeluErfInPlace(proj1);
            var proj2 = new float[textHidden];
            LinearNoBiasDirect(proj1, w.Projector2Weight, proj2, textHidden, textHidden);
            output[t] = proj2;
        });
        return output;
    }

    private static float[] SelfAttention(float[][] xRows, int hidden, VoxtralAudioLayerWeights w)
    {
        int frames = xRows.Length;
        int heads = VoxtralAudioEncoderWeights.NumHeads;
        int headDim = VoxtralAudioEncoderWeights.HeadDim;
        int window = VoxtralAudioEncoderWeights.SlidingWindow;
        float invSqrtD = 1f / MathF.Sqrt(headDim);
        int qkvDim = heads * headDim;

        var q = new float[frames][];
        var k = new float[frames][];
        var v = new float[frames][];

        Parallel.For(0, frames, t =>
        {
            var qt = new float[qkvDim];
            var kt = new float[qkvDim];
            var vt = new float[qkvDim];
            LinearQKV(xRows[t], w.QWeight, w.QBias, w.KWeight, w.VWeight, w.VBias, qt, kt, vt, hidden, qkvDim);
            RopeNeoxInPlace(qt, heads, headDim, t);
            RopeNeoxInPlace(kt, heads, headDim, t);
            q[t] = qt;
            k[t] = kt;
            v[t] = vt;
        });

        var contextFlat = new float[frames][];
        for (int t = 0; t < frames; t++) contextFlat[t] = new float[heads * headDim];

        Parallel.For(0, heads, h =>
        {
            int chBase = h * headDim;
            var scores = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                int begin = Math.Max(0, i - window + 1);
                float maxScore = float.NegativeInfinity;
                for (int j = begin; j <= i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][chBase + d] * k[j][chBase + d];
                    dot *= invSqrtD;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = begin; j <= i; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = begin; j <= i; j++) acc += scores[j] * invSum * v[j][chBase + d];
                    contextFlat[i][chBase + d] = acc;
                }
            }
        });

        var flatOut = new float[frames * hidden];
        Parallel.For(0, frames, t =>
        {
            unsafe
            {
                fixed (float* pDst = &flatOut[t * hidden])
                    LinearDirect(contextFlat[t], w.OWeight, w.OBias, pDst, heads * headDim, hidden);
            }
        });
        return flatOut;
    }

    private static unsafe void LinearQKV(float[] input, byte[] wQ, float[] qBias, byte[] wK, byte[] wV, float[] vBias,
        float[] outQ, float[] outK, float[] outV, int inDim, int outDim)
    {
        int bytesPerRow = (inDim / 32) * 34;
        int scratchBytes = OpenTail.Stingray.Cpu.SimdKernels.Q8_0ScratchBytes(inDim);
        byte* scratch = stackalloc byte[scratchBytes];
        fixed (float* pIn = input)
            OpenTail.Stingray.Cpu.SimdKernels.QuantizeRowToQ8_0(pIn, inDim, scratch);

        fixed (byte* pQ = wQ, pK = wK, pV = wV)
        fixed (float* pQB = qBias, pVB = vBias)
        fixed (float* pq = outQ, pk = outK, pv = outV)
        {
            for (int i = 0; i < outDim; i++)
            {
                long offset = (long)i * bytesPerRow;
                pq[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pQ + offset, scratch, inDim) + (pQB != null ? pQB[i] : 0f);
                pk[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pK + offset, scratch, inDim);
                pv[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pV + offset, scratch, inDim) + (pVB != null ? pVB[i] : 0f);
            }
        }
    }

    private static unsafe void LinearNoBiasDirect(float[] input, byte[] weightQ8_0, float[] output, int inDim, int outDim)
    {
        fixed (float* pOut = output)
            LinearDirect(input, weightQ8_0, null, pOut, inDim, outDim);
    }

    private static unsafe void LinearDirect(float[] input, byte[] weightQ8_0, float[]? bias, float* output, int inDim, int outDim)
    {
        int bytesPerRow = (inDim / 32) * 34;
        int scratchBytes = OpenTail.Stingray.Cpu.SimdKernels.Q8_0ScratchBytes(inDim);
        byte* scratch = stackalloc byte[scratchBytes];
        fixed (float* pIn = input)
            OpenTail.Stingray.Cpu.SimdKernels.QuantizeRowToQ8_0(pIn, inDim, scratch);

        fixed (byte* pW = weightQ8_0)
        fixed (float* pB = bias)
        {
            for (int i = 0; i < outDim; i++)
            {
                output[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pW + (long)i * bytesPerRow, scratch, inDim) +
                    (pB != null ? pB[i] : 0f);
            }
        }
    }

    /// <summary>Real GGML `GGML_ROPE_TYPE_NEOX` convention: rotates pairs `(i, i+headDim/2)`
    /// within each head (half-split, NOT the interleaved GPT-J `(2i,2i+1)` pairing) -- applied
    /// across the full head_dim (no partial-rotation `n_dims` reduction here, unlike CosyVoice3's
    /// RoPE usage elsewhere in this codebase).</summary>
    private static void RopeNeoxInPlace(float[] qkv, int heads, int headDim, int position)
    {
        int half = headDim / 2;
        for (int h = 0; h < heads; h++)
        {
            int baseIdx = h * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Pow(VoxtralAudioEncoderWeights.RopeTheta, -2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                float a = qkv[baseIdx + i];
                float b = qkv[baseIdx + half + i];
                qkv[baseIdx + i] = a * cos - b * sin;
                qkv[baseIdx + half + i] = a * sin + b * cos;
            }
        }
    }

    private static float[] Mlp(float[][] xRows, int hidden, VoxtralAudioLayerWeights w)
    {
        int frames = xRows.Length;
        int inter = VoxtralAudioEncoderWeights.IntermediateSize;
        var flatOut = new float[frames * hidden];

        Parallel.For(0, frames, t =>
        {
            unsafe
            {
                int inScratchBytes = OpenTail.Stingray.Cpu.SimdKernels.Q8_0ScratchBytes(hidden);
                byte* inScratch = stackalloc byte[inScratchBytes];
                fixed (float* pIn = xRows[t])
                    OpenTail.Stingray.Cpu.SimdKernels.QuantizeRowToQ8_0(pIn, hidden, inScratch);

                float* gate = stackalloc float[inter];
                int inBytesPerRow = (hidden / 32) * 34;
                fixed (byte* pGW = w.GateWeight, pUW = w.UpWeight)
                {
                    for (int i = 0; i < inter; i++)
                    {
                        long off = (long)i * inBytesPerRow;
                        float g = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pGW + off, inScratch, hidden);
                        float u = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pUW + off, inScratch, hidden);
                        gate[i] = Silu(g) * u;
                    }
                }

                int interScratchBytes = OpenTail.Stingray.Cpu.SimdKernels.Q8_0ScratchBytes(inter);
                byte* interScratch = stackalloc byte[interScratchBytes];
                OpenTail.Stingray.Cpu.SimdKernels.QuantizeRowToQ8_0(gate, inter, interScratch);

                int downBytesPerRow = (inter / 32) * 34;
                fixed (byte* pDW = w.DownWeight)
                fixed (float* pDB = w.DownBias)
                fixed (float* pOut = &flatOut[t * hidden])
                {
                    for (int i = 0; i < hidden; i++)
                    {
                        pOut[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pDW + (long)i * downBytesPerRow, interScratch, inter) +
                            (pDB != null ? pDB[i] : 0f);
                    }
                }
            }
        });
        return flatOut;
    }

    private static void AddRows(float[][] x, float[] flatDelta)
    {
        int hidden = x[0].Length;
        Parallel.For(0, x.Length, t =>
        {
            int baseIdx = t * hidden;
            var row = x[t];
            for (int c = 0; c < hidden; c++)
                row[c] += flatDelta[baseIdx + c];
        });
    }

    private static float[][] RmsNormRows(float[][] xRows, int hidden, float[] weight)
    {
        var output = new float[xRows.Length][];
        Parallel.For(0, xRows.Length, t =>
        {
            var row = xRows[t];
            double sumSq = 0;
            for (int c = 0; c < hidden; c++) sumSq += (double)row[c] * row[c];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / hidden + VoxtralAudioEncoderWeights.RmsNormEps));
            var outRow = new float[hidden];
            for (int c = 0; c < hidden; c++) outRow[c] = row[c] * invRms * weight[c];
            output[t] = outRow;
        });
        return output;
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static void GeluErfInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + Erf(v * 0.70710678f));
        }
    }

    /// <summary>Abramowitz-Stegun erf approximation (same one used elsewhere in this codebase for
    /// exact-erf GELU, e.g. RvcHubertEncoder).</summary>
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    /// <summary>Real "same"-padding Conv1d over a [C][T] channel-major tensor with support for
    /// stride (used for the stem's stride-2 second conv). Weight real PyTorch layout
    /// [outCh, inCh, kernel].</summary>
    private static float[] Conv1dSamePad(float[] x, int inCh, int frames, float[] weight, float[] bias, int outCh, int kernel, int stride)
    {
        int pad = (kernel - 1) / 2;
        int outFrames = (frames + 2 * pad - kernel) / stride + 1;
        var output = new float[outCh * outFrames];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wBase = oc * inCh * kernel;
            for (int ot = 0; ot < outFrames; ot++)
            {
                float sum = b;
                int start = ot * stride - pad;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int kk = 0; kk < kernel; kk++)
                    {
                        int it = start + kk;
                        if ((uint)it >= (uint)frames) continue;
                        sum += weight[wIcBase + kk] * x[ic * frames + it];
                    }
                }
                output[oc * outFrames + ot] = sum;
            }
        });
        return output;
    }
}
