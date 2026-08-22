using System;
using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real SAN-M encoder forward pass for Paraformer, transcribed from the real `funasr` Python
/// package (`funasr/models/sanm/{attention,encoder,positionwise_feed_forward}.py`, see
/// docs/audio-review-progress.md's FunASR section for the full derivation -- do not re-derive
/// or guess any of this, especially the two non-obvious details below).
///
/// <para><b>Two critical, easy-to-miss details, both confirmed from real source, not
/// assumed</b>: (1) `encoders0.0` (560-dim input) has NO residual connection around its
/// self-attention, since `in_size (560) != size (512)` -- every other layer (512-&gt;512) DOES
/// get the residual. (2) The FSMN memory branch is added to standard self-attention's output
/// (`att_outs + fsmn_memory`) -- `examples/paraformer.cpp` has this exact add commented out/
/// disabled, confirmed a known-broken reference on this detail, do not port its encoder as-is.
/// </para>
/// </summary>
public static class FunAsrEncoder
{
    /// <summary>Runs the full encoder: encoders0.0 (560-dim, no residual) -> 49x main encoders.N (512-dim, with residual) -> after_norm. Input is frame-major [T, 560] (already CMVN-normalized + mel-splice, see FunAsrMelExtractor). Returns [T, 512].</summary>
    public static float[][] Forward(FunAsrWeights w, float[][] input)
    {
        int t = input.Length;
        var x = EncoderLayer(input, w.Encoders0Layer, w.EncoderHeads, inSize: 560, size: FunAsrWeights_HiddenDim(w));

        foreach (var layer in w.EncoderLayerWeights)
            x = EncoderLayer(x, layer, w.EncoderHeads, inSize: FunAsrWeights_HiddenDim(w), size: FunAsrWeights_HiddenDim(w));

        var output = new float[t][];
        for (int i = 0; i < t; i++)
            output[i] = LayerNorm(x[i], w.EncoderAfterNormWeight, w.EncoderAfterNormBias);
        return output;
    }

    private static int FunAsrWeights_HiddenDim(FunAsrWeights w) => w.EncoderDim;

    private static float[][] EncoderLayer(float[][] x, FunAsrEncoderLayerWeights lw, int heads, int inSize, int size)
    {
        int t = x.Length;
        var normed1 = new float[t][];
        for (int i = 0; i < t; i++) normed1[i] = LayerNorm(x[i], lw.Norm1Weight, lw.Norm1Bias);

        var attnOut = SelfAttentionSanm(normed1, lw, heads);

        var afterAttn = new float[t][];
        if (inSize == size)
        {
            for (int i = 0; i < t; i++)
            {
                var row = new float[size];
                for (int d = 0; d < size; d++) row[d] = x[i][d] + attnOut[i][d];
                afterAttn[i] = row;
            }
        }
        else
        {
            // encoders0.0 only: in_size (560) != size (512), so the residual is skipped entirely
            // (confirmed from EncoderLayerSANM.forward's `if self.in_size == self.size` branch).
            afterAttn = attnOut;
        }

        var normed2 = new float[t][];
        for (int i = 0; i < t; i++) normed2[i] = LayerNorm(afterAttn[i], lw.Norm2Weight, lw.Norm2Bias);

        var ffnOut = FfnPlain(normed2, lw);

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[size];
            for (int d = 0; d < size; d++) row[d] = afterAttn[i][d] + ffnOut[i][d];
            output[i] = row;
        }
        return output;
    }

    /// <summary>Real `MultiHeadedAttentionSANM.forward`: standard scaled-dot-product attention PLUS the FSMN memory branch (both summed) -- see class doc comment.</summary>
    private static float[][] SelfAttentionSanm(float[][] x, FunAsrEncoderLayerWeights lw, int heads)
    {
        int t = x.Length;
        int nFeat = lw.AttnOutBias.Length; // 512
        int dK = nFeat / heads;

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var qkv = Linear(x[i], lw.AttnQkvWeight, lw.AttnQkvBias, lw.InputDim, nFeat * 3);
            q[i] = qkv.AsSpan(0, nFeat).ToArray();
            k[i] = qkv.AsSpan(nFeat, nFeat).ToArray();
            v[i] = qkv.AsSpan(2 * nFeat, nFeat).ToArray();
        }

        var fsmnMemory = FsmnForward(v, lw.AttnFsmnWeight, kernel: 11);

        float scale = MathF.Pow(dK, -0.5f);
        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[nFeat];

        for (int h = 0; h < heads; h++)
        {
            int off = h * dK;
            for (int i = 0; i < t; i++)
            {
                var scores = new float[t];
                for (int j = 0; j < t; j++)
                    scores[j] = TensorPrimitives.Dot(q[i].AsSpan(off, dK), k[j].AsSpan(off, dK)) * scale;
                SoftmaxInPlace(scores);

                var ctxSpan = context[i].AsSpan(off, dK);
                for (int j = 0; j < t; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, dK), scores[j], ctxSpan, ctxSpan);
            }
        }

        var attOuts = new float[t][];
        for (int i = 0; i < t; i++)
            attOuts[i] = Linear(context[i], lw.AttnOutWeight, lw.AttnOutBias, nFeat, nFeat);

        var result = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[nFeat];
            for (int d = 0; d < nFeat; d++) row[d] = attOuts[i][d] + fsmnMemory[i][d];
            result[i] = row;
        }
        return result;
    }

    /// <summary>Real `forward_fsmn`: depthwise (per-channel) Conv1d over v, symmetric pad, residual add of v itself (NOT the layer input).</summary>
    private static float[][] FsmnForward(float[][] v, float[] fsmnWeight, int kernel)
    {
        int t = v.Length;
        int c = v[0].Length;
        int left = (kernel - 1) / 2;
        int right = kernel - 1 - left;

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[c];
            for (int ch = 0; ch < c; ch++)
            {
                float sum = 0f;
                int wBase = ch * kernel;
                for (int kk = 0; kk < kernel; kk++)
                {
                    int srcT = ti - left + kk;
                    if ((uint)srcT < (uint)t) sum += v[srcT][ch] * fsmnWeight[wBase + kk];
                }
                row[ch] = sum + v[ti][ch];
            }
            output[ti] = row;
        }
        return output;
    }

    /// <summary>Real plain `PositionwiseFeedForward`: w_2(ReLU(w_1(x))), no internal norm.</summary>
    private static float[][] FfnPlain(float[][] x, FunAsrEncoderLayerWeights lw)
    {
        int t = x.Length;
        int size = lw.FfnW2Bias.Length;
        int ffnDim = lw.FfnW1Bias.Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = Linear(x[i], lw.FfnW1Weight, lw.FfnW1Bias, size, ffnDim);
            for (int d = 0; d < ffnDim; d++) h[d] = MathF.Max(0f, h[d]);
            output[i] = Linear(h, lw.FfnW2Weight, lw.FfnW2Bias, ffnDim, size);
        }
        return output;
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input)
        {
            for (int o = 0; o < outDim; o++)
                output[o] = bias[o] + SimdKernels.DotF32(wp + (long)o * inDim, xp, inDim);
        }
        return output;
    }

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-12f)
    {
        int n = x.Length;
        float mean = TensorPrimitives.Sum((ReadOnlySpan<float>)x) / n;
        float variance = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; variance += d * d; }
        variance /= n;
        float invStd = 1f / MathF.Sqrt(variance + eps);

        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
        return output;
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }
}
