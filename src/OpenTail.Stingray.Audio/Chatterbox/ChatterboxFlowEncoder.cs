using System;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen stage 1: UpsampleConformerEncoder (examples/chatterbox-tts-py/chatterbox/models/s3gen/
/// transformer/upsample_encoder.py), ported for the non-streaming/non-chunked, no-padding-mask
/// case this codebase needs (single sequence, batch size 1, offset=0). Turns concatenated
/// [prompt_token; speech_tokens] S3 speech-token ids into the CFM decoder's `mu` conditioning
/// tensor (channel-first [MelChannels, 2*T]) and projects the reference x-vector into the CFM's
/// speaker-embedding space.
///
/// Pipeline (s3gen.py's UpsampleConformerEncoder config: macaron_style=False, use_cnn_module=
/// False, so each ConformerEncoderLayer is just rel-pos self-attention + swish FFN, no macaron
/// FFN and no depthwise conv branch -- see encoder_layer.py):
///   tokenEmb = input_embedding[token]                              [T, 512]
///   x = LayerNorm(Linear(tokenEmb)); x *= sqrt(512); pos_emb = sinusoid[0:T]   (LinearNoSubsampling + RelPositionalEncoding)
///   x = PreLookaheadLayer(x)                                        (conv1 k=4 rightpad3 + leakyrelu, conv2 k=3 leftpad2, + residual)
///   x = 6x ConformerEncoderLayer(x, pos_emb)                        (enc.0..5)
///   x = Upsample1D(x)                                               (nearest x2, leftpad4, conv k=5)  -> [2T, 512]
///   x = LayerNorm(Linear(x)); x *= sqrt(512); pos_emb2 = sinusoid[0:2T]        (up_embed)
///   x = 4x ConformerEncoderLayer(x, pos_emb2)                       (ue.0..3)
///   x = LayerNorm(x)                                                (after_norm)
///   mu = Linear(x) -> [2T, 80], transposed to channel-first [80, 2T]
/// </summary>
public static class ChatterboxFlowEncoder
{
    public static (float[] Mu, int TotalFrames) Forward(
        ChatterboxS3GenWeights w, int[] promptTokens, int[] speechTokens)
    {
        int t = promptTokens.Length + speechTokens.Length;
        int dim = w.EncHidden;

        var tokenEmb = new float[t][];
        for (int i = 0; i < promptTokens.Length; i++)
            tokenEmb[i] = EmbedRow(w.InputEmbeddingWeight, promptTokens[i], dim);
        for (int i = 0; i < speechTokens.Length; i++)
            tokenEmb[promptTokens.Length + i] = EmbedRow(w.InputEmbeddingWeight, speechTokens[i], dim);

        var (x, posEmb) = EmbedAndScale(tokenEmb, w.EmbedLinearWeight, w.EmbedLinearBias, w.EmbedLnWeight, w.EmbedLnBias, dim);

        x = PreLookahead(x, w.PlaConv1Weight, w.PlaConv1Bias, w.PlaConv2Weight, w.PlaConv2Bias, dim);

        foreach (var layer in w.EncLayers)
            x = ConformerLayer(x, posEmb, layer, w.EncHeads, w.EncHeadDim, w.EncFfn);

        var upsampled = Upsample1D(x, w.UlConvWeight, w.UlConvBias, dim);
        int t2 = upsampled.Length;

        var (x2, posEmb2) = EmbedAndScale(upsampled, w.UpEmbedLinearWeight, w.UpEmbedLinearBias, w.UpEmbedLnWeight, w.UpEmbedLnBias, dim);

        foreach (var layer in w.UpEncLayers)
            x2 = ConformerLayer(x2, posEmb2, layer, w.EncHeads, w.EncHeadDim, w.EncFfn);

        for (int i = 0; i < t2; i++)
            x2[i] = LayerNorm(x2[i], w.AfterNormWeight, w.AfterNormBias);

        // encoder_proj: [T2, 512] -> [T2, 80], then transpose to channel-first [80, T2]
        var muRowMajor = new float[t2][];
        for (int i = 0; i < t2; i++)
            muRowMajor[i] = Linear(x2[i], w.EncoderProjWeight, w.EncoderProjBias, dim, w.MelChannels);

        var mu = new float[w.MelChannels * t2];
        for (int c = 0; c < w.MelChannels; c++)
            for (int i = 0; i < t2; i++)
                mu[c * t2 + i] = muRowMajor[i][c];

        return (mu, t2);
    }

    /// <summary>
    /// spk_embed_affine_layer: F.normalize(embedding, dim=1) then Linear(SpkEncDim, MelChannels).
    /// </summary>
    public static float[] ProjectSpeakerEmbedding(ChatterboxS3GenWeights w, float[] xvector)
    {
        float normSq = 0f;
        for (int i = 0; i < xvector.Length; i++) normSq += xvector[i] * xvector[i];
        float invNorm = normSq > 1e-12f ? 1f / MathF.Sqrt(normSq) : 0f;
        var normalized = new float[xvector.Length];
        for (int i = 0; i < xvector.Length; i++) normalized[i] = xvector[i] * invNorm;
        return Linear(normalized, w.SpkEmbedAffineWeight, w.SpkEmbedAffineBias, w.SpkEncDim, w.MelChannels);
    }

    private static float[] EmbedRow(float[] table, int index, int dim)
    {
        var row = new float[dim];
        Array.Copy(table, (long)index * dim, row, 0, dim);
        return row;
    }

    /// <summary>LinearNoSubsampling + RelPositionalEncoding.forward, offset=0.</summary>
    private static (float[][] X, float[][] PosEmb) EmbedAndScale(float[][] input, float[] linW, float[] linB, float[] lnW, float[] lnB, int dim)
    {
        int t = input.Length;
        float xscale = MathF.Sqrt(dim);
        var x = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = Linear(input[i], linW, linB, dim, dim);
            h = LayerNorm(h, lnW, lnB);
            for (int d = 0; d < dim; d++) h[d] *= xscale;
            x[i] = h;
        }

        var posEmb = new float[t][];
        for (int pos = 0; pos < t; pos++)
            posEmb[pos] = SinusoidalPositionEncoding(pos, dim);

        return (x, posEmb);
    }

    private static float[] SinusoidalPositionEncoding(int position, int dim)
    {
        var pe = new float[dim];
        for (int i = 0; i < dim; i += 2)
        {
            double divTerm = Math.Exp(i * (-Math.Log(10000.0) / dim));
            pe[i] = (float)Math.Sin(position * divTerm);
            if (i + 1 < dim) pe[i + 1] = (float)Math.Cos(position * divTerm);
        }
        return pe;
    }

    /// <summary>
    /// PreLookaheadLayer: right-pad 3 -> conv1(k=4) -> leaky_relu(0.01) -> left-pad 2 -> conv2(k=3)
    /// -> residual add with the layer's original input. Conv weight layout [outCh, inCh, kernel]
    /// (torch nn.Conv1d convention).
    /// </summary>
    private static float[][] PreLookahead(float[][] x, float[] conv1W, float[] conv1B, float[] conv2W, float[] conv2B, int dim)
    {
        int t = x.Length;

        // channel-first [dim, t+3] with 3 zero-padded columns on the right
        var padded1 = new float[dim * (t + 3)];
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
                padded1[c * (t + 3) + ti] = x[ti][c];

        var afterConv1 = Conv1dValid(padded1, dim, t + 3, conv1W, conv1B, dim, kernel: 4); // -> length t
        for (int i = 0; i < afterConv1.Length; i++)
            if (afterConv1[i] < 0f) afterConv1[i] *= 0.01f; // leaky_relu

        // left-pad 2
        var padded2 = new float[dim * (t + 2)];
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
                padded2[c * (t + 2) + (ti + 2)] = afterConv1[c * t + ti];

        var afterConv2 = Conv1dValid(padded2, dim, t + 2, conv2W, conv2B, dim, kernel: 3); // -> length t

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[dim];
            for (int c = 0; c < dim; c++) row[c] = afterConv2[c * t + ti] + x[ti][c];
            output[ti] = row;
        }
        return output;
    }

    /// <summary>
    /// Upsample1D: nearest x2 upsample along time, left-pad by 4 (stride*2), conv(k=5, stride=1,
    /// no padding).
    /// </summary>
    private static float[][] Upsample1D(float[][] x, float[] convW, float[] convB, int dim)
    {
        int t = x.Length;
        int t2 = t * 2;

        var upsampled = new float[dim * (t2 + 4)]; // channel-first, 4 zero columns on the left
        for (int c = 0; c < dim; c++)
        {
            for (int ti = 0; ti < t2; ti++)
            {
                int srcT = ti / 2; // nearest
                upsampled[c * (t2 + 4) + 4 + ti] = x[srcT][c];
            }
        }

        var afterConv = Conv1dValid(upsampled, dim, t2 + 4, convW, convB, dim, kernel: 5); // -> length t2

        var output = new float[t2][];
        for (int ti = 0; ti < t2; ti++)
        {
            var row = new float[dim];
            for (int c = 0; c < dim; c++) row[c] = afterConv[c * t2 + ti];
            output[ti] = row;
        }
        return output;
    }

    /// <summary>Valid (no padding) stride-1 Conv1d. input is channel-first [inCh, inT]; weight is
    /// torch layout [outCh, inCh, kernel]. Output is channel-first [outCh, inT-kernel+1].</summary>
    private static float[] Conv1dValid(float[] input, int inCh, int inT, float[] weight, float[] bias, int outCh, int kernel)
    {
        int outT = inT - kernel + 1;
        var output = new float[outCh * outT];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wOcBase = oc * inCh * kernel;
            for (int ot = 0; ot < outT; ot++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wBase = wOcBase + ic * kernel;
                    int srcBase = ic * inT + ot;
                    for (int k = 0; k < kernel; k++)
                        sum += weight[wBase + k] * input[srcBase + k];
                }
                output[oc * outT + ot] = sum;
            }
        }
        return output;
    }

    /// <summary>
    /// ConformerEncoderLayer (macaron_style=False, use_cnn_module=False): pre-LN rel-pos
    /// self-attention + residual, then pre-LN swish FFN + residual.
    /// </summary>
    private static float[][] ConformerLayer(float[][] x, float[][] posEmb, ChatterboxS3GenConformerLayer lw, int heads, int headDim, int ffnDim)
    {
        int t = x.Length;
        int dim = heads * headDim;

        var normed = new float[t][];
        for (int i = 0; i < t; i++) normed[i] = LayerNorm(x[i], lw.NormMhaWeight, lw.NormMhaBias);

        var attnOut = RelPositionSelfAttention(normed, posEmb, lw, heads, headDim);

        var afterAttn = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d];
            afterAttn[i] = row;
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var ffnNormed = LayerNorm(afterAttn[i], lw.NormFfWeight, lw.NormFfBias);
            var h1 = Linear(ffnNormed, lw.Ff1Weight, lw.Ff1Bias, dim, ffnDim);
            SwishInPlace(h1);
            var h2 = Linear(h1, lw.Ff2Weight, lw.Ff2Bias, ffnDim, dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterAttn[i][d] + h2[d];
            output[i] = row;
        }
        return output;
    }

    /// <summary>
    /// RelPositionMultiHeadedAttention.forward (attention.py), specialized for the case this
    /// encoder always hits: pos_emb length == key length, so matrix_ac and matrix_bd already have
    /// matching shapes and the Transformer-XL rel_shift trick is skipped (the reference itself
    /// only applies rel_shift "if matrix_ac.shape != matrix_bd.shape").
    /// </summary>
    private static float[][] RelPositionSelfAttention(float[][] x, float[][] posEmb, ChatterboxS3GenConformerLayer lw, int heads, int headDim)
    {
        int t = x.Length;
        int dim = heads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        var p = new float[t][];
        for (int i = 0; i < t; i++)
        {
            q[i] = Linear(x[i], lw.QWeight, lw.QBias, dim, dim);
            k[i] = Linear(x[i], lw.KWeight, lw.KBias, dim, dim);
            v[i] = Linear(x[i], lw.VWeight, lw.VBias, dim, dim);
            p[i] = LinearNoBias(posEmb[i], lw.PosWeight, dim, dim);
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++) output[i] = new float[dim];

        // Per head: precompute q_with_bias_u/v once per position (the original loop recomputed
        // them on every (i,j) pair -- O(t^2*headDim) redundant additions for a O(t*headDim)
        // quantity), then score/weighted-sum with SIMD dot products, parallelized across query
        // positions (each `i` only writes its own output[i] row, across all heads sequentially,
        // so this is race-free without needing to parallelize the outer head loop too).
        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            var qU = new float[t][];
            var qV = new float[t][];
            for (int i = 0; i < t; i++)
            {
                var u = new float[headDim];
                var vv = new float[headDim];
                for (int d = 0; d < headDim; d++)
                {
                    u[d] = q[i][hOff + d] + lw.PosBiasU[h * headDim + d];
                    vv[d] = q[i][hOff + d] + lw.PosBiasV[h * headDim + d];
                }
                qU[i] = u;
                qV[i] = vv;
            }

            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                var scores = new float[t];
                unsafe
                {
                    fixed (float* up = qU[i], vp = qV[i])
                    {
                        for (int j = 0; j < t; j++)
                        {
                            fixed (float* kp = k[j], pp = p[j])
                            {
                                float ac = SimdKernels.DotF32(up, kp + hOff, headDim);
                                float bd = SimdKernels.DotF32(vp, pp + hOff, headDim);
                                scores[j] = (ac + bd) * scale;
                            }
                        }
                    }
                }
                SoftmaxInPlace(scores);
                var outSpan = output[i].AsSpan(hOff, headDim);
                for (int j = 0; j < t; j++)
                {
                    var vRow = v[j].AsSpan(hOff, headDim);
                    System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(vRow, scores[j], outSpan, outSpan);
                }
            });
        }

        var projected = new float[t][];
        for (int i = 0; i < t; i++)
            projected[i] = Linear(output[i], lw.OutWeight, lw.OutBias, dim, dim);
        return projected;
    }

    private static void SwishInPlace(float[] x) => DenseKernels.SiluInPlace(x);

    private static void SoftmaxInPlace(float[] scores) => DenseKernels.SoftmaxInPlace(scores);

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim) =>
        DenseKernels.Linear(input, weight, bias, inDim, outDim);

    private static float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim) =>
        DenseKernels.LinearNoBias(input, weight, inDim, outDim);

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f) =>
        DenseKernels.LayerNorm(x, weight, bias, eps);
}
