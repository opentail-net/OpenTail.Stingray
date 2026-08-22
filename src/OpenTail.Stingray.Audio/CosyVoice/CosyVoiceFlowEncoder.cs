using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice2's flow "UpsampleConformerEncoder": turns concatenated [prompt_token;
/// speech_tokens] into the CFM decoder's `mu` conditioning tensor (channel-first
/// [MelChannels, 2*T]) and projects the reference x-vector into the CFM's speaker-embedding
/// space. Architecturally identical to `Chatterbox/ChatterboxFlowEncoder.cs`'s S3Gen encoder
/// (same lineage, real tensor names confirmed matching during the Phase-0 audit -- see
/// docs/audio-review-progress.md's CosyVoice section) -- this is a parallel port against
/// `CosyVoiceFlowWeights`'s HF-sourced tensors rather than a direct reuse of Chatterbox's
/// classes (different weight-loader/tensor-name source), but the block-level math is the same
/// derivation, copied and adapted rather than re-derived from scratch.
///
/// Pipeline (matches `ChatterboxFlowEncoder`'s, `macaron_style=False, use_cnn_module=False`):
///   tokenEmb = input_embedding[token]
///   x = LayerNorm(Linear(tokenEmb)); x *= sqrt(hidden); pos_emb = sinusoid[0:T]
///   x = PreLookaheadLayer(x)
///   x = 6x ConformerEncoderLayer(x, pos_emb)      (encoders.0..5)
///   x = Upsample1D(x)                             (nearest x2, leftpad4, conv k=5) -> [2T, hidden]
///   x = LayerNorm(Linear(x)); x *= sqrt(hidden); pos_emb2 = sinusoid[0:2T]
///   x = 4x ConformerEncoderLayer(x, pos_emb2)     (up_encoders.0..3)
///   x = LayerNorm(x)                              (after_norm)
///   mu = Linear(x) -> [2T, MelChannels], transposed to channel-first [MelChannels, 2T]
///
/// NOT YET golden-verified against a real oracle -- structurally complete, same caveat as
/// every pipeline in this doc before its golden-verification pass.
/// </summary>
public static class CosyVoiceFlowEncoder
{
    public static (float[] Mu, int TotalFrames) Forward(
        CosyVoiceFlowWeights w, int[] promptTokens, int[] speechTokens)
    {
        int t = promptTokens.Length + speechTokens.Length;
        int dim = w.HiddenDim;

        var tokenEmb = new float[t][];
        for (int i = 0; i < promptTokens.Length; i++)
            tokenEmb[i] = EmbedRow(w.InputEmbeddingWeight, promptTokens[i], dim);
        for (int i = 0; i < speechTokens.Length; i++)
            tokenEmb[promptTokens.Length + i] = EmbedRow(w.InputEmbeddingWeight, speechTokens[i], dim);

        var (x, posEmb) = EmbedAndScale(tokenEmb, w.EmbedLinearWeight, w.EmbedLinearBias, w.EmbedLnWeight, w.EmbedLnBias, dim);

        x = PreLookahead(x, w.PlaConv1Weight, w.PlaConv1Bias, w.PlaConv2Weight, w.PlaConv2Bias, dim);

        foreach (var layer in w.EncLayers)
            x = ConformerLayer(x, posEmb, layer, w.NumHeads, w.HeadDim, w.FfnDim);

        var upsampled = Upsample1D(x, w.UlConvWeight, w.UlConvBias, dim);
        int t2 = upsampled.Length;

        var (x2, posEmb2) = EmbedAndScale(upsampled, w.UpEmbedLinearWeight, w.UpEmbedLinearBias, w.UpEmbedLnWeight, w.UpEmbedLnBias, dim);

        foreach (var layer in w.UpEncLayers)
            x2 = ConformerLayer(x2, posEmb2, layer, w.NumHeads, w.HeadDim, w.FfnDim);

        for (int i = 0; i < t2; i++)
            x2[i] = DenseKernels.LayerNorm(x2[i], w.AfterNormWeight, w.AfterNormBias);

        var muRowMajor = new float[t2][];
        for (int i = 0; i < t2; i++)
            muRowMajor[i] = DenseKernels.Linear(x2[i], w.EncoderProjWeight, w.EncoderProjBias, dim, w.MelChannels);

        var mu = new float[w.MelChannels * t2];
        for (int c = 0; c < w.MelChannels; c++)
            for (int i = 0; i < t2; i++)
                mu[c * t2 + i] = muRowMajor[i][c];

        return (mu, t2);
    }

    /// <summary>spk_embed_affine_layer: F.normalize(embedding, dim=1) then Linear(SpkEmbedDim, MelChannels).</summary>
    public static float[] ProjectSpeakerEmbedding(CosyVoiceFlowWeights w, float[] xvector)
    {
        float normSq = 0f;
        for (int i = 0; i < xvector.Length; i++) normSq += xvector[i] * xvector[i];
        float invNorm = normSq > 1e-12f ? 1f / MathF.Sqrt(normSq) : 0f;
        var normalized = new float[xvector.Length];
        for (int i = 0; i < xvector.Length; i++) normalized[i] = xvector[i] * invNorm;
        return DenseKernels.Linear(normalized, w.SpkEmbedAffineWeight, w.SpkEmbedAffineBias, w.SpkEmbedDim, w.MelChannels);
    }

    private static float[] EmbedRow(float[] table, int index, int dim)
    {
        var row = new float[dim];
        Array.Copy(table, (long)index * dim, row, 0, dim);
        return row;
    }

    private static (float[][] X, float[][] PosEmb) EmbedAndScale(float[][] input, float[] linW, float[] linB, float[] lnW, float[] lnB, int dim)
    {
        int t = input.Length;
        float xscale = MathF.Sqrt(dim);
        var x = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = DenseKernels.Linear(input[i], linW, linB, dim, dim);
            h = DenseKernels.LayerNorm(h, lnW, lnB);
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

    private static float[][] PreLookahead(float[][] x, float[] conv1W, float[] conv1B, float[] conv2W, float[] conv2B, int dim)
    {
        int t = x.Length;

        var padded1 = new float[dim * (t + 3)];
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
                padded1[c * (t + 3) + ti] = x[ti][c];

        var afterConv1 = Conv1dValid(padded1, dim, t + 3, conv1W, conv1B, dim, kernel: 4);
        for (int i = 0; i < afterConv1.Length; i++)
            if (afterConv1[i] < 0f) afterConv1[i] *= 0.01f; // leaky_relu

        var padded2 = new float[dim * (t + 2)];
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
                padded2[c * (t + 2) + (ti + 2)] = afterConv1[c * t + ti];

        var afterConv2 = Conv1dValid(padded2, dim, t + 2, conv2W, conv2B, dim, kernel: 3);

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[dim];
            for (int c = 0; c < dim; c++) row[c] = afterConv2[c * t + ti] + x[ti][c];
            output[ti] = row;
        }
        return output;
    }

    private static float[][] Upsample1D(float[][] x, float[] convW, float[] convB, int dim)
    {
        int t = x.Length;
        int t2 = t * 2;

        var upsampled = new float[dim * (t2 + 4)];
        for (int c = 0; c < dim; c++)
        {
            for (int ti = 0; ti < t2; ti++)
            {
                int srcT = ti / 2;
                upsampled[c * (t2 + 4) + 4 + ti] = x[srcT][c];
            }
        }

        var afterConv = Conv1dValid(upsampled, dim, t2 + 4, convW, convB, dim, kernel: 5);

        var output = new float[t2][];
        for (int ti = 0; ti < t2; ti++)
        {
            var row = new float[dim];
            for (int c = 0; c < dim; c++) row[c] = afterConv[c * t2 + ti];
            output[ti] = row;
        }
        return output;
    }

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

    private static float[][] ConformerLayer(float[][] x, float[][] posEmb, CosyVoiceFlowLayerWeights lw, int heads, int headDim, int ffnDim)
    {
        int t = x.Length;
        int dim = heads * headDim;

        var normed = new float[t][];
        for (int i = 0; i < t; i++) normed[i] = DenseKernels.LayerNorm(x[i], lw.NormMhaWeight, lw.NormMhaBias);

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
            var ffnNormed = DenseKernels.LayerNorm(afterAttn[i], lw.NormFfWeight, lw.NormFfBias);
            var h1 = DenseKernels.Linear(ffnNormed, lw.Ff1Weight, lw.Ff1Bias, dim, ffnDim);
            DenseKernels.SiluInPlace(h1);
            var h2 = DenseKernels.Linear(h1, lw.Ff2Weight, lw.Ff2Bias, ffnDim, dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterAttn[i][d] + h2[d];
            output[i] = row;
        }
        return output;
    }

    /// <summary>RelPositionMultiHeadedAttention, specialized for pos_emb length == key length (the case this encoder always hits), so `rel_shift` is skipped -- same specialization Chatterbox's version uses, verified applicable here too since the reference module is the same.</summary>
    private static float[][] RelPositionSelfAttention(float[][] x, float[][] posEmb, CosyVoiceFlowLayerWeights lw, int heads, int headDim)
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
            q[i] = DenseKernels.Linear(x[i], lw.QWeight, lw.QBias, dim, dim);
            k[i] = DenseKernels.Linear(x[i], lw.KWeight, lw.KBias, dim, dim);
            v[i] = DenseKernels.Linear(x[i], lw.VWeight, lw.VBias, dim, dim);
            p[i] = DenseKernels.LinearNoBias(posEmb[i], lw.PosWeight, dim, dim);
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++) output[i] = new float[dim];

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
                DenseKernels.SoftmaxInPlace(scores);
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
            projected[i] = DenseKernels.Linear(output[i], lw.OutWeight, lw.OutBias, dim, dim);
        return projected;
    }
}
