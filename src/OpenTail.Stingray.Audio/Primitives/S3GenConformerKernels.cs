using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared math for the S3Gen-family "token -> mu conditioning" flow encoder used by both
/// Chatterbox (`Chatterbox/ChatterboxFlowEncoder.cs`, real, golden-verified against PyTorch)
/// and CosyVoice2 (`CosyVoice/CosyVoiceFlowEncoder.cs`) -- confirmed to be architecturally
/// identical by real tensor shapes during this session's CosyVoice audit (same lineage: S3Gen
/// was derived from CosyVoice's own flow encoder). Extracted after both pipelines' encoders
/// were independently ported with near-duplicate logic (`PreLookahead`, `Upsample1D`,
/// `ConformerLayer`, `RelPositionSelfAttention`, `Conv1dValid`, `EmbedAndScale`) -- keep this
/// the single source of truth for any future S3Gen-lineage flow encoder, following the same
/// pattern already established for `VitsAttentionKernels`/`DenseKernels`.
///
/// Extraction was deliberately deferred until BOTH pipelines had independently-verified
/// implementations to check against each other (Chatterbox's golden test, CosyVoice2's real-
/// weights structural test) rather than attempted while either was still unverified -- see
/// docs/audio-review-progress.md's CosyVoice section for why.
/// </summary>
public static class S3GenConformerKernels
{
    /// <summary>tokenEmb is per-position input-embedding rows, already looked up by the caller (different pipelines source their embedding table differently -- Chatterbox concatenates prompt+generated S3 tokens, CosyVoice concatenates prompt+LLM speech tokens, but the lookup itself isn't part of the shared math).</summary>
    public static (float[] Mu, int TotalFrames) Forward(IS3GenFlowEncoderWeights w, float[][] tokenEmb)
    {
        int dim = w.HiddenDim;
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

    public static float[] EmbedRow(float[] table, int index, int dim)
    {
        var row = new float[dim];
        Array.Copy(table, (long)index * dim, row, 0, dim);
        return row;
    }

    /// <summary>spk_embed_affine_layer: F.normalize(embedding, dim=1) then Linear(spkEmbedDim, melChannels).</summary>
    public static float[] ProjectSpeakerEmbedding(float[] spkEmbedAffineWeight, float[] spkEmbedAffineBias, int spkEmbedDim, int melChannels, float[] xvector)
    {
        float normSq = 0f;
        for (int i = 0; i < xvector.Length; i++) normSq += xvector[i] * xvector[i];
        float invNorm = normSq > 1e-12f ? 1f / MathF.Sqrt(normSq) : 0f;
        var normalized = new float[xvector.Length];
        for (int i = 0; i < xvector.Length; i++) normalized[i] = xvector[i] * invNorm;
        return DenseKernels.Linear(normalized, spkEmbedAffineWeight, spkEmbedAffineBias, spkEmbedDim, melChannels);
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

    private static float[][] ConformerLayer(float[][] x, float[][] posEmb, IS3GenConformerLayerWeights lw, int heads, int headDim, int ffnDim)
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

    /// <summary>RelPositionMultiHeadedAttention, specialized for pos_emb length == key length (the case this encoder always hits, in both pipelines), so `rel_shift` is skipped.</summary>
    private static float[][] RelPositionSelfAttention(float[][] x, float[][] posEmb, IS3GenConformerLayerWeights lw, int heads, int headDim)
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

/// <summary>One S3Gen-family Conformer block's weights: rel-pos self-attention (untied u/v biases) + Swish FFN, no macaron, no conv module.</summary>
public interface IS3GenConformerLayerWeights
{
    float[] NormMhaWeight { get; }
    float[] NormMhaBias { get; }
    float[] QWeight { get; }
    float[] QBias { get; }
    float[] KWeight { get; }
    float[] KBias { get; }
    float[] VWeight { get; }
    float[] VBias { get; }
    float[] OutWeight { get; }
    float[] OutBias { get; }
    float[] PosWeight { get; }
    float[] PosBiasU { get; }
    float[] PosBiasV { get; }
    float[] NormFfWeight { get; }
    float[] NormFfBias { get; }
    float[] Ff1Weight { get; }
    float[] Ff1Bias { get; }
    float[] Ff2Weight { get; }
    float[] Ff2Bias { get; }
}

/// <summary>Top-level weights an S3Gen-family flow encoder needs, independent of which format (GGUF, safetensors, ...) they were loaded from.</summary>
public interface IS3GenFlowEncoderWeights
{
    int HiddenDim { get; }
    int NumHeads { get; }
    int HeadDim { get; }
    int FfnDim { get; }
    int MelChannels { get; }
    float[] EmbedLinearWeight { get; }
    float[] EmbedLinearBias { get; }
    float[] EmbedLnWeight { get; }
    float[] EmbedLnBias { get; }
    float[] PlaConv1Weight { get; }
    float[] PlaConv1Bias { get; }
    float[] PlaConv2Weight { get; }
    float[] PlaConv2Bias { get; }
    float[] UlConvWeight { get; }
    float[] UlConvBias { get; }
    float[] UpEmbedLinearWeight { get; }
    float[] UpEmbedLinearBias { get; }
    float[] UpEmbedLnWeight { get; }
    float[] UpEmbedLnBias { get; }
    float[] AfterNormWeight { get; }
    float[] AfterNormBias { get; }
    float[] EncoderProjWeight { get; }
    float[] EncoderProjBias { get; }
    IS3GenConformerLayerWeights[] EncLayers { get; }
    IS3GenConformerLayerWeights[] UpEncLayers { get; }
}
