using System;
using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared math for the S3Gen-family CFM (Conditional Flow Matching) UNet estimator used by
/// both Chatterbox (`Chatterbox/ChatterboxCfmDecoder.cs`'s `ConditionalDecoder`, real, golden-
/// verified) and CosyVoice2 (`decoder.estimator.*` in `models/cosyvoice2_flow.safetensors`) --
/// confirmed architecturally identical UNet body by real tensor shapes (down: 320-&gt;256 +Nx
/// transformer, 12x mid 256-&gt;256 +Nx transformer, up: 512-&gt;256 [concat skip] +Nx transformer,
/// final 256-&gt;80) during this session's audit. The ONE real, confirmed difference between the
/// two checkpoints is the time-embedding computation: Chatterbox's is meanflow-distilled
/// (embeds t AND r, concatenates, mixes via an extra bias-free Linear) while CosyVoice2's
/// checkpoint has no such mixer tensor (`decoder.estimator.time_mlp.linear_1/2` only) --
/// standard single-timestep flow matching. Rather than bake either convention into the shared
/// kernel, <see cref="RunEstimator"/> takes an ALREADY-COMPUTED time-embedding vector and each
/// pipeline computes its own upstream, exactly the way the DiT-family kernels take an
/// already-embedded hidden state.
/// </summary>
public static class CfmUNetKernels
{
    /// <summary>
    /// input = concat([x, mu, spks_broadcast, cond], channels) -> down/mid/up UNet -> final
    /// block+proj. x/mu/cond are channel-first [melDim, t]; spkEmbed is [melDim]. Returns the
    /// predicted velocity, channel-first [melDim, t].
    /// </summary>
    public static float[] RunEstimator(
        IUnetStageWeights down, IUnetStageWeights[] mid, IUnetStageWeights up,
        float[] finalBlockConvWeight, float[] finalBlockConvBias, float[] finalBlockLnWeight, float[] finalBlockLnBias,
        float[] finalProjWeight, float[] finalProjBias,
        float[] x, float[] mu, float[] cond, float[] spkEmbed, float[] timeEmb,
        int t, int melDim, int channels, int heads, int headDim)
    {
        int inCh = melDim * 4;
        var input = new float[inCh * t];
        CopyChannels(x, input, 0, melDim, t);
        CopyChannels(mu, input, melDim, melDim, t);
        for (int c = 0; c < melDim; c++)
            for (int ti = 0; ti < t; ti++)
                input[(2 * melDim + c) * t + ti] = spkEmbed[c];
        CopyChannels(cond, input, 3 * melDim, melDim, t);

        var downOut = ResnetBlock(input, inCh, t, timeEmb, down.Resnet, channels);
        foreach (var tb in down.TransformerBlocks)
            downOut = TransformerBlock(downOut, channels, t, tb, heads, headDim);
        var skip = downOut;
        downOut = CausalConv1d(downOut, channels, t, down.ResampleConvWeight!, down.ResampleConvBias!, channels, kernel: 3);

        var midOut = downOut;
        foreach (var stage in mid)
        {
            midOut = ResnetBlock(midOut, channels, t, timeEmb, stage.Resnet, channels);
            foreach (var tb in stage.TransformerBlocks)
                midOut = TransformerBlock(midOut, channels, t, tb, heads, headDim);
        }

        var upIn = new float[2 * channels * t];
        CopyChannels(midOut, upIn, 0, channels, t);
        CopyChannels(skip, upIn, channels, channels, t);
        var upOut = ResnetBlock(upIn, 2 * channels, t, timeEmb, up.Resnet, channels);
        foreach (var tb in up.TransformerBlocks)
            upOut = TransformerBlock(upOut, channels, t, tb, heads, headDim);
        upOut = CausalConv1d(upOut, channels, t, up.ResampleConvWeight!, up.ResampleConvBias!, channels, kernel: 3);

        var finalConv = CausalConv1d(upOut, channels, t, finalBlockConvWeight, finalBlockConvBias, channels, kernel: 3);
        var finalNormed = LayerNormChannelFirst(finalConv, channels, t, finalBlockLnWeight, finalBlockLnBias);
        MishInPlace(finalNormed);

        return Conv1dK1(finalNormed, channels, t, finalProjWeight, finalProjBias, melDim);
    }

    /// <summary>SinusoidalPosEmb(dim), matcha/decoder.py convention: scale=1000.</summary>
    public static float[] SinusoidalPosEmb(float x, int dim)
    {
        int half = dim / 2;
        var emb = new float[dim];
        double logStep = Math.Log(10000.0) / (half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-i * logStep);
            double angle = 1000.0 * x * freq;
            emb[i] = (float)Math.Sin(angle);
            emb[half + i] = (float)Math.Cos(angle);
        }
        return emb;
    }

    private static float[] ResnetBlock(float[] x, int dimIn, int t, float[] timeEmb, IResnetBlockWeights rw, int dimOut)
    {
        var h = CausalConv1d(x, dimIn, t, rw.Block1ConvWeight, rw.Block1ConvBias, dimOut, kernel: 3);
        h = LayerNormChannelFirst(h, dimOut, t, rw.Block1LnWeight, rw.Block1LnBias);
        MishInPlace(h);

        var mishTime = new float[timeEmb.Length];
        for (int i = 0; i < mishTime.Length; i++) mishTime[i] = Mish(timeEmb[i]);
        var timeProj = Linear(mishTime, rw.MlpWeight, rw.MlpBias);
        for (int c = 0; c < dimOut; c++)
        {
            float bias = timeProj[c];
            int rowBase = c * t;
            for (int ti = 0; ti < t; ti++) h[rowBase + ti] += bias;
        }

        h = CausalConv1d(h, dimOut, t, rw.Block2ConvWeight, rw.Block2ConvBias, dimOut, kernel: 3);
        h = LayerNormChannelFirst(h, dimOut, t, rw.Block2LnWeight, rw.Block2LnBias);
        MishInPlace(h);

        var resConv = Conv1dK1(x, dimIn, t, rw.ResConvWeight, rw.ResConvBias, dimOut);
        for (int i = 0; i < h.Length; i++) h[i] += resConv[i];
        return h;
    }

    private static float[] TransformerBlock(float[] x, int dim, int t, IUnetTransformerBlockWeights tw, int heads, int headDim)
    {
        var normed = LayerNormChannelFirstToRowMajor(x, dim, t, tw.Norm1Weight, tw.Norm1Bias);
        var attnOut = SelfAttention(normed, t, dim, tw, heads, headDim);

        var afterAttn = new float[dim * t];
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
                afterAttn[c * t + ti] = x[c * t + ti] + attnOut[ti * dim + c];

        var normed3 = LayerNormChannelFirstToRowMajor(afterAttn, dim, t, tw.Norm3Weight, tw.Norm3Bias);
        var output = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[dim];
            Array.Copy(normed3, ti * dim, row, 0, dim);
            var up = Linear(row, tw.FfUpWeight, tw.FfUpBias);
            GeluInPlace(up);
            var down = Linear(up, tw.FfDownWeight, tw.FfDownBias);
            for (int c = 0; c < dim; c++) output[c * t + ti] = afterAttn[c * t + ti] + down[c];
        }
        return output;
    }

    private static float[] SelfAttention(float[] inputRowMajor, int t, int dim, IUnetTransformerBlockWeights tw, int heads, int headDim)
    {
        int qkvDim = heads * headDim;
        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            Array.Copy(inputRowMajor, i * dim, row, 0, dim);
            q[i] = tw.QWeight.MatVec(row);
            k[i] = tw.KWeight.MatVec(row);
            v[i] = tw.VWeight.MatVec(row);
        }

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[qkvDim];

        float scale = 1f / MathF.Sqrt(headDim);
        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                var scores = new float[t];
                unsafe
                {
                    fixed (float* qp = q[i])
                    {
                        for (int j = 0; j < t; j++)
                        {
                            fixed (float* kp = k[j])
                                scores[j] = SimdKernels.DotF32(qp + hOff, kp + hOff, headDim) * scale;
                        }
                    }
                }
                DenseKernels.SoftmaxInPlace(scores);
                var ctxSpan = context[i].AsSpan(hOff, headDim);
                for (int j = 0; j < t; j++)
                {
                    var vRow = v[j].AsSpan(hOff, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, scores[j], ctxSpan, ctxSpan);
                }
            });
        }

        var output = new float[t * dim];
        for (int i = 0; i < t; i++)
        {
            var projected = Linear(context[i], tw.OutWeight, tw.OutBias);
            Array.Copy(projected, 0, output, i * dim, dim);
        }
        return output;
    }

    private static void CopyChannels(float[] src, float[] dst, int dstChannelOffset, int channels, int t) =>
        Array.Copy(src, 0, dst, dstChannelOffset * t, channels * t);

    private static float[] CausalConv1d(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
    {
        int pad = kernel - 1;
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = new float[t];
            Array.Fill(outRow, bias[oc]);
            int wOcBase = oc * inCh * kernel;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * t, t);
                int wBase = wOcBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int shift = k - pad;
                    AxpyShifted(inRow, weight[wBase + k], outRow, shift, t);
                }
            }
            Array.Copy(outRow, 0, output, oc * t, t);
        });
        return output;
    }

    private static float[] Conv1dK1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = output.AsSpan(oc * t, t);
            outRow.Fill(bias[oc]);
            int wBase = oc * inCh;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * t, t);
                TensorPrimitives.MultiplyAdd(inRow, weight[wBase + ic], outRow, outRow);
            }
        });
        return output;
    }

    private static void AxpyShifted(ReadOnlySpan<float> input, float scale, Span<float> output, int shift, int t)
    {
        int start = Math.Max(0, -shift);
        int end = Math.Min(t, t - shift);
        int len = end - start;
        if (len <= 0) return;
        var inSlice = input.Slice(start + shift, len);
        var outSlice = output.Slice(start, len);
        TensorPrimitives.MultiplyAdd(inSlice, scale, outSlice, outSlice);
    }

    private static float[] LayerNormChannelFirst(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[ch * t];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < ch; c++) mean += x[c * t + ti];
            mean /= ch;
            double var = 0;
            for (int c = 0; c < ch; c++) { double d = x[c * t + ti] - mean; var += d * d; }
            var /= ch;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int c = 0; c < ch; c++)
                output[c * t + ti] = (float)((x[c * t + ti] - mean) * invStd) * weight[c] + bias[c];
        }
        return output;
    }

    private static float[] LayerNormChannelFirstToRowMajor(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[t * ch];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < ch; c++) mean += x[c * t + ti];
            mean /= ch;
            double var = 0;
            for (int c = 0; c < ch; c++) { double d = x[c * t + ti] - mean; var += d * d; }
            var /= ch;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int c = 0; c < ch; c++)
                output[ti * ch + c] = (float)((x[c * t + ti] - mean) * invStd) * weight[c] + bias[c];
        }
        return output;
    }

    private static float Mish(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));

    private static void MishInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = Mish(x[i]);
    }

    private static void GeluInPlace(float[] x)
    {
        const float c = 0.7978845608028654f;
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            float inner = c * (v + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
    }

    /// <summary>Linear layer via a <see cref="CfmLinearWeight"/> (real hardware F16C when available, F32 fallback otherwise) plus bias.</summary>
    private static float[] Linear(float[] input, CfmLinearWeight weight, float[] bias)
    {
        var output = weight.MatVec(input);
        for (int o = 0; o < output.Length; o++) output[o] += bias[o];
        return output;
    }
}

public interface IResnetBlockWeights
{
    float[] Block1ConvWeight { get; }
    float[] Block1ConvBias { get; }
    float[] Block1LnWeight { get; }
    float[] Block1LnBias { get; }
    CfmLinearWeight MlpWeight { get; }
    float[] MlpBias { get; }
    float[] Block2ConvWeight { get; }
    float[] Block2ConvBias { get; }
    float[] Block2LnWeight { get; }
    float[] Block2LnBias { get; }
    float[] ResConvWeight { get; }
    float[] ResConvBias { get; }
}

public interface IUnetTransformerBlockWeights
{
    float[] Norm1Weight { get; }
    float[] Norm1Bias { get; }
    CfmLinearWeight QWeight { get; }
    CfmLinearWeight KWeight { get; }
    CfmLinearWeight VWeight { get; }
    CfmLinearWeight OutWeight { get; }
    float[] OutBias { get; }
    float[] Norm3Weight { get; }
    float[] Norm3Bias { get; }
    CfmLinearWeight FfUpWeight { get; }
    float[] FfUpBias { get; }
    CfmLinearWeight FfDownWeight { get; }
    float[] FfDownBias { get; }
}

/// <summary>One UNet stage: a resnet block + N transformer blocks + (for down/up stages only) a resample conv.</summary>
public interface IUnetStageWeights
{
    IResnetBlockWeights Resnet { get; }
    IUnetTransformerBlockWeights[] TransformerBlocks { get; }
    float[]? ResampleConvWeight { get; }
    float[]? ResampleConvBias { get; }
}
