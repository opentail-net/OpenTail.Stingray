using System;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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

        using var scratch = new UnetScratchBuffers(t, channels, 1024, heads * headDim);

        var downOut = ResnetBlock(input, inCh, t, timeEmb, down.Resnet, channels);
        int chT = channels * t;
        var tbBufA = ArrayPool<float>.Shared.Rent(chT);
        var tbBufB = ArrayPool<float>.Shared.Rent(chT);
        try
        {
            Array.Copy(downOut, tbBufA, chT);
            float[] curIn = tbBufA, curOut = tbBufB;
            foreach (var tb in down.TransformerBlocks)
            {
                TransformerBlock(curIn, curOut, channels, t, tb, heads, headDim, scratch);
                (curIn, curOut) = (curOut, curIn);
            }
            Array.Copy(curIn, downOut, chT);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tbBufA);
            ArrayPool<float>.Shared.Return(tbBufB);
        }
        var skip = downOut;
        downOut = CausalConv1d(downOut, channels, t, down.ResampleConvWeight!, down.ResampleConvBias!, channels, kernel: 3);

        var midOut = downOut;
        foreach (var stage in mid)
        {
            midOut = ResnetBlock(midOut, channels, t, timeEmb, stage.Resnet, channels);
            var midBufA = ArrayPool<float>.Shared.Rent(chT);
            var midBufB = ArrayPool<float>.Shared.Rent(chT);
            try
            {
                Array.Copy(midOut, midBufA, chT);
                float[] curIn = midBufA, curOut = midBufB;
                foreach (var tb in stage.TransformerBlocks)
                {
                    TransformerBlock(curIn, curOut, channels, t, tb, heads, headDim, scratch);
                    (curIn, curOut) = (curOut, curIn);
                }
                Array.Copy(curIn, midOut, chT);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(midBufA);
                ArrayPool<float>.Shared.Return(midBufB);
            }
        }

        var upIn = new float[2 * channels * t];
        CopyChannels(midOut, upIn, 0, channels, t);
        CopyChannels(skip, upIn, channels, channels, t);
        var upOut = ResnetBlock(upIn, 2 * channels, t, timeEmb, up.Resnet, channels);
        var upBufA = ArrayPool<float>.Shared.Rent(chT);
        var upBufB = ArrayPool<float>.Shared.Rent(chT);
        try
        {
            Array.Copy(upOut, upBufA, chT);
            float[] curIn = upBufA, curOut = upBufB;
            foreach (var tb in up.TransformerBlocks)
            {
                TransformerBlock(curIn, curOut, channels, t, tb, heads, headDim, scratch);
                (curIn, curOut) = (curOut, curIn);
            }
            Array.Copy(curIn, upOut, chT);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(upBufA);
            ArrayPool<float>.Shared.Return(upBufB);
        }
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
        LayerNormChannelFirstInPlace(h, dimOut, t, rw.Block1LnWeight, rw.Block1LnBias);
        MishInPlace(h);

        var mishTime = ArrayPool<float>.Shared.Rent(timeEmb.Length);
        try
        {
            for (int i = 0; i < timeEmb.Length; i++) mishTime[i] = Mish(timeEmb[i]);
            var timeProj = Linear(mishTime, rw.MlpWeight, rw.MlpBias);
            for (int c = 0; c < dimOut; c++)
            {
                float bias = timeProj[c];
                int rowBase = c * t;
                for (int ti = 0; ti < t; ti++) h[rowBase + ti] += bias;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(mishTime);
        }

        var h2 = CausalConv1d(h, dimOut, t, rw.Block2ConvWeight, rw.Block2ConvBias, dimOut, kernel: 3);
        LayerNormChannelFirstInPlace(h2, dimOut, t, rw.Block2LnWeight, rw.Block2LnBias);
        MishInPlace(h2);

        var resConv = Conv1dK1(x, dimIn, t, rw.ResConvWeight, rw.ResConvBias, dimOut);

        int vLen = Vector<float>.Count;
        int limit = h2.Length - (h2.Length % vLen);
        for (int i = 0; i < limit; i += vLen)
        {
            var vh = new Vector<float>(h2, i);
            var vr = new Vector<float>(resConv, i);
            (vh + vr).CopyTo(h2, i);
        }
        for (int i = limit; i < h2.Length; i++)
            h2[i] += resConv[i];

        return h2;
    }

    private sealed class UnetScratchBuffers : IDisposable
    {
        public readonly float[] Normed;
        public readonly float[] AttnOut;
        public readonly float[] AfterAttn;
        public readonly float[] Normed3;
        public readonly float[] FfUp;
        public readonly float[] FfDown;
        public readonly float[] QBuf;
        public readonly float[] KBuf;
        public readonly float[] VBuf;
        public readonly float[] ContextBuf;

        public UnetScratchBuffers(int t, int dim, int maxFfDim, int qkvDim)
        {
            int tDim = t * dim;
            int tFf = t * maxFfDim;
            int tQkv = t * qkvDim;

            Normed = ArrayPool<float>.Shared.Rent(tDim);
            AttnOut = ArrayPool<float>.Shared.Rent(tDim);
            AfterAttn = ArrayPool<float>.Shared.Rent(tDim);
            Normed3 = ArrayPool<float>.Shared.Rent(tDim);
            FfUp = ArrayPool<float>.Shared.Rent(tFf);
            FfDown = ArrayPool<float>.Shared.Rent(tDim);
            QBuf = ArrayPool<float>.Shared.Rent(tQkv);
            KBuf = ArrayPool<float>.Shared.Rent(tQkv);
            VBuf = ArrayPool<float>.Shared.Rent(tQkv);
            ContextBuf = ArrayPool<float>.Shared.Rent(tQkv);
        }

        public void Dispose()
        {
            ArrayPool<float>.Shared.Return(Normed);
            ArrayPool<float>.Shared.Return(AttnOut);
            ArrayPool<float>.Shared.Return(AfterAttn);
            ArrayPool<float>.Shared.Return(Normed3);
            ArrayPool<float>.Shared.Return(FfUp);
            ArrayPool<float>.Shared.Return(FfDown);
            ArrayPool<float>.Shared.Return(QBuf);
            ArrayPool<float>.Shared.Return(KBuf);
            ArrayPool<float>.Shared.Return(VBuf);
            ArrayPool<float>.Shared.Return(ContextBuf);
        }
    }

    private static unsafe void TransformerBlock(float[] x, float[] output, int dim, int t, IUnetTransformerBlockWeights tw, int heads, int headDim, UnetScratchBuffers scratch)
    {
        int ffDim = tw.FfUpWeight.OutDim;
        int tFf = t * ffDim;

        var normed = scratch.Normed;
        var attnOut = scratch.AttnOut;
        var afterAttn = scratch.AfterAttn;
        var normed3 = scratch.Normed3;
        var ffUp = scratch.FfUp;
        var ffDown = scratch.FfDown;

        LayerNormChannelFirstToRowMajor(x, normed, dim, t, tw.Norm1Weight, tw.Norm1Bias);
        SelfAttention(normed, attnOut, t, dim, tw, heads, headDim, scratch);

        System.Threading.Tasks.Parallel.For(0, dim, c =>
        {
            int cOff = c * t;
            for (int ti = 0; ti < t; ti++)
                afterAttn[cOff + ti] = x[cOff + ti] + attnOut[ti * dim + c];
        });

        LayerNormChannelFirstToRowMajor(afterAttn, normed3, dim, t, tw.Norm3Weight, tw.Norm3Bias);

        fixed (float* n3Ptr = normed3, ffUpPtr = ffUp, ffUpBiasPtr = tw.FfUpBias,
                      ffDownPtr = ffDown, ffDownBiasPtr = tw.FfDownBias)
        {
            tw.FfUpWeight.MatMul(n3Ptr, t, ffUpPtr, ffUpBiasPtr);
            GeluInPlace(ffUp, tFf);
            tw.FfDownWeight.MatMul(ffUpPtr, t, ffDownPtr, ffDownBiasPtr);
        }

        System.Threading.Tasks.Parallel.For(0, dim, c =>
        {
            int cOff = c * t;
            for (int ti = 0; ti < t; ti++)
                output[cOff + ti] = afterAttn[cOff + ti] + ffDown[ti * dim + c];
        });
    }

    private static unsafe void SelfAttention(float[] inputRowMajor, float[] output, int t, int dim, IUnetTransformerBlockWeights tw, int heads, int headDim, UnetScratchBuffers scratch)
    {
        int qkvDim = heads * headDim;

        var qBuf = scratch.QBuf;
        var kBuf = scratch.KBuf;
        var vBuf = scratch.VBuf;
        var contextBuf = scratch.ContextBuf;

        fixed (float* inPtr = inputRowMajor, qPtr = qBuf, kPtr = kBuf, vPtr = vBuf, ctxPtr = contextBuf, outPtr = output, outBiasPtr = tw.OutBias)
        {
            tw.QWeight.MatMul(inPtr, t, qPtr);
            tw.KWeight.MatMul(inPtr, t, kPtr);
            tw.VWeight.MatMul(inPtr, t, vPtr);

            float scale = 1f / MathF.Sqrt(headDim);

            nint qAddr = (nint)qPtr;
            nint kAddr = (nint)kPtr;
            nint vAddr = (nint)vPtr;
            nint ctxAddr = (nint)ctxPtr;

            bool useAvx64 = headDim == 64 && Avx.IsSupported && Fma.IsSupported;

            System.Threading.Tasks.Parallel.For(0, heads * t, ht =>
            {
                int h = ht / t;
                int i = ht % t;
                int hOff = h * headDim;

                var scores = stackalloc float[t];
                float* qHead = (float*)qAddr;
                float* kHead = (float*)kAddr;
                float* vHead = (float*)vAddr;
                float* ctxHeadBase = (float*)ctxAddr;
                float* qRow = qHead + (nuint)i * (nuint)qkvDim + (nuint)hOff;
                float* ctxHead = ctxHeadBase + (nuint)i * (nuint)qkvDim + (nuint)hOff;

                if (useAvx64)
                {
                    var q0 = Avx.LoadVector256(qRow);
                    var q1 = Avx.LoadVector256(qRow + 8);
                    var q2 = Avx.LoadVector256(qRow + 16);
                    var q3 = Avx.LoadVector256(qRow + 24);
                    var q4 = Avx.LoadVector256(qRow + 32);
                    var q5 = Avx.LoadVector256(qRow + 40);
                    var q6 = Avx.LoadVector256(qRow + 48);
                    var q7 = Avx.LoadVector256(qRow + 56);

                    for (int j = 0; j < t; j++)
                    {
                        float* kRow = kHead + (nuint)j * (nuint)qkvDim + (nuint)hOff;
                        var acc = Fma.MultiplyAdd(q0, Avx.LoadVector256(kRow), Vector256<float>.Zero);
                        acc = Fma.MultiplyAdd(q1, Avx.LoadVector256(kRow + 8), acc);
                        acc = Fma.MultiplyAdd(q2, Avx.LoadVector256(kRow + 16), acc);
                        acc = Fma.MultiplyAdd(q3, Avx.LoadVector256(kRow + 24), acc);
                        acc = Fma.MultiplyAdd(q4, Avx.LoadVector256(kRow + 32), acc);
                        acc = Fma.MultiplyAdd(q5, Avx.LoadVector256(kRow + 40), acc);
                        acc = Fma.MultiplyAdd(q6, Avx.LoadVector256(kRow + 48), acc);
                        acc = Fma.MultiplyAdd(q7, Avx.LoadVector256(kRow + 56), acc);
                        scores[j] = Vector256.Sum(acc) * scale;
                    }

                    DenseKernels.SoftmaxInPlace(new Span<float>(scores, t));

                    var c0 = Vector256<float>.Zero;
                    var c1 = Vector256<float>.Zero;
                    var c2 = Vector256<float>.Zero;
                    var c3 = Vector256<float>.Zero;
                    var c4 = Vector256<float>.Zero;
                    var c5 = Vector256<float>.Zero;
                    var c6 = Vector256<float>.Zero;
                    var c7 = Vector256<float>.Zero;

                    for (int j = 0; j < t; j++)
                    {
                        float s = scores[j];
                        if (s == 0f) continue;
                        var sVec = Vector256.Create(s);
                        float* vRow = vHead + (nuint)j * (nuint)qkvDim + (nuint)hOff;
                        c0 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow), c0);
                        c1 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 8), c1);
                        c2 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 16), c2);
                        c3 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 24), c3);
                        c4 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 32), c4);
                        c5 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 40), c5);
                        c6 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 48), c6);
                        c7 = Fma.MultiplyAdd(sVec, Avx.LoadVector256(vRow + 56), c7);
                    }

                    Avx.Store(ctxHead, c0);
                    Avx.Store(ctxHead + 8, c1);
                    Avx.Store(ctxHead + 16, c2);
                    Avx.Store(ctxHead + 24, c3);
                    Avx.Store(ctxHead + 32, c4);
                    Avx.Store(ctxHead + 40, c5);
                    Avx.Store(ctxHead + 48, c6);
                    Avx.Store(ctxHead + 56, c7);
                }
                else
                {
                    for (int j = 0; j < t; j++)
                    {
                        float* kRow = kHead + (nuint)j * (nuint)qkvDim + (nuint)hOff;
                        scores[j] = SimdKernels.DotF32(qRow, kRow, headDim) * scale;
                    }

                    DenseKernels.SoftmaxInPlace(new Span<float>(scores, t));

                    for (int d = 0; d < headDim; d++) ctxHead[d] = 0f;

                    for (int j = 0; j < t; j++)
                    {
                        float s = scores[j];
                        if (s == 0f) continue;
                        float* vRow = vHead + (nuint)j * (nuint)qkvDim + (nuint)hOff;
                        for (int d = 0; d < headDim; d++)
                            ctxHead[d] += s * vRow[d];
                    }
                }
            });

            tw.OutWeight.MatMul(ctxPtr, t, outPtr, outBiasPtr);
        }
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

    private static void LayerNormChannelFirstInPlace(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
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
                x[c * t + ti] = (float)((x[c * t + ti] - mean) * invStd) * weight[c] + bias[c];
        }
    }

    private static float[] LayerNormChannelFirst(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[ch * t];
        Array.Copy(x, output, ch * t);
        LayerNormChannelFirstInPlace(output, ch, t, weight, bias, eps);
        return output;
    }

    private static void LayerNormChannelFirstToRowMajor(float[] x, float[] output, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
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
    }

    private static float[] LayerNormChannelFirstToRowMajor(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[t * ch];
        LayerNormChannelFirstToRowMajor(x, output, ch, t, weight, bias, eps);
        return output;
    }

    private static float Mish(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));

    private static void MishInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = Mish(x[i]);
    }

    private static void GeluInPlace(float[] x, int len)
    {
        const float c = 0.7978845608028654f;
        for (int i = 0; i < len; i++)
        {
            float v = x[i];
            float inner = c * (v + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
    }

    private static void GeluInPlace(float[] x)
    {
        GeluInPlace(x, x.Length);
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
