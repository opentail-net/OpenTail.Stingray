using System;
using System.Buffers;
using System.Numerics;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Orpheus;

/// <summary>
/// Real SNAC 24kHz decoder forward pass, transcribed directly from the real `snac` Python
/// package (`snac/layers.py`, `snac/vq.py`, `snac/snac.py`, fetched via `pip download snac
/// --no-deps`), config confirmed from the real `hubertsiuzdak/snac_24khz` HF `config.json` -- see
/// <see cref="SnacWeights"/>'s doc comment for the full derivation and real tensor names. Only
/// the DECODE path is ported (`ResidualVectorQuantize.from_codes` -> `Decoder.forward`) since
/// Orpheus only ever calls `SNAC.decode(codes)`, never encodes real audio.
///
/// <para><b>NoiseBlock is made a no-op, matching the real C++ reference's documented choice, not
/// a shortcut of this port's own invention</b>: the real `NoiseBlock.forward` injects
/// `torch.randn(...)` at every call, making the real PyTorch decoder itself non-deterministic.
/// `examples/CrispASR/tools/reference_backends/orpheus_snac.py` (the real oracle used for golden
/// verification here) explicitly monkey-patches `NoiseBlock.forward = lambda self, x: x` for
/// exactly this reason, noting "the noise contribution is ~1e-2 of the signal RMS" -- this port
/// follows the same documented convention, so `noise.weight` is loaded but never used.</para>
///
/// <para><b>Real per-layer math, in order (do not reorder/guess)</b>: `Decoder.forward`:
/// depthwise conv (in0, k=7, 768ch) -&gt; pointwise conv (in1, 768-&gt;1024, k=1) -&gt; 4x
/// `DecoderBlock` (each: `Snake1d` -&gt; `ConvTranspose1d` upsample -&gt; [NoiseBlock, no-op'd] -&gt;
/// 3x `ResidualUnit` with dilations 1/3/9) -&gt; final `Snake1d` -&gt; conv (64-&gt;1, k=7) -&gt;
/// `Tanh`. Each `ResidualUnit`: `Snake1d` -&gt; depthwise dilated conv (k=7, `pad=(kernel-1)*
/// dilation/2`) -&gt; `Snake1d` -&gt; pointwise conv (k=1) -&gt; residual add (real code
/// center-crops the residual to match the conv output length via `pad=(x.T - y.T)/2`, but with
/// same-padding depthwise convs T never shrinks here, so the crop is always a no-op -- kept as
/// an explicit no-op-safe add, not silently assumed away).</para>
/// </summary>
public static class SnacDecoder
{
    /// <summary>Real in-place `snake(x, alpha)`: `x + (1/(alpha+1e-9)) * sin(alpha*x)^2`, per-channel alpha.</summary>
    private static void Snake1dInPlace(float[] x, int channels, int t, float[] alpha)
    {
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            float invA = 1f / (a + 1e-9f);
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = x[baseIdx + i];
                float s = MathF.Sin(a * v);
                x[baseIdx + i] = v + invA * s * s;
            }
        });
    }

    /// <summary>Real out-of-place `snake(x, alpha)`: writes `src + (1/(alpha+1e-9)) * sin(alpha*src)^2` into `dst`.</summary>
    private static void Snake1d(float[] src, float[] dst, int channels, int t, float[] alpha)
    {
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            float invA = 1f / (a + 1e-9f);
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = src[baseIdx + i];
                float s = MathF.Sin(a * v);
                dst[baseIdx + i] = v + invA * s * s;
            }
        });
    }

    /// <summary>Real depthwise (groups=channels) dilated Conv1d, same-padding (`pad=(kernel-1)*dilation/2`), weight layout [out=channels, inPerGroup=1, kernel] flat row-major (confirmed matches this GGUF's real flat byte layout, see SnacWeights doc comment).</summary>
    private static void DepthwiseConv1d(float[] x, float[] output, int channels, int t, float[] weight, float[] bias, int kernel, int dilation)
    {
        int pad = (kernel - 1) * dilation / 2;
        Parallel.For(0, channels, c =>
        {
            int xBase = c * t;
            int wBase = c * kernel;
            float b = bias[c];

            int tiValidStart = Math.Min(pad, t);
            int tiValidEnd = Math.Max(tiValidStart, t - pad);

            // Left boundary with bounds checking
            for (int ti = 0; ti < tiValidStart; ti++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k * dilation;
                    if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                }
                output[xBase + ti] = sum;
            }

            // Unrolled interior region (guaranteed within [0, t) bounds)
            if (kernel == 7)
            {
                float w0 = weight[wBase + 0];
                float w1 = weight[wBase + 1];
                float w2 = weight[wBase + 2];
                float w3 = weight[wBase + 3];
                float w4 = weight[wBase + 4];
                float w5 = weight[wBase + 5];
                float w6 = weight[wBase + 6];

                int d1 = dilation;
                int d2 = 2 * dilation;
                int d3 = 3 * dilation;
                int d4 = 4 * dilation;
                int d5 = 5 * dilation;
                int d6 = 6 * dilation;

                for (int ti = tiValidStart; ti < tiValidEnd; ti++)
                {
                    int xOff = xBase + (ti - pad);
                    output[xBase + ti] = b
                        + x[xOff] * w0
                        + x[xOff + d1] * w1
                        + x[xOff + d2] * w2
                        + x[xOff + d3] * w3
                        + x[xOff + d4] * w4
                        + x[xOff + d5] * w5
                        + x[xOff + d6] * w6;
                }
            }
            else
            {
                for (int ti = tiValidStart; ti < tiValidEnd; ti++)
                {
                    float sum = b;
                    int src0 = ti - pad;
                    for (int k = 0; k < kernel; k++)
                    {
                        sum += x[xBase + src0 + k * dilation] * weight[wBase + k];
                    }
                    output[xBase + ti] = sum;
                }
            }

            // Right boundary with bounds checking
            for (int ti = tiValidEnd; ti < t; ti++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k * dilation;
                    if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                }
                output[xBase + ti] = sum;
            }
        });
    }

    /// <summary>
    /// Real pointwise (kernel=1) Conv1d: weight layout [out, in, 1] flat row-major -> effectively a per-position Linear across channels.
    ///
    /// <para>`x` is channel-major ([ch, t]), so for fixed `ti` the `ic` values are strided `t`
    /// apart -- not vectorizable directly. Transpose once into a contiguous [t, inCh] buffer (a
    /// cheap gather, independent of `oc`), then each output channel reduces to one AVX2/FMA
    /// <see cref="SimdKernels.DotF32"/> call per timestep -- same im2col-style technique as
    /// <c>FishSpeechCodec.FullConv1d</c>/<c>DacDecoder.FullConv1d</c>, kernel=1 special case.</para>
    /// </summary>
    private static unsafe void PointwiseConv1d(float[] x, float[] output, int inCh, int outCh, int t, float[] weight, float[] bias)
    {
        int xTLen = t * inCh;
        var xT = ArrayPool<float>.Shared.Rent(xTLen);
        try
        {
            fixed (float* xPtr = x, xtPtr = xT, weightPtr = weight, outputPtr = output)
            {
                var xLocal = xPtr;
                var xtLocal = xtPtr;
                var weightLocal = weightPtr;
                var outputLocal = outputPtr;

                Parallel.For(0, t, ti =>
                {
                    float* row = xtLocal + ti * inCh;
                    for (int ic = 0; ic < inCh; ic++) row[ic] = xLocal[ic * t + ti];
                });

                if (System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
                {
                    Parallel.For(0, outCh, oc =>
                    {
                        float b = bias[oc];
                        float* wOc = weightLocal + oc * inCh;
                        float* outBase = outputLocal + oc * t;

                        int ti = 0;
                        for (; ti <= t - 4; ti += 4)
                        {
                            float* xt0 = xtLocal + (ti + 0) * inCh;
                            float* xt1 = xtLocal + (ti + 1) * inCh;
                            float* xt2 = xtLocal + (ti + 2) * inCh;
                            float* xt3 = xtLocal + (ti + 3) * inCh;

                            var acc0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                            var acc1 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                            var acc2 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                            var acc3 = System.Runtime.Intrinsics.Vector256<float>.Zero;

                            int k = 0;
                            for (; k <= inCh - 8; k += 8)
                            {
                                var w = System.Runtime.Intrinsics.X86.Avx.LoadVector256(wOc + k);
                                acc0 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(w, System.Runtime.Intrinsics.X86.Avx.LoadVector256(xt0 + k), acc0);
                                acc1 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(w, System.Runtime.Intrinsics.X86.Avx.LoadVector256(xt1 + k), acc1);
                                acc2 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(w, System.Runtime.Intrinsics.X86.Avx.LoadVector256(xt2 + k), acc2);
                                acc3 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(w, System.Runtime.Intrinsics.X86.Avx.LoadVector256(xt3 + k), acc3);
                            }

                            float sum0 = b + System.Runtime.Intrinsics.Vector256.Sum(acc0);
                            float sum1 = b + System.Runtime.Intrinsics.Vector256.Sum(acc1);
                            float sum2 = b + System.Runtime.Intrinsics.Vector256.Sum(acc2);
                            float sum3 = b + System.Runtime.Intrinsics.Vector256.Sum(acc3);

                            for (; k < inCh; k++)
                            {
                                float wk = wOc[k];
                                sum0 += wk * xt0[k];
                                sum1 += wk * xt1[k];
                                sum2 += wk * xt2[k];
                                sum3 += wk * xt3[k];
                            }

                            outBase[ti + 0] = sum0;
                            outBase[ti + 1] = sum1;
                            outBase[ti + 2] = sum2;
                            outBase[ti + 3] = sum3;
                        }

                        for (; ti < t; ti++)
                            outBase[ti] = b + SimdKernels.DotF32(wOc, xtLocal + ti * inCh, inCh);
                    });
                }
                else
                {
                    Parallel.For(0, outCh, oc =>
                    {
                        float b = bias[oc];
                        float* wOc = weightLocal + oc * inCh;
                        float* outBase = outputLocal + oc * t;
                        for (int ti = 0; ti < t; ti++)
                            outBase[ti] = b + SimdKernels.DotF32(wOc, xtLocal + ti * inCh, inCh);
                    });
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xT);
        }
    }

    /// <summary>Real ConvTranspose1d, weight layout [in, out, kernel] flat row-major (matches HiFTVocoderKernels' established convention). `output_padding` is always 0 for this checkpoint's strides (2/4/8 all even -> `stride % 2 == 0`), so it's omitted rather than modeled unused.</summary>
    private static unsafe void ConvTranspose1d(float[] x, float[] output, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
    {
        int outT = (t - 1) * stride - 2 * padding + kernel;
        fixed (float* xPtr = x, outputPtr = output, weightPtr = weight, biasPtr = bias)
        {
            var xLocal = xPtr;
            var outputLocal = outputPtr;
            var weightLocal = weightPtr;
            var biasLocal = biasPtr;

            int tiValidStart = (padding + stride - 1) / stride;
            if (tiValidStart < 0) tiValidStart = 0;
            if (tiValidStart > t) tiValidStart = t;

            int tiValidEnd = (outT + padding - kernel) / stride + 1;
            if (tiValidEnd < tiValidStart) tiValidEnd = tiValidStart;
            if (tiValidEnd > t) tiValidEnd = t;

            Parallel.For(0, outCh, oc =>
            {
                float b = biasLocal[oc];
                float* dstBase = outputLocal + oc * outT;
                for (int ti = 0; ti < outT; ti++) dstBase[ti] = b;

                for (int ic = 0; ic < inCh; ic++)
                {
                    float* srcBase = xLocal + ic * t;
                    float* wBase = weightLocal + (ic * outCh + oc) * kernel;

                    // Left edge
                    for (int ti = 0; ti < tiValidStart; ti++)
                    {
                        float v = srcBase[ti];
                        if (v == 0f) continue;
                        int outStart = ti * stride - padding;
                        for (int k = 0; k < kernel; k++)
                        {
                            int to = outStart + k;
                            if ((uint)to < (uint)outT) dstBase[to] += v * wBase[k];
                        }
                    }

                    // Interior region (bounds checks eliminated + SIMD vectorized)
                    if (kernel == 16 && System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
                    {
                        var w0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(wBase);
                        var w1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(wBase + 8);
                        for (int ti = tiValidStart; ti < tiValidEnd; ti++)
                        {
                            float v = srcBase[ti];
                            if (v == 0f) continue;
                            var vVec = System.Runtime.Intrinsics.Vector256.Create(v);
                            float* dst = dstBase + (ti * stride - padding);
                            var cur0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(dst);
                            var cur1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(dst + 8);
                            System.Runtime.Intrinsics.X86.Avx.Store(dst, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(vVec, w0, cur0));
                            System.Runtime.Intrinsics.X86.Avx.Store(dst + 8, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(vVec, w1, cur1));
                        }
                    }
                    else if (kernel == 8 && System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
                    {
                        var w0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(wBase);
                        for (int ti = tiValidStart; ti < tiValidEnd; ti++)
                        {
                            float v = srcBase[ti];
                            if (v == 0f) continue;
                            var vVec = System.Runtime.Intrinsics.Vector256.Create(v);
                            float* dst = dstBase + (ti * stride - padding);
                            var cur0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(dst);
                            System.Runtime.Intrinsics.X86.Avx.Store(dst, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(vVec, w0, cur0));
                        }
                    }
                    else if (kernel == 4 && System.Runtime.Intrinsics.X86.Sse.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
                    {
                        var w0 = System.Runtime.Intrinsics.X86.Sse.LoadVector128(wBase);
                        for (int ti = tiValidStart; ti < tiValidEnd; ti++)
                        {
                            float v = srcBase[ti];
                            if (v == 0f) continue;
                            var vVec = System.Runtime.Intrinsics.Vector128.Create(v);
                            float* dst = dstBase + (ti * stride - padding);
                            var cur0 = System.Runtime.Intrinsics.X86.Sse.LoadVector128(dst);
                            System.Runtime.Intrinsics.X86.Sse.Store(dst, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(vVec, w0, cur0));
                        }
                    }
                    else
                    {
                        for (int ti = tiValidStart; ti < tiValidEnd; ti++)
                        {
                            float v = srcBase[ti];
                            if (v == 0f) continue;
                            float* dst = dstBase + (ti * stride - padding);
                            for (int k = 0; k < kernel; k++)
                            {
                                dst[k] += v * wBase[k];
                            }
                        }
                    }

                    // Right edge
                    for (int ti = tiValidEnd; ti < t; ti++)
                    {
                        float v = srcBase[ti];
                        if (v == 0f) continue;
                        int outStart = ti * stride - padding;
                        for (int k = 0; k < kernel; k++)
                        {
                            int to = outStart + k;
                            if ((uint)to < (uint)outT) dstBase[to] += v * wBase[k];
                        }
                    }
                }
            });
        }
    }

    private static void ResidualUnit(float[] x, float[] scratch1, float[] scratch2, int channels, int t, SnacResidualUnitWeights w, int dilation)
    {
        Snake1d(x, scratch1, channels, t, w.Alpha0);
        DepthwiseConv1d(scratch1, scratch2, channels, t, w.Conv0Weight, w.Conv0Bias, kernel: 7, dilation: dilation);
        Snake1dInPlace(scratch2, channels, t, w.Alpha1);
        PointwiseConv1d(scratch2, scratch1, channels, channels, t, w.Conv1Weight, w.Conv1Bias);

        // Real code center-crops the residual if the conv output is shorter; same-padding convs
        // above never shrink T here, so this is always a no-op in practice -- vectorized in-place add:
        int len = channels * t;
        int vLen = Vector<float>.Count;
        int i = 0;
        for (; i <= len - vLen; i += vLen)
        {
            var vx = new Vector<float>(x, i);
            var vy = new Vector<float>(scratch1, i);
            (vx + vy).CopyTo(x, i);
        }
        for (; i < len; i++)
        {
            x[i] += scratch1[i];
        }
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, SnacDecoderBlockWeights w, int stride)
    {
        Snake1dInPlace(x, inCh, t, w.Alpha);
        int kernel = 2 * stride;
        int padding = (int)MathF.Ceiling(stride / 2f);
        int outT = (t - 1) * stride - 2 * padding + kernel;
        int upLen = outCh * outT;
        var up = ArrayPool<float>.Shared.Rent(upLen);
        ConvTranspose1d(x, up, inCh, outCh, t, w.UpWeight, w.UpBias, kernel, stride, padding);

        // NoiseBlock is a documented no-op here, see class doc comment -- w.Res is applied
        // directly to `up`, matching the real graph with NoiseBlock.forward patched to identity.
        var scratch1 = ArrayPool<float>.Shared.Rent(upLen);
        var scratch2 = ArrayPool<float>.Shared.Rent(upLen);
        try
        {
            ResidualUnit(up, scratch1, scratch2, outCh, outT, w.Res[0], dilation: 1);
            ResidualUnit(up, scratch1, scratch2, outCh, outT, w.Res[1], dilation: 3);
            ResidualUnit(up, scratch1, scratch2, outCh, outT, w.Res[2], dilation: 9);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch1);
            ArrayPool<float>.Shared.Return(scratch2);
        }
        return (up, outT);
    }

    /// <summary>Real `ResidualVectorQuantize.from_codes`: per-quantizer embedding lookup -> out_proj (pointwise conv) -> nearest-neighbor time-upsample by that quantizer's own stride -> sum across quantizers. `codes[i]` are the real de-interleaved SNAC codebook indices (see FunASR-analogous doc comment in the pipeline class for the token de-interleaving formula), each in `[0, CodebookSize)`.</summary>
    public static float[] QuantizerFromCodes(SnacWeights w, int[][] codes)
    {
        int tOut = codes[^1].Length; // codes[2] (stride 1) defines the decoder-input rate
        var zq = new float[SnacWeights.LatentDim * tOut];

        for (int qi = 0; qi < w.Quantizers.Length; qi++)
        {
            var q = w.Quantizers[qi];
            int tIn = codes[qi].Length;

            int embedLen = SnacWeights.CodebookDim * tIn;
            var embed = ArrayPool<float>.Shared.Rent(embedLen);
            int projLen = SnacWeights.LatentDim * tIn;
            var proj = ArrayPool<float>.Shared.Rent(projLen);

            try
            {
                // decode_code: embedding lookup [T, CodebookDim] -> effectively transpose to [CodebookDim, T].
                for (int ti = 0; ti < tIn; ti++)
                {
                    int code = codes[qi][ti];
                    int cbBase = code * SnacWeights.CodebookDim;
                    for (int d = 0; d < SnacWeights.CodebookDim; d++)
                        embed[d * tIn + ti] = q.Codebook[cbBase + d];
                }

                PointwiseConv1d(embed, proj, SnacWeights.CodebookDim, SnacWeights.LatentDim, tIn, q.OutProjWeight, q.OutProjBias);

                // repeat_interleave(stride, dim=-1): nearest-neighbor upsample along time, then sum in.
                int stride = q.Stride;
                Parallel.For(0, SnacWeights.LatentDim, d =>
                {
                    int srcBase = d * tIn;
                    int dstBase = d * tOut;
                    for (int ti = 0; ti < tIn; ti++)
                    {
                        float v = proj[srcBase + ti];
                        int dstStart = ti * stride;
                        for (int s = 0; s < stride; s++)
                            zq[dstBase + dstStart + s] += v;
                    }
                });
            }
            finally
            {
                ArrayPool<float>.Shared.Return(embed);
                ArrayPool<float>.Shared.Return(proj);
            }
        }

        return zq;
    }

    /// <summary>Full real decode: 3 codebook streams -> quantizer.from_codes -> Decoder.forward -> mono float32 PCM at 24kHz, range [-1, 1] (post-Tanh).</summary>
    public static float[] Decode(SnacWeights w, int[][] codes)
    {
        var zq = QuantizerFromCodes(w, codes);
        int t = codes[^1].Length;

        int in0Len = SnacWeights.LatentDim * t;
        var in0Buf = ArrayPool<float>.Shared.Rent(in0Len);
        int in1Len = SnacWeights.DecoderDim * t;
        var in1Buf = ArrayPool<float>.Shared.Rent(in1Len);

        float[] x;
        int ch = SnacWeights.DecoderDim;
        int curT = t;

        try
        {
            DepthwiseConv1d(zq, in0Buf, SnacWeights.LatentDim, t, w.In0Weight, w.In0Bias, kernel: 7, dilation: 1);
            PointwiseConv1d(in0Buf, in1Buf, SnacWeights.LatentDim, SnacWeights.DecoderDim, t, w.In1Weight, w.In1Bias);
            x = in1Buf;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(in0Buf);
        }

        for (int i = 0; i < SnacWeights.DecoderRates.Length; i++)
        {
            int outCh = ch / 2;
            (var nextX, int nextT) = DecoderBlock(x, ch, outCh, curT, w.DecBlocks[i], SnacWeights.DecoderRates[i]);
            ArrayPool<float>.Shared.Return(x);
            x = nextX;
            ch = outCh;
            curT = nextT;
        }

        float[] pcm;
        try
        {
            Snake1dInPlace(x, ch, curT, w.OutAlpha);
            pcm = FullConv1dToMono(x, ch, curT, w.OutWeight, w.OutBias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
        }

        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Tanh(pcm[i]);
        return pcm;
    }

    /// <summary>Real final conv: FULL (non-grouped, `groups=1`, unlike the depthwise `ResidualUnit`/`in0` convs above) Conv1d, `channels -> 1`, kernel=7, same-padding. Weight layout [out=1, in=channels, kernel] flat row-major.</summary>
    private static unsafe float[] FullConv1dToMono(float[] x, int channels, int t, float[] weight, float[] bias)
    {
        const int kernel = 7;
        const int pad = 3;
        int rowLen = channels * kernel;
        int colLen = t * rowLen;
        var col = ArrayPool<float>.Shared.Rent(colLen);
        try
        {
            fixed (float* xPtr = x, colPtr = col, weightPtr = weight)
            {
                var xLocal = xPtr;
                var colLocal = colPtr;
                var weightLocal = weightPtr;

                Parallel.For(0, t, ti =>
                {
                    int rowBase = ti * rowLen;
                    for (int c = 0; c < channels; c++)
                    {
                        int xBase = c * t;
                        int rBase = rowBase + c * kernel;
                        for (int k = 0; k < kernel; k++)
                        {
                            int src = ti - pad + k;
                            colLocal[rBase + k] = (uint)src < (uint)t ? xLocal[xBase + src] : 0f;
                        }
                    }
                });

                var output = new float[t];
                float b = bias[0];
                fixed (float* outputPtr = output)
                {
                    var outputLocal = outputPtr;
                    Parallel.For(0, t, ti => outputLocal[ti] = b + SimdKernels.DotF32(weightLocal, colLocal + ti * rowLen, rowLen));
                }
                return output;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(col);
        }
    }
}

