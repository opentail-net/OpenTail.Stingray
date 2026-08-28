using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Real Fish Speech S2 Pro codec DECODE-only forward pass, transcribed directly from the real
/// `fishaudio/fish-speech` GitHub repo's `fish_speech/models/dac/rvq.py`
/// (`DownsampleResidualVectorQuantize.decode`) and `modded_dac.py` (`DAC.decoder`, `ResidualUnit`,
/// `DecoderBlock`, causal conv helpers) -- see <see cref="FishSpeechCodecWeights"/>'s doc comment
/// for the full derivation, including the real correction that the 8-layer transformer
/// (`pre_module`) is NOT needed for decode-only inference.
///
/// <para><b>Real decode chain</b>: codes (1 semantic + 9 residual, all same time resolution) -&gt;
/// per-quantizer embedding lookup -&gt; out_proj -&gt; SUM (semantic + residual, no time-
/// upsampling within the quantizer itself, same flat pattern as Parler-TTS's DAC) -&gt; real
/// `post_module` (a bare RMSNorm, NOT a transformer) -&gt; 2x upsample stage (causal
/// ConvTranspose1d(k=2,stride=2) -&gt; `ConvNeXtBlock`) -&gt; `DAC.decoder` (causal conv -&gt; 4x
/// causal `DecoderBlock` -&gt; causal conv -&gt; Tanh).</para>
///
/// <para><b>All convolutions are CAUSAL (left-pad only)</b> -- confirmed from the real
/// `CausalConvNet`/`CausalTransConvNet` classes. For a same-resolution conv (kernel=7,
/// dilation=d, stride=1): pad LEFT by `(kernel-1)*dilation`, pad RIGHT by 0 -- output length
/// equals input length (derived from the real `get_extra_padding_for_conv1d` formula, confirmed
/// to reduce to a pure left-pad for stride=1). For the causal `ConvTranspose1d` (kernel=2*stride,
/// stride=stride): run the RAW (unpadded) transpose conv, giving length `(T+1)*stride`, then crop
/// `stride` samples off the RIGHT only (confirmed from the real `CausalTransConvNet.forward`'s
/// `unpad1d(x, (0, padding_right))` call, `padding_left=0` always for this kernel/stride
/// relationship).</para>
/// </summary>
public static class FishSpeechCodec
{
    /// <summary>
    /// Real `ResidualVectorQuantize.from_codes`-equivalent for one quantizer set: embedding lookup
    /// -> pointwise out_proj, summed across all quantizers in the set (semantic has 1, residual
    /// has 9). Same time resolution for every codebook, no upsampling.
    ///
    /// <para><b>Real bug found and fixed (2026-08-28, comparing directly against
    /// `examples/s2.cpp/src/s2_codec.cpp`'s real `clamp_decode_code`/`sanitize_decode_codes`)</b>:
    /// codes were used raw, with no bounds check, before this fix -- silently correct as long as
    /// every code stayed in-range by chance (true for plain greedy `Argmax`, which empirically
    /// never wandered near the boundary), but a real, reference-confirmed requirement once real
    /// temperature/top_p/top_k sampling was wired in (see `FishSpeechPipeline.GenerateFrames`):
    /// `fast_output.weight`'s real output width is 4096 (the SAME width as the semantic
    /// vocabulary, shared/reused), but each residual codebook's REAL embedding table
    /// (<see cref="FishSpeechCodecWeights.ResidualCodebookSize"/> = 1024) only has 1024 valid
    /// rows -- sampling (unlike greedy) can legitimately draw a value in [1024, 4095] for a
    /// residual codebook, which crashed this method with an `IndexOutOfRangeException` (caught via
    /// a real repro, not by inspection). The reference clamps every decoded code to its own
    /// codebook's valid range right before this exact embedding lookup (`code = max(0, min(code,
    /// codebook_size - 1))`) -- ported verbatim here, not guessed.</para>
    /// </summary>
    private static float[] QuantizerSetFromCodes(FishSpeechCodecQuantizerWeights[] quantizers, int[][] codes, int t, int codebookSize)
    {
        var zq = new float[FishSpeechCodecWeights.LatentDim * t];
        for (int qi = 0; qi < quantizers.Length; qi++)
        {
            var q = quantizers[qi];
            var embed = new float[FishSpeechCodecWeights.CodebookDim * t];
            for (int ti = 0; ti < t; ti++)
            {
                int code = Math.Clamp(codes[qi][ti], 0, codebookSize - 1);
                int cbBase = code * FishSpeechCodecWeights.CodebookDim;
                for (int d = 0; d < FishSpeechCodecWeights.CodebookDim; d++)
                    embed[d * t + ti] = q.Codebook[cbBase + d];
            }
            var proj = FullConv1d(embed, FishSpeechCodecWeights.CodebookDim, FishSpeechCodecWeights.LatentDim, t, q.OutProjWeight, q.OutProjBias, kernel: 1, dilation: 1, causalPadLeft: 0);
            for (int i = 0; i < zq.Length; i++) zq[i] += proj[i];
        }
        return zq;
    }

    /// <summary>
    /// Real causal FULL Conv1d: left-pad only by `(kernel-1)*dilation` (equivalently `causalPadLeft`), zero right-pad. Weight layout [out, in, kernel] flat row-major.
    ///
    /// <para>Implemented as im2col + GEMM (mirrors the real `ggml_conv_1d`'s own im2col+mul_mat
    /// lowering, see `examples/s2.cpp`'s `s2_codec.cpp`): the gather (im2col) is independent of
    /// `oc`, so it is hoisted out of the per-output-channel loop and done once; each output
    /// channel then reduces to one contiguous AVX2/FMA dot product per timestep via
    /// <see cref="SimdKernels.DotF32"/>, instead of a scalar `inCh*kernel` loop. Measured ~9x
    /// faster in the full codec decode (see docs/audio-review-progress.md's Fish Speech codec
    /// SIMD section) -- the previous scalar-loop version was the dominant remaining bottleneck.</para>
    /// </summary>
    private static unsafe float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int dilation, int causalPadLeft)
    {
        int rowLen = inCh * kernel;
        var col = new float[t * rowLen]; // [ti][ic*kernel+k], matches weight's [oc][ic*kernel+k] layout
        Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - causalPadLeft + k * dilation;
                    col[rBase + k] = (uint)src < (uint)t ? x[xBase + src] : 0f;
                }
            }
        });

        var output = new float[outCh * t];
        fixed (float* colPtr = col, weightPtr = weight, outputPtr = output)
        {
            var colPtrLocal = colPtr;
            var weightPtrLocal = weightPtr;
            var outputPtrLocal = outputPtr;
            Parallel.For(0, outCh, oc =>
            {
                float b = bias[oc];
                float* wOc = weightPtrLocal + oc * rowLen;
                float* outBase = outputPtrLocal + oc * t;
                for (int ti = 0; ti < t; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colPtrLocal + ti * rowLen, rowLen);
            });
        }
        return output;
    }

    /// <summary>Real causal depthwise Conv1d (groups=channels), same left-pad convention as <see cref="FullConv1d"/>. Weight layout [channels, 1, kernel] flat -> effectively [channels, kernel].</summary>
    private static float[] DepthwiseConv1d(float[] x, int channels, int t, float[] weight, float[] bias, int kernel, int causalPadLeft)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float b = bias[c];
            int wBase = c * kernel;
            int xBase = c * t;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - causalPadLeft + k;
                    if ((uint)src < (uint)t) sum += x[xBase + src] * weight[wBase + k];
                }
                output[xBase + ti] = sum;
            }
        });
        return output;
    }

    /// <summary>
    /// Real causal ConvTranspose1d: raw unpadded transpose conv (length becomes `(T-1)*stride+
    /// kernel`), then crop `pad = kernel - stride` samples off the RIGHT only (`padding_left=0`
    /// always, confirmed from the real `CausalTransConvNet.forward`'s `padding_left = pad -
    /// ceil(pad)` which is 0 whenever `pad` is already an integer, true for every real call site
    /// here). Weight layout [in, out, kernel] flat row-major.
    ///
    /// <para><b>Real bug found and fixed this fire</b>: the quantizer's own upsample stages call
    /// this with `kernel=stride` (real `rvq.py`: `transconvnet_type(..., kernel_size=factor,
    /// stride=factor)`), NOT `kernel=2*stride` like the DAC decoder's `DecoderBlock` -- an
    /// earlier version of this method hardcoded the crop to `stride`, which is only correct for
    /// the `kernel=2*stride` case (crop=stride) and silently produced HALF the correct output
    /// length for the `kernel=stride` case (crop=0, should be a no-op). Caught via the golden
    /// oracle producing an unexpectedly short PCM length (1024 instead of the expected 4096 for
    /// a 2-timestep input) before any cosine-similarity check was even needed.</para>
    /// </summary>
    private static float[] CausalConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride)
    {
        int rawT = (t - 1) * stride + kernel; // padding=0 in the underlying nn.ConvTranspose1d
        var raw = new float[outCh * rawT];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int dstBase = oc * rawT;
            for (int ti = 0; ti < rawT; ti++) raw[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = x[srcBase + ti];
                    int outStart = ti * stride;
                    for (int k = 0; k < kernel; k++)
                        raw[dstBase + outStart + k] += v * weight[wBase + k];
                }
            }
        });

        int cropRight = kernel - stride; // real `pad = kernel_size - stride`, padding_left=0
        int outT = rawT - cropRight;
        var output = new float[outCh * outT];
        for (int oc = 0; oc < outCh; oc++)
            Array.Copy(raw, oc * rawT, output, oc * outT, outT);
        return output;
    }

    /// <summary>
    /// Real Snake activation, ported verbatim from `examples/s2.cpp/src/s2_codec.cpp`'s
    /// `snake_activation`: `x + sin(alpha*x)^2 / alpha` -- confirmed via direct source
    /// comparison this has NO epsilon anywhere (this port previously added `1e-9` to the
    /// denominator, `1/(alpha+1e-9)`, a real discrepancy from the reference, found while
    /// investigating a "gargly" audio-quality report -- see docs/audio-review-progress.md).
    /// </summary>
    private static float[] Snake1d(float[] x, int channels, int t, float[] alpha)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            float invA = 1f / a;
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = x[baseIdx + i];
                float s = MathF.Sin(a * v);
                output[baseIdx + i] = v + invA * s * s;
            }
        });
        return output;
    }

    /// <summary>Real `ConvNeXtBlock`: causal depthwise conv (k=7) -> per-position (channels-last) LayerNorm -> Linear expand (4x) -> GELU -> Linear project -> per-channel LayerScale (`gamma`) -> residual.</summary>
    private static unsafe float[] ConvNeXtBlock(float[] x, int channels, int t, FishSpeechConvNeXtBlockWeights w)
    {
        var y = DepthwiseConv1d(x, channels, t, w.DwConvWeight, w.DwConvBias, kernel: 7, causalPadLeft: 6);

        int hidden = channels * 4;
        var output = new float[x.Length];
        fixed (float* pw1 = w.PwConv1Weight, pw2 = w.PwConv2Weight)
        {
            var pw1Local = pw1;
            var pw2Local = pw2;
            Parallel.For(0, t, ti =>
            {
                // Gather this position's channel vector (channels-last for LayerNorm/Linear;
                // contiguous so the two Linear layers below can use a straight AVX2 dot product).
                var row = new float[channels];
                for (int c = 0; c < channels; c++) row[c] = y[c * t + ti];

                float mean = 0f;
                for (int c = 0; c < channels; c++) mean += row[c];
                mean /= channels;
                float variance = 0f;
                for (int c = 0; c < channels; c++) { float d = row[c] - mean; variance += d * d; }
                variance /= channels;
                float invStd = 1f / MathF.Sqrt(variance + 1e-6f);
                for (int c = 0; c < channels; c++) row[c] = (row[c] - mean) * invStd * w.NormWeight[c] + w.NormBias[c];

                var h = new float[hidden];
                fixed (float* rowPtr = row, hPtr = h)
                {
                    for (int o = 0; o < hidden; o++)
                        hPtr[o] = Gelu(w.PwConv1Bias[o] + SimdKernels.DotF32(rowPtr, pw1Local + o * channels, channels));

                    for (int c = 0; c < channels; c++)
                    {
                        float sum = w.PwConv2Bias[c] + SimdKernels.DotF32(hPtr, pw2Local + c * hidden, hidden);
                        output[c * t + ti] = x[c * t + ti] + sum * w.Gamma[c];
                    }
                }
            });
        }
        return output;
    }

    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    private static float Erf(float x)
    {
        float sign = MathF.Sign(x);
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }

    private static float[] ResidualUnit(float[] x, int channels, int t, FishSpeechCodecResidualUnitWeights w, int dilation)
    {
        int padLeft = (7 - 1) * dilation;
        var y = Snake1d(x, channels, t, w.Alpha0);
        y = FullConv1d(y, channels, channels, t, w.Conv0Weight, w.Conv0Bias, kernel: 7, dilation: dilation, causalPadLeft: padLeft);
        y = Snake1d(y, channels, t, w.Alpha1);
        y = FullConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias, kernel: 1, dilation: 1, causalPadLeft: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i];
        return output;
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, FishSpeechCodecDecoderBlockWeights w, int stride)
    {
        var y = Snake1d(x, inCh, t, w.Alpha);
        int kernel = 2 * stride;
        var up = CausalConvTranspose1d(y, inCh, outCh, t, w.UpWeight, w.UpBias, kernel, stride);
        int outT = t * stride;

        var cur = up;
        cur = ResidualUnit(cur, outCh, outT, w.Res[0], dilation: 1);
        cur = ResidualUnit(cur, outCh, outT, w.Res[1], dilation: 3);
        cur = ResidualUnit(cur, outCh, outT, w.Res[2], dilation: 9);
        return (cur, outT);
    }

    /// <summary>
    /// Real `quantizer.post_module` transformer, ported directly from `examples/s2.cpp/src/
    /// s2_codec.cpp`'s `build_transformer` (see `FishSpeechCodecWeights`'s doc comment for full
    /// derivation and why an earlier pass wrongly skipped this entirely): N pre-norm layers, each
    /// RMSNorm -&gt; fused-QKV self-attention (full MHA, no GQA, real interleaved RoPE, a sliding
    /// causal window that's a no-op for any `t` shorter than the window) -&gt; per-channel
    /// multiplicative LayerScale -&gt; residual -&gt; RMSNorm -&gt; SwiGLU FFN -&gt; per-channel
    /// LayerScale -&gt; residual, followed by one final RMSNorm (`PostModuleNormWeight` -- the
    /// tensor the earlier, incorrect pass found and mistook for the entire module).
    /// </summary>
    private static float[] ApplyQuantizerTransformer(float[] zq, int t, FishSpeechCodecWeights w)
    {
        int dim = FishSpeechCodecWeights.LatentDim;
        int nHead = w.QuantizerTransformerNumHeads;
        int headDim = w.QuantizerTransformerHeadDim;
        float ropeBase = w.QuantizerTransformerRopeBase;
        float eps = w.QuantizerTransformerNormEps;
        int windowSize = w.QuantizerTransformerWindowSize;

        // zq is channel-first [dim, t]; transformer math is naturally per-position -- convert to x[t][dim].
        var x = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = zq[d * t + ti];
            x[ti] = row;
        }

        foreach (var lw in w.PostModuleTransformerLayers)
            x = QuantizerTransformerLayer(x, lw, t, dim, nHead, headDim, ropeBase, eps, windowSize);

        for (int ti = 0; ti < t; ti++)
            x[ti] = FishSpeechFastAr.RmsNorm(x[ti], w.PostModuleNormWeight, eps);

        var output = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
            for (int d = 0; d < dim; d++)
                output[d * t + ti] = x[ti][d];
        return output;
    }

    private static float[][] QuantizerTransformerLayer(float[][] x, FishSpeechCodecTransformerLayerWeights lw, int t, int dim, int nHead, int headDim, float ropeBase, float eps, int windowSize)
    {
        int qkvSize = nHead * headDim; // == dim, full MHA (n_local_heads == n_head for this checkpoint)

        // Real perf fix (2026-08-29): each position's RMSNorm/QKV projection is independent of
        // every other position (only the attention step itself, already parallelized over heads
        // below, mixes across positions) -- was a plain sequential loop over t, single-threaded,
        // for what's actually an embarrassingly parallel per-position workload. Measured as the
        // dominant cost in the codec's newly-added quantizer transformer (see this file's Decode
        // doc comment) -- 15-frame codec decode alone took ~2.6s before this fix.
        var normed = new float[t][];
        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        Parallel.For(0, t, i =>
        {
            normed[i] = FishSpeechFastAr.RmsNorm(x[i], lw.AttentionNormWeight, eps);
            var qkv = FishSpeechFastAr.LinearNoBias(normed[i], lw.WqkvWeight, dim, 3 * qkvSize);
            q[i] = qkv.AsSpan(0, qkvSize).ToArray();
            k[i] = qkv.AsSpan(qkvSize, qkvSize).ToArray();
            v[i] = qkv.AsSpan(2 * qkvSize, qkvSize).ToArray();
        });

        for (int i = 0; i < t; i++)
        {
            FishSpeechFastAr.ApplyRope(q[i], nHead, headDim, i, ropeBase);
            FishSpeechFastAr.ApplyRope(k[i], nHead, headDim, i, ropeBase);
        }

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[qkvSize];

        bool useWindow = windowSize > 0 && windowSize < t;
        float scale = 1f / MathF.Sqrt(headDim);
        Parallel.For(0, nHead, h =>
        {
            int off = h * headDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                int minK = useWindow ? Math.Max(0, i - windowSize + 1) : 0;
                for (int j = minK; j <= i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][off + d] * k[j][off + d];
                    scores[j] = dot * scale;
                }

                float max = float.NegativeInfinity;
                for (int j = minK; j <= i; j++) if (scores[j] > max) max = scores[j];
                float sum = 0f;
                for (int j = minK; j <= i; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                float inv = 1f / sum;

                var ctxSpan = context[i].AsSpan(off, headDim);
                for (int j = minK; j <= i; j++)
                {
                    float p = scores[j] * inv;
                    for (int d = 0; d < headDim; d++) ctxSpan[d] += p * v[j][off + d];
                }
            }
        });

        var h1 = new float[t][];
        Parallel.For(0, t, i =>
        {
            var o = FishSpeechFastAr.LinearNoBias(context[i], lw.WoWeight, qkvSize, dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + o[d] * lw.AttentionGamma[d];
            h1[i] = row;
        });

        int ffnDim = lw.W1Weight.Length / dim;
        var output = new float[t][];
        Parallel.For(0, t, i =>
        {
            var ffnNormed = FishSpeechFastAr.RmsNorm(h1[i], lw.FfnNormWeight, eps);
            var gate = FishSpeechFastAr.LinearNoBias(ffnNormed, lw.W1Weight, dim, ffnDim);
            var up = FishSpeechFastAr.LinearNoBias(ffnNormed, lw.W3Weight, dim, ffnDim);
            for (int d = 0; d < ffnDim; d++) gate[d] = FishSpeechFastAr.Silu(gate[d]) * up[d];
            var ffnOut = FishSpeechFastAr.LinearNoBias(gate, lw.W2Weight, ffnDim, dim);

            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = h1[i][d] + ffnOut[d] * lw.FfnGamma[d];
            output[i] = row;
        });
        return output;
    }

    /// <summary>Full real decode: 10 codebook streams (1 semantic + 9 residual, same time resolution) -> mono float32 PCM at 44.1kHz, range [-1, 1] (post-Tanh).</summary>
    public static float[] Decode(FishSpeechCodecWeights w, int[] semanticCodes, int[][] residualCodes)
    {
        int t = semanticCodes.Length;

        var zqSemantic = QuantizerSetFromCodes([w.SemanticQuantizer], [semanticCodes], t, FishSpeechCodecWeights.SemanticCodebookSize);
        var zqResidual = QuantizerSetFromCodes(w.ResidualQuantizers, residualCodes, t, FishSpeechCodecWeights.ResidualCodebookSize);
        var zq = new float[zqSemantic.Length];
        for (int i = 0; i < zq.Length; i++) zq[i] = zqSemantic[i] + zqResidual[i];

        // Real `quantizer.post_module`: a full 8-layer transformer (see FishSpeechCodecWeights's
        // doc comment) -- NOT just the final RMSNorm this port previously stopped at. That earlier
        // scoping mistake silently skipped the entire transformer while investigating a "gargly"
        // audio-quality report; fixed by porting `build_transformer` faithfully (below).
        zq = ApplyQuantizerTransformer(zq, t, w);

        var x = zq;
        int curT = t;
        foreach (var stage in w.UpsampleStages)
        {
            x = CausalConvTranspose1d(x, FishSpeechCodecWeights.LatentDim, FishSpeechCodecWeights.LatentDim, curT, stage.ConvWeight, stage.ConvBias, kernel: 2, stride: 2);
            curT *= 2;
            x = ConvNeXtBlock(x, FishSpeechCodecWeights.LatentDim, curT, stage.Block);
        }

        x = FullConv1d(x, FishSpeechCodecWeights.LatentDim, FishSpeechCodecWeights.DecoderDim, curT, w.DecIn0Weight, w.DecIn0Bias, kernel: 7, dilation: 1, causalPadLeft: 6);

        int ch = FishSpeechCodecWeights.DecoderDim;
        for (int i = 0; i < FishSpeechCodecWeights.DecoderRates.Length; i++)
        {
            int outCh = ch / 2;
            (x, curT) = DecoderBlock(x, ch, outCh, curT, w.DecBlocks[i], FishSpeechCodecWeights.DecoderRates[i]);
            ch = outCh;
        }

        x = Snake1d(x, ch, curT, w.DecOutAlpha);
        var pcm = FullConv1d(x, ch, 1, curT, w.DecOutWeight, w.DecOutBias, kernel: 7, dilation: 1, causalPadLeft: 6);

        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Tanh(pcm[i]);
        return pcm;
    }
}
