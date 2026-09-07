namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2 AudioVAE's real decoder forward pass, from
/// `examples/audio.cpp/src/models/voxcpm2/audiovae.cpp`'s `build_decoder`/`decoder_block`/
/// `residual_unit`/`causal_conv1d`/`causal_conv_transpose1d`/`snake_exact`/`apply_sr_condition`
/// (not guessed). Real, DAC-lineage architecture (same family as this codebase's existing
/// <see cref="Parler.DacDecoder"/>) but NOT identical: every conv here is CAUSAL (left-pad only,
/// no lookahead -- this codec is designed for streaming), the first decoder stage is a real
/// depthwise+pointwise separable conv pair (not a single full conv), residual units' first conv
/// is DEPTHWISE (not full, unlike DAC's), and each decoder block applies a real per-checkpoint
/// FiLM-style sample-rate conditioning (`sr_cond`: `x*scale+bias`, a fixed vector selected by the
/// real `output_sample_rate`'s bucket at load time) before its Snake activation.
///
/// <para><b>Decode-only</b>: this pipeline only ever DECODES generated continuous latent features
/// into audio (never re-encodes a reference clip through the AudioVAE encoder as part of this
/// port's scope) -- matches the same scoping decision made for MOSS-TTS-Nano's audio codec this
/// session.</para>
///
/// <para><b>Real per-stage math</b>: `decoder.model.0` (depthwise conv, k=7, causal) -&gt;
/// `decoder.model.1` (pointwise conv, k=1) -&gt; N decoder blocks (`sr_cond` FiLM -&gt; Snake -&gt;
/// causal `ConvTranspose1d` upsample (kernel `2*stride`, TRUNCATED to keep only the first
/// `steps*stride` output samples -- the real causal trick: a real, unclamped transpose-conv
/// produces `(steps-1)*stride+kernel` samples, of which only the causal prefix is kept) -&gt; 3x
/// residual units (dilations 1/3/9: Snake -&gt; depthwise causal conv k=7 -&gt; Snake -&gt; pointwise
/// conv k=1 -&gt; residual add)) -&gt; final Snake -&gt; causal conv k=7 (channels-&gt;1) -&gt; Tanh.</para>
/// </summary>
public static class VoxCpm2AudioVaeDecoder
{
    /// <summary>Real exact Snake activation: x + (1/(alpha+eps)) * sin(alpha*x)^2, per-channel alpha.</summary>
    private static float[] Snake(float[] x, int channels, int t, float[] alpha)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            float invA = 1f / (a + 1e-9f);
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

    private static float[] ApplySrCondition(float[] x, int channels, int t, float[] scale, float[] bias)
    {
        var output = new float[x.Length];
        Parallel.For(0, channels, c =>
        {
            float s = scale[c];
            float b = bias[c];
            int baseIdx = c * t;
            for (int i = 0; i < t; i++) output[baseIdx + i] = x[baseIdx + i] * s + b;
        });
        return output;
    }

    /// <summary>Real causal-padded FULL Conv1d (im2col+GEMM, same technique as `DacDecoder.FullConv1d`):
    /// left-pads by `(kernel-1)*dilation` (stride is always 1 in this decode-only path -- the
    /// reference's strided/output_padding variant is only used by the encoder, not ported here),
    /// then a standard valid convolution. Output length equals input length.</summary>
    [ThreadStatic] private static float[]? t_colBuf;

    private static float[] GetColBuffer(int size)
    {
        if (t_colBuf == null || t_colBuf.Length < size) t_colBuf = new float[Math.Max(size, 65536)];
        return t_colBuf;
    }

    private static unsafe float[] CausalConv1d(float[] x, int t, VoxCpm2Conv1dWeights w, int dilation)
    {
        int inCh = w.InChannels;
        int outCh = w.OutChannels;
        int kernel = w.Kernel;
        int leftPad = (kernel - 1) * dilation;

        if (w.Depthwise)
        {
            // out[c][ti] = bias[c] + sum_k weight[c][k] * x[c][ti - leftPad + k*dilation]
            var output = new float[outCh * t];
            Parallel.For(0, outCh, c =>
            {
                float b = w.Bias[c];
                int xBase = c * t;
                int wBase = c * kernel;
                int outBase = c * t;
                for (int ti = 0; ti < t; ti++)
                {
                    float sum = b;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - leftPad + k * dilation;
                        if ((uint)src < (uint)t) sum += w.Weight[wBase + k] * x[xBase + src];
                    }
                    output[outBase + ti] = sum;
                }
            });
            return output;
        }

        int rowLen = inCh * kernel;
        var col = GetColBuffer(t * rowLen);
        Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - leftPad + k * dilation;
                    col[rBase + k] = (uint)src < (uint)t ? x[xBase + src] : 0f;
                }
            }
        });

        var outResult = new float[outCh * t];
        fixed (float* colPtr = col, weightPtr = w.Weight, outputPtr = outResult)
        {
            var colPtrLocal = colPtr;
            var weightPtrLocal = weightPtr;
            var outputPtrLocal = outputPtr;
            Parallel.For(0, outCh, oc =>
            {
                float b = w.Bias[oc];
                float* wOc = weightPtrLocal + oc * rowLen;
                float* outBase = outputPtrLocal + oc * t;
                for (int ti = 0; ti < t; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colPtrLocal + ti * rowLen, rowLen);
            });
        }
        return outResult;
    }

    /// <summary>Real causal ConvTranspose1d: computes the full (unclamped) transpose-conv output
    /// of length `(t-1)*stride+kernel`, then keeps only the first `t*stride` samples (the causal
    /// prefix) -- matches `causal_conv_transpose1d`'s `ggml_view_3d` truncation exactly.</summary>
    private static float[] CausalConvTranspose1d(float[] x, int t, VoxCpm2ConvTranspose1dWeights w, int stride)
    {
        int inCh = w.InChannels;
        int outCh = w.OutChannels;
        int kernel = w.Kernel;
        int fullT = (t - 1) * stride + kernel;
        int keepT = t * stride;

        var full = new float[outCh * fullT];
        Parallel.For(0, outCh, oc =>
        {
            float b = w.Bias[oc];
            int dstBase = oc * fullT;
            for (int ti = 0; ti < fullT; ti++) full[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = x[srcBase + ti];
                    int outStart = ti * stride; // padding=0 in the reference's transpose-conv call

                    int kStart = 0;
                    int kEnd = outStart + kernel > fullT ? fullT - outStart : kernel;
                    if (kStart >= kEnd) continue;

                    var wSpan = w.Weight.AsSpan(wBase + kStart, kEnd - kStart);
                    var dstSpan = full.AsSpan(dstBase + outStart + kStart, kEnd - kStart);
                    TensorPrimitives.MultiplyAdd(wSpan, v, dstSpan, dstSpan);
                }
            }
        });

        if (keepT == fullT) return full;
        var output = new float[outCh * keepT];
        for (int c = 0; c < outCh; c++)
            Array.Copy(full, c * fullT, output, c * keepT, keepT);
        return output;
    }

    private static float[] ResidualUnit(float[] x, int channels, int t, VoxCpm2ResidualUnitWeights w, int dilation)
    {
        var h = Snake(x, channels, t, w.Snake1Alpha);
        h = CausalConv1d(h, t, w.Conv1, dilation);
        h = Snake(h, channels, t, w.Snake2Alpha);
        h = CausalConv1d(h, t, w.Conv2, dilation: 1);

        var output = new float[h.Length];
        for (int i = 0; i < h.Length; i++) output[i] = x[i] + h[i];
        return output;
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int t, VoxCpm2DecoderBlockWeights w)
    {
        var h = ApplySrCondition(x, w.InputChannels, t, w.SrCondScale, w.SrCondBias);
        h = Snake(h, w.InputChannels, t, w.SnakeAlpha);
        h = CausalConvTranspose1d(h, t, w.Upsample, w.Stride);
        int outT = t * w.Stride;

        h = ResidualUnit(h, w.OutputChannels, outT, w.Res[0], dilation: 1);
        h = ResidualUnit(h, w.OutputChannels, outT, w.Res[1], dilation: 3);
        h = ResidualUnit(h, w.OutputChannels, outT, w.Res[2], dilation: 9);
        return (h, outT);
    }

    /// <summary>
    /// Decodes `[latentDim, frames]` channel-major continuous latent features into mono float32
    /// PCM at the checkpoint's real `output_sample_rate` (post-Tanh, range [-1, 1]).
    /// </summary>
    public static float[] Decode(VoxCpm2AudioVaeConfig config, VoxCpm2AudioVaeDecoderWeights w, float[] latentChannelMajor, int frames)
    {
        var x = CausalConv1d(latentChannelMajor, frames, w.DecoderFirstDepthwise, dilation: 1);
        x = CausalConv1d(x, frames, w.DecoderFirstPointwise, dilation: 1);

        int curT = frames;
        foreach (var block in w.DecoderBlocks)
            (x, curT) = DecoderBlock(x, curT, block);

        int finalChannels = config.DecoderDim >> config.DecoderRates.Length;
        x = Snake(x, finalChannels, curT, w.DecoderFinalSnakeAlpha);
        var pcm = CausalConv1d(x, curT, w.DecoderFinalConv, dilation: 1);

        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Tanh(pcm[i]);
        return pcm;
    }
}
