
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real RVC synthesizer forward pass, transcribed directly from
/// `examples/audio.cpp/src/models/rvc/synthesizer.cpp`'s `build_text_encoder_stats`/
/// `build_flow_reverse`/`build_generator` (not guessed) -- see
/// `docs/audio-review-progress.md`'s RVC section for the architecture notes. A VITS-style
/// pipeline: a 6-layer relative-position-attention text encoder produces per-frame Gaussian
/// (mean, log-std) statistics over a 192-dim latent; a sampled `z_p` is pushed backward through a
/// 4-block affine-coupling normalizing flow (conditioned on a speaker embedding) to get `z`; `z`
/// then drives an NSF-HiFiGAN generator (with a sine-excitation source for f0-aware models) to
/// produce the final waveform. Tensors are channel-major flat arrays (`[C*T]`, index `c*T+t`,
/// matching this codebase's `HiFTVocoderKernels` convention) except where noted.
/// </summary>
public static class RvcSynthesizerEncoder
{
    private const float LReluSlope = 0.1f;
    private const float LayerNormEps = 1e-5f;

    /// <summary>All real intermediate tensors, exposed for golden-verification against the
    /// reference's own `rvc.synth.{m,logs,z_p,z,raw_output}` traces. All channel-major [C*T]
    /// except <see cref="Audio"/> (raw waveform samples).</summary>
    public readonly record struct ForwardResult(float[] M, float[] Logs, float[] Zp, float[] Z, float[] Audio);

    /// <summary>Real end-to-end forward pass. <paramref name="featuresBtc"/> is frame-major
    /// [frames][featureDim] (HuBERT content, optionally retrieval-blended). <paramref
    /// name="pitchIds"/>/<paramref name="sineSource"/> are only used when <c>weights.HasF0</c>.
    /// <paramref name="noiseChannelMajor"/> is the real Gaussian noise added to `z_p`
    /// (`[InterChannels*frames]`, channel-major) -- the reference generates this via a
    /// Philox-based CUDA RNG (`generate_torch_cuda_randn`, seed 1234) that this port does not
    /// replicate; for golden verification, pass the reference's own dumped noise array directly
    /// rather than trying to reproduce its RNG bit-for-bit. Returns the raw generator waveform
    /// (pre RMS-mix / pad-crop, matching the reference's own `RvcSynthesizerOutput.audio` before
    /// `native_pipeline.cpp`'s post-processing).</summary>
    public static ForwardResult Forward(
        RvcSynthesizerWeights w,
        float[][] featuresBtc,
        int[] pitchIds,
        ReadOnlySpan<float> sineSource,
        int speakerId,
        ReadOnlySpan<float> noiseChannelMajor)
    {
        int frames = featuresBtc.Length;
        var (m, logs) = BuildTextEncoderStats(w, featuresBtc, pitchIds);

        int inter = RvcSynthesizerWeights.InterChannels;
        var zp = new float[inter * frames];
        for (int i = 0; i < zp.Length; i++)
            zp[i] = m[i] + MathF.Exp(logs[i]) * noiseChannelMajor[i] * 0.66666f;

        var g = SpeakerEmbedding(w, speakerId); // [GinChannels]
        var z = BuildFlowReverse(w, zp, frames, g);
        var audio = BuildGenerator(w, z, frames, g, sineSource);
        return new ForwardResult(m, logs, zp, z, audio);
    }

    /// <summary>Convenience overload for non-golden-verification callers: samples its own noise
    /// from <paramref name="noiseRng"/> instead of requiring an externally-supplied array.</summary>
    public static ForwardResult Forward(
        RvcSynthesizerWeights w,
        float[][] featuresBtc,
        int[] pitchIds,
        ReadOnlySpan<float> sineSource,
        int speakerId,
        Random noiseRng)
    {
        int frames = featuresBtc.Length;
        var noise = new float[RvcSynthesizerWeights.InterChannels * frames];
        for (int i = 0; i < noise.Length; i++) noise[i] = SampleStandardNormal(noiseRng);
        return Forward(w, featuresBtc, pitchIds, sineSource, speakerId, noise);
    }

    private static float SampleStandardNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    // ---------------------------------------------------------------- text encoder

    private static (float[] M, float[] Logs) BuildTextEncoderStats(RvcSynthesizerWeights w, float[][] featuresBtc, int[] pitchIds)
    {
        int frames = featuresBtc.Length;
        int hidden = RvcSynthesizerWeights.HiddenChannels;
        int featureDim = featuresBtc[0].Length;

        // x[t] = emb_phone(features[t]) [+ emb_pitch[pitchIds[t]]], scaled, leaky-relu'd.
        var xBtc = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = LinearRow(featuresBtc[t], w.EmbPhoneWeight, w.EmbPhoneBias, featureDim, hidden);
            if (w.HasF0)
            {
                var pitchRow = w.EmbPitchWeight!;
                int baseIdx = pitchIds[t] * hidden;
                for (int c = 0; c < hidden; c++) row[c] += pitchRow[baseIdx + c];
            }
            float scale = MathF.Sqrt(hidden);
            for (int c = 0; c < hidden; c++) row[c] = LeakyRelu(row[c] * scale);
            xBtc[t] = row;
        }

        var x = ToChannelMajor(xBtc, hidden, frames);
        for (int layer = 0; layer < RvcSynthesizerWeights.TextLayers; layer++)
        {
            var y = SelfAttention(x, hidden, frames, w.AttnLayers[layer]);
            AddInPlace(x, y);
            x = LayerNormBct(x, hidden, frames, w.NormLayers1[layer]);
            y = Ffn(x, hidden, frames, w.FfnConv1[layer], w.FfnConv2[layer]);
            AddInPlace(x, y);
            x = LayerNormBct(x, hidden, frames, w.NormLayers2[layer]);
        }

        // enc_p.proj: Conv1d(hidden -> InterChannels*2, kernel=1), applied as a per-frame Linear.
        int interX2 = RvcSynthesizerWeights.InterChannels * 2;
        var stats = new float[interX2 * frames];
        for (int t = 0; t < frames; t++)
        {
            var xt = new float[hidden];
            for (int c = 0; c < hidden; c++) xt[c] = x[c * frames + t];
            var proj = LinearRow(xt, w.ProjWeight, w.ProjBias, hidden, interX2);
            for (int c = 0; c < interX2; c++) stats[c * frames + t] = proj[c];
        }

        int inter = RvcSynthesizerWeights.InterChannels;
        var m = new float[inter * frames];
        var logs = new float[inter * frames];
        Array.Copy(stats, 0, m, 0, inter * frames);
        Array.Copy(stats, inter * frames, logs, 0, inter * frames);
        return (m, logs);
    }

    private static float[] SelfAttention(float[] xBct, int hidden, int frames, RvcTextAttnLayer layer)
    {
        int heads = RvcSynthesizerWeights.Heads;
        int headDim = RvcSynthesizerWeights.HeadDim;
        int window = RvcSynthesizerWeights.RelativeWindowSize;
        float invSqrtD = 1f / MathF.Sqrt(headDim);

        var q = Conv1dK1Bct(xBct, hidden, frames, layer.ConvQ.Weight, layer.ConvQ.Bias, hidden);
        var k = Conv1dK1Bct(xBct, hidden, frames, layer.ConvK.Weight, layer.ConvK.Bias, hidden);
        var v = Conv1dK1Bct(xBct, hidden, frames, layer.ConvV.Weight, layer.ConvV.Bias, hidden);

        var outBct = new float[hidden * frames];
        var scores = new float[frames];
        for (int h = 0; h < heads; h++)
        {
            int chBase = h * headDim;
            for (int i = 0; i < frames; i++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < frames; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[(chBase + d) * frames + i] * k[(chBase + d) * frames + j];
                    dot *= invSqrtD;
                    int dist = j - i;
                    if (dist >= -window && dist <= window)
                    {
                        int relRow = (dist + window) * headDim;
                        float relDot = 0f;
                        for (int d = 0; d < headDim; d++) relDot += q[(chBase + d) * frames + i] * layer.EmbRelK[relRow + d];
                        dot += relDot * invSqrtD;
                    }
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = 0; j < frames; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;

                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < frames; j++) acc += scores[j] * invSum * v[(chBase + d) * frames + j];
                    outBct[(chBase + d) * frames + i] = acc;
                }
                for (int j = 0; j < frames; j++)
                {
                    int dist = j - i;
                    if (dist < -window || dist > window) continue;
                    int relRow = (dist + window) * headDim;
                    float weight = scores[j] * invSum;
                    for (int d = 0; d < headDim; d++)
                        outBct[(chBase + d) * frames + i] += weight * layer.EmbRelV[relRow + d];
                }
            }
        }
        return Conv1dK1Bct(outBct, hidden, frames, layer.ConvO.Weight, layer.ConvO.Bias, hidden);
    }

    private static float[] Ffn(float[] xBct, int hidden, int frames, RvcConv1d conv1, RvcConv1d conv2)
    {
        int filter = RvcSynthesizerWeights.FilterChannels;
        var y = Conv1dSamePad(xBct, hidden, frames, conv1.Weight, conv1.Bias, filter, kernel: 3);
        for (int i = 0; i < y.Length; i++) y[i] = MathF.Max(0f, y[i]); // real ReluModule (not leaky) in rvc_ffn
        return Conv1dSamePad(y, filter, frames, conv2.Weight, conv2.Bias, hidden, kernel: 3);
    }

    // ---------------------------------------------------------------- flow

    private static float[] BuildFlowReverse(RvcSynthesizerWeights w, float[] zpBct, int frames, float[] g)
    {
        int inter = RvcSynthesizerWeights.InterChannels;
        var z = zpBct;
        z = FlipChannels(z, inter, frames);
        z = ResidualCouplingReverse(z, inter, frames, g, w.Flows[3]);
        z = FlipChannels(z, inter, frames);
        z = ResidualCouplingReverse(z, inter, frames, g, w.Flows[2]);
        z = FlipChannels(z, inter, frames);
        z = ResidualCouplingReverse(z, inter, frames, g, w.Flows[1]);
        z = FlipChannels(z, inter, frames);
        z = ResidualCouplingReverse(z, inter, frames, g, w.Flows[0]);
        return z;
    }

    private static float[] FlipChannels(float[] x, int channels, int frames)
    {
        var output = new float[x.Length];
        for (int c = 0; c < channels; c++)
            Array.Copy(x, (channels - 1 - c) * frames, output, c * frames, frames);
        return output;
    }

    private static float[] ResidualCouplingReverse(float[] x, int channels, int frames, float[] g, RvcFlow flow)
    {
        int half = channels / 2;
        var x0 = new float[half * frames];
        var x1 = new float[half * frames];
        Array.Copy(x, 0, x0, 0, half * frames);
        Array.Copy(x, half * frames, x1, 0, half * frames);

        var h = Conv1dK1Bct(x0, half, frames, flow.Pre.Weight, flow.Pre.Bias, RvcSynthesizerWeights.HiddenChannels);
        h = Wn(h, frames, g, flow);
        var mean = Conv1dK1Bct(h, RvcSynthesizerWeights.HiddenChannels, frames, flow.Post.Weight, flow.Post.Bias, half);
        for (int i = 0; i < x1.Length; i++) x1[i] -= mean[i];

        var output = new float[channels * frames];
        Array.Copy(x0, 0, output, 0, half * frames);
        Array.Copy(x1, 0, output, half * frames, half * frames);
        return output;
    }

    /// <summary>Real WaveNet-style dilated gated-conv stack (3 layers, dilation=1, fixed by this
    /// architecture -- confirmed against `build_wn`), conditioned on the speaker embedding via a
    /// single 1x1 "cond" projection sliced per layer.</summary>
    private static float[] Wn(float[] x, int frames, float[] g, RvcFlow flow)
    {
        int hidden = RvcSynthesizerWeights.HiddenChannels;
        int gin = RvcSynthesizerWeights.GinChannels;
        var gBct = Repeat(g, frames); // [gin*frames]
        var cond = Conv1dK1Bct(gBct, gin, frames, flow.CondLayer.Weight, flow.CondLayer.Bias, hidden * 2 * 3);

        var output = new float[hidden * frames];
        var current = x;
        for (int layer = 0; layer < 3; layer++)
        {
            var xIn = Conv1dSamePad(current, hidden, frames, flow.InLayers[layer].Weight, flow.InLayers[layer].Bias, hidden * 2, kernel: 5);
            int gOffset = layer * hidden * 2 * frames;
            var acts = new float[hidden * frames];
            for (int c = 0; c < hidden; c++)
            {
                for (int t = 0; t < frames; t++)
                {
                    float xa = xIn[c * frames + t] + cond[gOffset + c * frames + t];
                    float xb = xIn[(hidden + c) * frames + t] + cond[gOffset + (hidden + c) * frames + t];
                    acts[c * frames + t] = MathF.Tanh(xa) * Sigmoid(xb);
                }
            }
            int resSkipCh = layer < 2 ? hidden * 2 : hidden;
            var resSkip = Conv1dK1Bct(acts, hidden, frames, flow.ResSkipLayers[layer].Weight, flow.ResSkipLayers[layer].Bias, resSkipCh);
            if (layer < 2)
            {
                var next = new float[hidden * frames];
                for (int i = 0; i < hidden * frames; i++) next[i] = current[i] + resSkip[i];
                for (int i = 0; i < hidden * frames; i++) output[i] += resSkip[hidden * frames + i];
                current = next;
            }
            else
            {
                for (int i = 0; i < hidden * frames; i++) output[i] += resSkip[i];
            }
        }
        return output;
    }

    // ---------------------------------------------------------------- generator

    private static float[] BuildGenerator(RvcSynthesizerWeights w, float[] z, int frames, float[] g, ReadOnlySpan<float> sineSource)
    {
        int inter = RvcSynthesizerWeights.InterChannels;
        float[]? sourceBct = null;
        if (w.HasF0)
        {
            // dec.m_source.l_linear: Linear(1,1) applied per sample, then tanh, then transposed to BCT (1 channel).
            var source = new float[sineSource.Length];
            for (int i = 0; i < sineSource.Length; i++)
                source[i] = MathF.Tanh(sineSource[i] * w.SourceLinearWeight![0] + w.SourceLinearBias![0]);
            sourceBct = source;
        }

        var x = Conv1dSamePad(z, inter, frames, w.DecConvPre.Weight, w.DecConvPre.Bias, 512, kernel: 7);
        var cond = Conv1dK1Bct(Repeat(g, frames), RvcSynthesizerWeights.GinChannels, frames, w.DecCond.Weight, w.DecCond.Bias, 512);
        for (int i = 0; i < x.Length; i++) x[i] += cond[i];

        int[] channels = [256, 128, 64, 32];
        int[] inChannels = [512, 256, 128, 64];
        int[] kernels = [3, 7, 11];
        int t = frames;
        int curCh = 512;

        for (int up = 0; up < 4; up++)
        {
            int upRate = w.UpsampleRates[up];
            int upKernel = w.UpsampleKernelSizes[up];
            for (int i = 0; i < x.Length; i++) x[i] = LeakyRelu(x[i]);

            int padding = (upKernel - upRate) / 2;
            var (upsampled, newT) = ConvTranspose1dPyTorch(x, curCh, t, w.DecUps[up].Weight, w.DecUps[up].Bias, channels[up], upKernel, upRate, padding);
            x = upsampled;
            t = newT;
            curCh = channels[up];

            if (w.HasF0)
            {
                int noiseStride = 1;
                for (int next = up + 1; next < 4; next++) noiseStride *= w.UpsampleRates[next];
                int noiseKernel = noiseStride == 1 ? 1 : noiseStride * 2;
                int noisePadding = noiseStride / 2;
                var xSource = Conv1dStrided(sourceBct!, 1, sourceBct!.Length, w.DecNoiseConvs[up]!.Value.Weight, w.DecNoiseConvs[up]!.Value.Bias, curCh, noiseKernel, noiseStride, noisePadding);
                int n = Math.Min(xSource.Length, x.Length);
                for (int i = 0; i < n; i++) x[i] += xSource[i];
            }

            float[]? sum = null;
            for (int kernelIndex = 0; kernelIndex < 3; kernelIndex++)
            {
                var rb = ResBlock1(x, curCh, t, w.DecResBlocks[up * 3 + kernelIndex], kernels[kernelIndex]);
                if (sum is null) sum = rb;
                else for (int i = 0; i < sum.Length; i++) sum[i] += rb[i];
            }
            for (int i = 0; i < sum!.Length; i++) x[i] = sum[i] / 3f;
        }

        for (int i = 0; i < x.Length; i++) x[i] = LeakyRelu(x[i]);
        var post = Conv1dSamePad(x, curCh, t, w.DecConvPostWeight, bias: null, outCh: 1, kernel: 7);
        for (int i = 0; i < post.Length; i++) post[i] = MathF.Tanh(post[i]);
        return post;
    }

    private static float[] ResBlock1(float[] x, int channels, int t, RvcResBlock1 block, int kernel)
    {
        int[] dilations = [1, 3, 5];
        var current = x;
        for (int layer = 0; layer < 3; layer++)
        {
            var xt = new float[current.Length];
            for (int i = 0; i < current.Length; i++) xt[i] = LeakyRelu(current[i]);
            int dilation = dilations[layer];
            int padding = (kernel * dilation - dilation) / 2;
            xt = Conv1dDilatedSamePad(xt, channels, t, block.Convs1[layer].Weight, block.Convs1[layer].Bias, channels, kernel, dilation, padding);
            for (int i = 0; i < xt.Length; i++) xt[i] = LeakyRelu(xt[i]);
            xt = Conv1dSamePad(xt, channels, t, block.Convs2[layer].Weight, block.Convs2[layer].Bias, channels, kernel);
            var next = new float[current.Length];
            for (int i = 0; i < current.Length; i++) next[i] = current[i] + xt[i];
            current = next;
        }
        return current;
    }

    // ---------------------------------------------------------------- shared primitives

    private static float[] SpeakerEmbedding(RvcSynthesizerWeights w, int speakerId)
    {
        int gin = RvcSynthesizerWeights.GinChannels;
        var g = new float[gin];
        Array.Copy(w.EmbGWeight, speakerId * gin, g, 0, gin);
        return g;
    }

    private static float[] Repeat(float[] g, int frames)
    {
        var output = new float[g.Length * frames];
        for (int c = 0; c < g.Length; c++)
        {
            float v = g[c];
            for (int t = 0; t < frames; t++) output[c * frames + t] = v;
        }
        return output;
    }

    private static float[] ToChannelMajor(float[][] btc, int channels, int frames)
    {
        var output = new float[channels * frames];
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < channels; c++)
                output[c * frames + t] = btc[t][c];
        return output;
    }

    private static void AddInPlace(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++) a[i] += b[i];
    }

    private static float[] LayerNormBct(float[] xBct, int channels, int frames, RvcLayerNorm norm)
    {
        var output = new float[xBct.Length];
        for (int t = 0; t < frames; t++)
        {
            double mean = 0;
            for (int c = 0; c < channels; c++) mean += xBct[c * frames + t];
            mean /= channels;
            double variance = 0;
            for (int c = 0; c < channels; c++)
            {
                double d = xBct[c * frames + t] - mean;
                variance += d * d;
            }
            variance /= channels;
            float invStd = (float)(1.0 / Math.Sqrt(variance + LayerNormEps));
            for (int c = 0; c < channels; c++)
            {
                float normed = (float)((xBct[c * frames + t] - mean) * invStd);
                output[c * frames + t] = normed * norm.Gamma[c] + norm.Beta[c];
            }
        }
        return output;
    }

    private static float[] LinearRow(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }

    private static float LeakyRelu(float x) => x >= 0f ? x : x * LReluSlope;
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Conv1d(kernel=1) over a [C][T] channel-major tensor -- equivalent to a per-frame
    /// Linear over channels.</summary>
    private static float[] Conv1dK1Bct(float[] x, int inCh, int frames, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * frames];
        for (int oc = 0; oc < outCh; oc++)
        {
            int wBase = oc * inCh;
            float b = bias[oc];
            for (int t = 0; t < frames; t++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++) sum += weight[wBase + ic] * x[ic * frames + t];
                output[oc * frames + t] = sum;
            }
        }
        return output;
    }

    /// <summary>Real "same"-padding (stride 1, dilation 1) Conv1d over a [C][T] tensor. Weight
    /// real PyTorch layout [outCh, inCh, kernel].</summary>
    private static float[] Conv1dSamePad(float[] x, int inCh, int frames, float[] weight, float[]? bias, int outCh, int kernel)
    {
        int pad = (kernel - 1) / 2;
        var output = new float[outCh * frames];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias is null ? 0f : bias[oc];
            int wBase = oc * inCh * kernel;
            for (int t = 0; t < frames; t++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int it = t - pad + k;
                        if ((uint)it >= (uint)frames) continue;
                        sum += weight[wIcBase + k] * x[ic * frames + it];
                    }
                }
                output[oc * frames + t] = sum;
            }
        }
        return output;
    }

    private static float[] Conv1dDilatedSamePad(float[] x, int inCh, int frames, float[] weight, float[] bias, int outCh, int kernel, int dilation, int padding)
    {
        var output = new float[outCh * frames];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wBase = oc * inCh * kernel;
            for (int t = 0; t < frames; t++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int it = t - padding + k * dilation;
                        if ((uint)it >= (uint)frames) continue;
                        sum += weight[wIcBase + k] * x[ic * frames + it];
                    }
                }
                output[oc * frames + t] = sum;
            }
        }
        return output;
    }

    private static float[] Conv1dStrided(float[] x, int inCh, int inLen, float[] weight, float[] bias, int outCh, int kernel, int stride, int padding)
    {
        int outLen = (inLen + 2 * padding - kernel) / stride + 1;
        var output = new float[outCh * outLen];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wBase = oc * inCh * kernel;
            for (int ot = 0; ot < outLen; ot++)
            {
                float sum = b;
                int start = ot * stride - padding;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int it = start + k;
                        if ((uint)it >= (uint)inLen) continue;
                        sum += weight[wIcBase + k] * x[ic * inLen + it];
                    }
                }
                output[oc * outLen + ot] = sum;
            }
        }
        return output;
    }

    /// <summary>Real PyTorch ConvTranspose1d(stride,kernel,pad=0) followed by trimming `padding`
    /// samples off each end -- matches the reference's `conv_transpose1d_pytorch_padding` exactly
    /// (compute the untrimmed transpose-conv, then slice `[padding, padding+trimmedLen)`).</summary>
    private static (float[] Output, int OutLen) ConvTranspose1dPyTorch(float[] x, int inCh, int inLen, float[] weight, float[] bias, int outCh, int kernel, int stride, int padding)
    {
        int rawLen = (inLen - 1) * stride + kernel;
        var raw = new float[outCh * rawLen];
        for (int oc = 0; oc < outCh; oc++)
            for (int ot = 0; ot < rawLen; ot++) raw[oc * rawLen + ot] = bias[oc];
        for (int ic = 0; ic < inCh; ic++)
        {
            for (int it = 0; it < inLen; it++)
            {
                float v = x[ic * inLen + it];
                if (v == 0f) continue;
                for (int oc = 0; oc < outCh; oc++)
                {
                    int wBase = (ic * outCh + oc) * kernel;
                    int outBase = stride * it;
                    for (int k = 0; k < kernel; k++)
                        raw[oc * rawLen + outBase + k] += v * weight[wBase + k];
                }
            }
        }
        int trimmedLen = rawLen - 2 * padding;
        var output = new float[outCh * trimmedLen];
        for (int oc = 0; oc < outCh; oc++)
            Array.Copy(raw, oc * rawLen + padding, output, oc * trimmedLen, trimmedLen);
        return (output, trimmedLen);
    }
}
