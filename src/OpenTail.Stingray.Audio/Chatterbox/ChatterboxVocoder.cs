using System;
using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen stage 3: HiFTGenerator (examples/chatterbox-tts-py/chatterbox/models/s3gen/hifigan.py) --
/// Neural Source Filter + ISTFTNet vocoder turning a mel-spectrogram into a 24kHz waveform.
/// Structurally very close to Kokoro's Decoder/Generator (KokoroDecoder.cs): NSF harmonic sine
/// source, learned inverse-STFT head, Snake-activated HiFiGAN resblocks with per-(stage,kernel)
/// weights -- KokoroDecoder.cs's AdaINResBlock1 kernel-size bug (each resblock in a HiFiGAN-style
/// multi-receptive-field stack needs its OWN kernel size, not a hardcoded one) is deliberately
/// avoided here from the start by threading VocResblockKernels/VocSourceResblockKernels through.
/// Differences from Kokoro's NSF source: the random per-harmonic phase here is a single additive
/// constant (sampled once per call, broadcast across all timesteps) rather than injected only at
/// t=0 into an accumulating phase -- see SineGen in hifigan.py.
/// </summary>
public static class ChatterboxVocoder
{
    /// <summary>mel is channel-first [80, T]. Returns the waveform samples.</summary>
    public static float[] Generate(ChatterboxS3GenWeights w, float[] mel, int t, Random rng)
    {
        float[] f0 = PredictF0(w.F0Predictor, mel, t);

        int totalUp = w.VocIstftHopLen;
        foreach (int r in w.VocUpsampleRates) totalUp *= r;
        int sampleLen = t * totalUp;
        float[] f0Up = NearestUpsample1D(f0, sampleLen);

        float[] harSource = SineGen(f0Up, sampleLen, w.SampleRate, w.VocNbHarmonics, rng,
                                    sineAmp: 0.1f, noiseStd: 0.003f, voicedThreshold: 10f);
        float[] excitation = LinearTanhMerge(harSource, sampleLen, w.VocNbHarmonics + 1, w.Vocoder.MSourceLinearWeight, w.Vocoder.MSourceLinearBias);

        return Decode(w, mel, t, excitation, sampleLen);
    }

    // -----------------------------------------------------------------------
    // F0 prediction (ConvRNNF0Predictor)
    // -----------------------------------------------------------------------

    private static float[] PredictF0(ChatterboxF0PredictorWeights f0w, float[] mel, int t)
    {
        var x = mel;
        int inCh = 80;
        for (int i = 0; i < 5; i++)
        {
            int outCh = f0w.ConvBias[i].Length;
            x = Conv1dSamePad(x, inCh, t, f0w.ConvWeight[i], f0w.ConvBias[i], outCh, kernel: 3);
            EluInPlace(x);
            inCh = outCh;
        }

        var f0 = new float[t];
        for (int ti = 0; ti < t; ti++)
        {
            float sum = f0w.ClassifierBias[0];
            for (int c = 0; c < inCh; c++) sum += f0w.ClassifierWeight[c] * x[c * t + ti];
            f0[ti] = MathF.Abs(sum);
        }
        return f0;
    }

    // -----------------------------------------------------------------------
    // NSF harmonic sine source (hifigan.py's SineGen / SourceModuleHnNSF)
    // -----------------------------------------------------------------------

    private static float[] SineGen(float[] f0Up, int len, int sampleRate, int harmonicNum, Random rng,
                                    float sineAmp, float noiseStd, float voicedThreshold)
    {
        int dim = harmonicNum + 1;
        var sineWaves = new float[dim * len]; // channel-first [dim, len], matches theta_mat's [B,dim,len]

        // phase_vec: one Uniform(-pi,pi) draw per harmonic, added as a constant bias across all of
        // time (NOT accumulated into the running phase) -- harmonic 0 (fundamental) gets no offset.
        var phaseOffset = new float[dim];
        for (int h = 1; h < dim; h++) phaseOffset[h] = (float)((rng.NextDouble() * 2.0 - 1.0) * Math.PI);

        var cumPhase = new double[dim];
        for (int h = 0; h < dim; h++)
        {
            double harmonicMul = h + 1;
            double freqScale = harmonicMul / sampleRate;
            int row = h * len;
            for (int n = 0; n < len; n++)
            {
                cumPhase[h] = (cumPhase[h] + f0Up[n] * freqScale) % 1.0;
                double theta = 2.0 * Math.PI * cumPhase[h];
                sineWaves[row + n] = sineAmp * MathF.Sin((float)theta + phaseOffset[h]);
            }
        }

        for (int n = 0; n < len; n++)
        {
            bool voiced = f0Up[n] > voicedThreshold;
            float uv = voiced ? 1f : 0f;
            float noiseAmp = uv * noiseStd + (1f - uv) * sineAmp / 3f;
            for (int h = 0; h < dim; h++)
            {
                int idx = h * len + n;
                float u1 = MathF.Max(1e-7f, (float)rng.NextDouble());
                float u2 = (float)rng.NextDouble();
                float noise = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2) * noiseAmp;
                sineWaves[idx] = sineWaves[idx] * uv + noise;
            }
        }
        return sineWaves;
    }

    /// <summary>m_source.l_linear + tanh: merges `dim` harmonic channels into one excitation channel per sample.</summary>
    private static float[] LinearTanhMerge(float[] sines, int len, int dim, float[] weight, float[] bias)
    {
        var output = new float[len];
        float b = bias[0];
        for (int n = 0; n < len; n++)
        {
            float sum = b;
            for (int h = 0; h < dim; h++) sum += weight[h] * sines[h * len + n];
            output[n] = MathF.Tanh(sum);
        }
        return output;
    }

    // -----------------------------------------------------------------------
    // HiFTGenerator.decode
    // -----------------------------------------------------------------------

    private static float[] Decode(ChatterboxS3GenWeights w, float[] mel, int t, float[] excitation, int sampleLen)
    {
        var vw = w.Vocoder;
        int nFft = w.VocIstftNFft;
        int hop = w.VocIstftHopLen;
        int numStages = w.VocUpsampleRates.Length;
        int numKernels = w.VocResblockKernels.Length;

        int stftFrames = Math.Max(1, (sampleLen - nFft) / hop + 1);
        float[] sStft = RealStft(excitation, sampleLen, nFft, hop, stftFrames); // [ (nFft/2+1)*2, stftFrames ]
        int stftCh = (nFft / 2 + 1) * 2;

        // downsample_cum_rates = cumprod([1] + reverse(upsampleRates)[:-1]), reversed for stage order.
        var downsampleRates = new int[numStages];
        downsampleRates[0] = 1;
        for (int i = 1; i < numStages; i++) downsampleRates[i] = w.VocUpsampleRates[numStages - i];
        var cumRates = new int[numStages];
        int acc = 1;
        for (int i = 0; i < numStages; i++) { acc *= downsampleRates[i]; cumRates[i] = acc; }
        Array.Reverse(cumRates); // stage-order u values

        float[] x = Conv1dSamePad(mel, 80, t, vw.ConvPreWeight, vw.ConvPreBias, w.VocBaseChannels, kernel: 7);
        int curCh = w.VocBaseChannels;
        int curT = t;

        for (int i = 0; i < numStages; i++)
        {
            LeakyReluInPlace(x, 0.1f);

            int chOut = w.VocBaseChannels >> (i + 1);
            int upK = w.VocUpsampleKernels[i];
            int upS = w.VocUpsampleRates[i];
            int upPad = (upK - upS) / 2;
            x = ConvTranspose1d(x, vw.UpWeight[i], vw.UpBias[i], curCh, chOut, curT, upK, upS, upPad);
            curT = x.Length / chOut;
            curCh = chOut;

            if (i == numStages - 1)
            {
                x = ReflectionPadLeft1(x, curCh, curT);
                curT += 1;
            }

            int u = cumRates[i];
            float[] si;
            if (u == 1)
            {
                si = Conv1dK1(sStft, stftCh, stftFrames, vw.SourceDownWeight[i], vw.SourceDownBias[i], curCh);
            }
            else
            {
                int kernel = u * 2;
                int pad = u / 2;
                si = Conv1dStrided(sStft, stftCh, stftFrames, vw.SourceDownWeight[i], vw.SourceDownBias[i], curCh, kernel, u, pad);
            }
            int siT = si.Length / curCh;
            si = AlignTimeLength(si, curCh, siT, curT);

            si = HifiResBlockForward(vw.SourceResBlocks[i], si, curCh, curT, w.VocSourceResblockKernels[i]);
            for (int j = 0; j < x.Length; j++) x[j] += si[j];

            float[]? xs = null;
            for (int j = 0; j < numKernels; j++)
            {
                int rbIdx = i * numKernels + j;
                var rbOut = HifiResBlockForward(vw.ResBlocks[rbIdx], x, curCh, curT, w.VocResblockKernels[j]);
                if (xs == null) xs = rbOut;
                else for (int k2 = 0; k2 < xs.Length; k2++) xs[k2] += rbOut[k2];
            }
            float invK = 1f / numKernels;
            for (int j = 0; j < xs!.Length; j++) xs[j] *= invK;
            x = xs;
        }

        LeakyReluInPlace(x, 0.01f);
        x = Conv1dSamePad(x, curCh, curT, vw.ConvPostWeight, vw.ConvPostBias, nFft + 2, kernel: 7);
        int outT = x.Length / (nFft + 2);

        int specCh = nFft / 2 + 1;
        var spec = new float[specCh * outT];
        var phase = new float[specCh * outT];
        for (int c = 0; c < specCh; c++)
            for (int ti = 0; ti < outT; ti++)
                spec[c * outT + ti] = MathF.Min(MathF.Exp(x[c * outT + ti]), 1e2f);
        for (int c = 0; c < specCh; c++)
            for (int ti = 0; ti < outT; ti++)
                phase[c * outT + ti] = MathF.Sin(x[(specCh + c) * outT + ti]);

        var wav = InverseStft(spec, phase, specCh, outT, nFft, hop);
        for (int i = 0; i < wav.Length; i++) wav[i] = Math.Clamp(wav[i], -0.99f, 0.99f);
        return wav;
    }

    private static float[] AlignTimeLength(float[] input, int ch, int inT, int targetT)
    {
        if (inT == targetT) return input;
        var output = new float[ch * targetT];
        int copyT = Math.Min(inT, targetT);
        for (int c = 0; c < ch; c++)
            Array.Copy(input, c * inT, output, c * targetT, copyT);
        return output;
    }

    // -----------------------------------------------------------------------
    // HiFiGAN ResBlock (Snake-activated, per-kernel-size dilated conv pairs)
    // -----------------------------------------------------------------------

    /// <summary>Returns a fresh array; does not mutate <paramref name="x"/> (the resblock-averaging
    /// callers in Decode() invoke this multiple times against the same input and need each call's
    /// result independent).</summary>
    private static float[] HifiResBlockForward(ChatterboxHifiResBlockWeights rw, float[] x, int ch, int t, int kernel)
    {
        var cur = (float[])x.Clone();
        int[] dilations = [1, 3, 5];
        for (int i = 0; i < 3; i++)
        {
            var xt = (float[])cur.Clone();
            SnakeInPlace(xt, rw.Alpha1[i], ch, t);
            xt = Conv1dDilated(xt, rw.Convs1Weight[i], rw.Convs1Bias[i], ch, ch, t, kernel, dilations[i]);

            SnakeInPlace(xt, rw.Alpha2[i], ch, t);
            xt = Conv1dDilated(xt, rw.Convs2Weight[i], rw.Convs2Bias[i], ch, ch, t, kernel, dilation: 1);

            for (int j = 0; j < cur.Length; j++) cur[j] += xt[j];
        }
        return cur;
    }

    private static void SnakeInPlace(float[] x, float[] alpha, int ch, int t)
    {
        for (int c = 0; c < ch; c++)
        {
            float a = alpha[c];
            float invA = MathF.Abs(a) > 1e-9f ? 1f / a : 0f;
            int row = c * t;
            for (int ti = 0; ti < t; ti++)
            {
                float v = x[row + ti];
                float sv = MathF.Sin(a * v);
                x[row + ti] = v + invA * sv * sv;
            }
        }
    }

    // -----------------------------------------------------------------------
    // STFT / iSTFT (real STFT for the excitation source; learned inverse-STFT head for the output)
    // -----------------------------------------------------------------------

    private static float[] RealStft(float[] signal, int sigLen, int nFft, int hop, int frames)
    {
        int specBins = nFft / 2 + 1;
        int outCh = specBins * 2; // [real; imag]
        var output = new float[outCh * frames];

        var window = HannWindow(nFft);
        // Each frame writes disjoint (k,f) columns -- safe to parallelize over frames. nFft is
        // tiny (16) but `frames` scales with the full sample count (hop=4), so this is the right
        // axis to parallelize.
        System.Threading.Tasks.Parallel.For(0, frames, f =>
        {
            int start = f * hop;
            for (int k = 0; k < specBins; k++)
            {
                float real = 0f, imag = 0f;
                float step = -2f * MathF.PI * k / nFft;
                for (int n = 0; n < nFft; n++)
                {
                    float s = (start + n < sigLen ? signal[start + n] : 0f) * window[n];
                    float angle = step * n;
                    real += s * MathF.Cos(angle);
                    imag += s * MathF.Sin(angle);
                }
                output[k * frames + f] = real;
                output[(specBins + k) * frames + f] = imag;
            }
        });
        return output;
    }

    private static float[] InverseStft(float[] spec, float[] phase, int specBins, int frames, int nFft, int hop)
    {
        int outputLen = (frames - 1) * hop + nFft;
        var output = new float[outputLen];
        var norm = new float[outputLen];
        var window = HannWindow(nFft);

        // Overlap-add accumulation can't be parallelized directly (adjacent frames' windows
        // overlap since hop < nFft, so concurrent += would race). Instead, compute each frame's
        // small (nFft-length) windowed contribution in parallel -- the actual expensive part,
        // since it's an O(nFft*specBins) trig-heavy inner loop repeated over every frame, and
        // `frames` scales with the full sample count -- then accumulate sequentially, which is
        // cheap (O(frames*nFft) additions, no trig).
        var frameContrib = new float[frames][];
        System.Threading.Tasks.Parallel.For(0, frames, f =>
        {
            var local = new float[nFft];
            for (int n = 0; n < nFft; n++)
            {
                float val = 0f;
                for (int k = 0; k < specBins; k++)
                {
                    float mag = spec[k * frames + f];
                    float ph = phase[k * frames + f];
                    float rC = mag * MathF.Cos(ph);
                    float iC = mag * MathF.Sin(ph);
                    float angle = 2f * MathF.PI * k * n / nFft;
                    val += rC * MathF.Cos(angle) - iC * MathF.Sin(angle);
                    if (k > 0 && k < specBins - 1)
                        val += rC * MathF.Cos(angle) + iC * MathF.Sin(angle);
                }
                local[n] = val / nFft * window[n];
            }
            frameContrib[f] = local;
        });

        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            var local = frameContrib[f];
            for (int n = 0; n < nFft; n++)
            {
                int si = start + n;
                if (si < outputLen)
                {
                    output[si] += local[n];
                    norm[si] += window[n] * window[n];
                }
            }
        }
        for (int i = 0; i < outputLen; i++)
            if (norm[i] > 1e-8f) output[i] /= norm[i];
        return output;
    }

    private static float[] HannWindow(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / n));
        return w;
    }

    // -----------------------------------------------------------------------
    // Conv / activation primitives
    // -----------------------------------------------------------------------

    // NOTE: all conv helpers below parallelize over output channels (Parallel.For(0, outCh, ...))
    // -- each output channel is computed independently, and by the later HiFTGenerator upsample
    // stages `t` approaches the full 24kHz sample count, so these convs (particularly the
    // dilated HiFiGAN resblock ones, kernel up to 11) are a real cost, not a rounding error.

    /// <summary>
    /// Scale-and-shift-add formulation: for a fixed (oc,ic,k), `output[ti] += w*input[ti+shift]`
    /// over the valid ti range is a single vectorizable TensorPrimitives.MultiplyAdd over a
    /// contiguous span, instead of the ti-outer/ic-middle/k-inner scalar loop's strided
    /// per-element reads. Same reordering insight as the attention weighted-sum fix (see
    /// WhisperEncoder.cs/ChatterboxCfmDecoder.cs's SelfAttention) applied to convolution --
    /// this is the dominant remaining cost in the vocoder (dilated HiFiGAN resblock convs running
    /// at up to ~35k positions in the later upsample stages), so it's worth the loop-order
    /// restructuring rather than just parallelizing over channels.
    /// </summary>
    private static float[] Conv1dSamePad(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
    {
        int pad = kernel / 2;
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

    private static float[] Conv1dDilated(float[] input, float[] weight, float[] bias, int inCh, int outCh, int t, int kernel, int dilation)
    {
        int pad = (kernel * dilation - dilation) / 2;
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
                    int shift = k * dilation - pad;
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

    /// <summary>output[ti] += scale * input[ti + shift] for every ti where ti+shift is in range;
    /// a single vectorized call over the valid contiguous overlap instead of a per-element
    /// bounds-checked scalar loop.</summary>
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

    private static float[] Conv1dStrided(float[] input, int inCh, int inT, float[] weight, float[] bias, int outCh, int kernel, int stride, int padding)
    {
        int outT = Math.Max(1, (inT + 2 * padding - kernel) / stride + 1);
        var output = new float[outCh * outT];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wOcBase = oc * inCh * kernel;
            for (int ot = 0; ot < outT; ot++)
            {
                float sum = b;
                int center = ot * stride - padding;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wBase = wOcBase + ic * kernel;
                    int srcBase = ic * inT;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = center + k;
                        if ((uint)src < (uint)inT) sum += weight[wBase + k] * input[srcBase + src];
                    }
                }
                output[oc * outT + ot] = sum;
            }
        });
        return output;
    }

    // ConvTranspose1d weight layout: torch [inCh, outCh, kernel]. Loop order is oc-outer (each
    // output channel reads from all input channels/positions) so this parallelizes the same way
    // as the other conv helpers, unlike a naive ic-outer scatter which would need per-output
    // locking or atomics.
    private static float[] ConvTranspose1d(float[] input, float[] weight, float[] bias, int inCh, int outCh, int inT, int kernel, int stride, int padding)
    {
        int outT = (inT - 1) * stride - 2 * padding + kernel;
        var output = new float[outCh * outT];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int dstBase = oc * outT;
            for (int ti = 0; ti < outT; ti++) output[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * inT;
                int wBase0 = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < inT; ti++)
                {
                    float v = input[srcBase + ti];
                    int outStart = ti * stride - padding;
                    for (int k = 0; k < kernel; k++)
                    {
                        int to = outStart + k;
                        if ((uint)to < (uint)outT) output[dstBase + to] += v * weight[wBase0 + k];
                    }
                }
            }
        });
        return output;
    }

    private static float[] ReflectionPadLeft1(float[] input, int ch, int t)
    {
        int newT = t + 1;
        var output = new float[ch * newT];
        for (int c = 0; c < ch; c++)
        {
            int srcBase = c * t;
            int dstBase = c * newT;
            output[dstBase] = input[srcBase + Math.Min(1, t - 1)];
            Array.Copy(input, srcBase, output, dstBase + 1, t);
        }
        return output;
    }

    private static float[] NearestUpsample1D(float[] input, int targetLen)
    {
        int inLen = input.Length;
        var output = new float[targetLen];
        for (int i = 0; i < targetLen; i++)
        {
            int src = (int)((long)i * inLen / targetLen);
            output[i] = input[Math.Min(src, inLen - 1)];
        }
        return output;
    }

    private static void LeakyReluInPlace(float[] x, float slope)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0f) x[i] *= slope;
    }

    private static void EluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0f) x[i] = MathF.Exp(x[i]) - 1f;
    }
}
