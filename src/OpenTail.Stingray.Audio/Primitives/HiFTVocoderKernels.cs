using System;
using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared math for the HiFT (NSF-source + ISTFTNet HiFiGAN) vocoder family used by both
/// Chatterbox (`Chatterbox/ChatterboxVocoder.cs`, real, golden-verified against PyTorch) and
/// CosyVoice2/3 (CosyVoice's own HiFTGenerator -- S3Gen's HiFT stage was itself derived from
/// CosyVoice's, same lineage). Extracted after both pipelines needed the same conv/activation/
/// STFT primitives and orchestration -- keep this the single source of truth, following the
/// same pattern as `S3GenConformerKernels`/`DenseKernels`.
///
/// Extraction deferred until a second pipeline actually needed this code (CosyVoice2's HiFT
/// weight loader existed with real weights but no forward pass yet) rather than attempted
/// speculatively -- see docs/audio-review-progress.md's CosyVoice section.
/// </summary>
public static class HiFTVocoderKernels
{
    /// <summary>mel is channel-first [MelDim, T]. Returns the waveform samples.</summary>
    public static float[] Generate(IHiFTVocoderWeights w, float[] mel, int t, Random rng, int melDim = 80)
    {
        float[] f0 = PredictF0(w.F0Predictor, mel, t, melDim, w.IsCausal);

        int totalUp = w.IstftHopLen;
        foreach (int r in w.UpsampleRates) totalUp *= r;
        int sampleLen = t * totalUp;
        float[] f0Up = NearestUpsample1D(f0, sampleLen);

        float[] harSource = SineGen(f0Up, sampleLen, w.SampleRate, w.NbHarmonics, rng,
                                    sineAmp: 0.1f, noiseStd: 0.003f, voicedThreshold: 10f);
        float[] excitation = LinearTanhMerge(harSource, sampleLen, w.NbHarmonics + 1, w.MSourceLinearWeight, w.MSourceLinearBias);

        return Decode(w, mel, t, excitation, sampleLen, melDim);
    }

    internal static float[] PredictF0ForTest(IF0PredictorWeights f0w, float[] mel, int t, int melDim) => PredictF0(f0w, mel, t, melDim, isCausal: false);

    private static float[] PredictF0(IF0PredictorWeights f0w, float[] mel, int t, int melDim, bool isCausal)
    {
        var x = mel;
        int inCh = melDim;
        for (int i = 0; i < 5; i++)
        {
            int outCh = f0w.ConvBias[i].Length;
            int kernel = f0w.ConvWeight[i].Length / (outCh * inCh);
            if (isCausal)
            {
                x = i == 0
                    ? CausalConv1dRightPad(x, inCh, t, f0w.ConvWeight[i], f0w.ConvBias[i], outCh, kernel)
                    : CausalConv1dLeftPad(x, inCh, t, f0w.ConvWeight[i], f0w.ConvBias[i], outCh, kernel);
            }
            else
            {
                x = Conv1dSamePad(x, inCh, t, f0w.ConvWeight[i], f0w.ConvBias[i], outCh, kernel);
            }
            EluInPlace(x);
            inCh = outCh;
        }

        var f0 = new float[t];
        float f0Min = float.MaxValue, f0Max = float.MinValue, f0Sum = 0f;
        for (int ti = 0; ti < t; ti++)
        {
            float sum = f0w.ClassifierBias[0];
            for (int c = 0; c < inCh; c++) sum += f0w.ClassifierWeight[c] * x[c * t + ti];
            float v = MathF.Abs(sum);
            f0[ti] = v;
            if (v < f0Min) f0Min = v;
            if (v > f0Max) f0Max = v;
            f0Sum += v;
        }

        // When the checkpoint's F0 classifier predicts normalized pitch in [0, 1] rather than raw Hz,
        // scale by 500Hz to restore the physical fundamental frequency (Hz).
        if (f0Max < 5.0f && f0Max > 0.05f)
        {
            const float f0Scale = 500.0f;
            for (int ti = 0; ti < t; ti++) f0[ti] *= f0Scale;
            f0Min *= f0Scale;
            f0Max *= f0Scale;
            f0Sum *= f0Scale;
        }

        if (Environment.GetEnvironmentVariable("STINGRAY_AUDIO_DIAGNOSTIC_DUMP") == "1")
        {
            Console.WriteLine($"[F0Diag] F0 frames: {t}, min={f0Min:F2}Hz, max={f0Max:F2}Hz, mean={f0Sum / t:F2}Hz");
        }
        return f0;
    }

    private static float[] SineGen(float[] f0Up, int len, int sampleRate, int harmonicNum, Random rng,
                                    float sineAmp, float noiseStd, float voicedThreshold)
    {
        int dim = harmonicNum + 1;
        var sineWaves = new float[dim * len];

        var phaseOffset = new float[dim];
        for (int h = 1; h < dim; h++) phaseOffset[h] = (float)((rng.NextDouble() * 2.0 - 1.0) * Math.PI);

        // Each harmonic's cumulative phase only depends on its own row of f0Up -- independent
        // across h, and this loop touches no shared RNG state (phaseOffset is precomputed above),
        // so parallelizing across harmonics is a pure speedup with no output/ordering change.
        System.Threading.Tasks.Parallel.For(0, dim, h =>
        {
            double harmonicMul = h + 1;
            double freqScale = harmonicMul / sampleRate;
            int row = h * len;
            double cumPhaseH = 0.0;
            for (int n = 0; n < len; n++)
            {
                cumPhaseH = (cumPhaseH + f0Up[n] * freqScale) % 1.0;
                double theta = 2.0 * Math.PI * cumPhaseH;
                sineWaves[row + n] = sineAmp * MathF.Sin((float)theta + phaseOffset[h]);
            }
        });

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

    private static float[] Decode(IHiFTVocoderWeights w, float[] mel, int t, float[] excitation, int sampleLen, int melDim)
    {
        int nFft = w.IstftNFft;
        int hop = w.IstftHopLen;
        int numStages = w.UpsampleRates.Length;
        int numKernels = w.ResblockKernels.Length;

        int stftFrames = sampleLen / hop + 1;
        float[] sStft = RealStft(excitation, sampleLen, nFft, hop, stftFrames);
        int stftCh = (nFft / 2 + 1) * 2;

        var downsampleRates = new int[numStages];
        downsampleRates[0] = 1;
        for (int i = 1; i < numStages; i++) downsampleRates[i] = w.UpsampleRates[numStages - i];
        var cumRates = new int[numStages];
        int acc = 1;
        for (int i = 0; i < numStages; i++) { acc *= downsampleRates[i]; cumRates[i] = acc; }
        Array.Reverse(cumRates);

        float[] x = w.IsCausal
            ? CausalConv1dRightPad(mel, melDim, t, w.ConvPreWeight, w.ConvPreBias, w.BaseChannels, kernel: w.ConvPreKernel)
            : Conv1dSamePad(mel, melDim, t, w.ConvPreWeight, w.ConvPreBias, w.BaseChannels, kernel: w.ConvPreKernel);
        int curCh = w.BaseChannels;
        int curT = t;

        for (int i = 0; i < numStages; i++)
        {
            LeakyReluInPlace(x, 0.1f);

            int chOut = w.BaseChannels >> (i + 1);
            int upK = w.UpsampleKernels[i];
            int upS = w.UpsampleRates[i];
            int upPad = (upK - upS) / 2;
            x = ConvTranspose1d(x, w.UpWeight[i], w.UpBias[i], curCh, chOut, curT, upK, upS, upPad);
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
                si = Conv1dK1(sStft, stftCh, stftFrames, w.SourceDownWeight[i], w.SourceDownBias[i], curCh);
            }
            else
            {
                int kernel = u * 2;
                int pad = u / 2;
                si = Conv1dStrided(sStft, stftCh, stftFrames, w.SourceDownWeight[i], w.SourceDownBias[i], curCh, kernel, u, pad);
            }
            int siT = si.Length / curCh;
            si = AlignTimeLength(si, curCh, siT, curT);

            si = HifiResBlockForward(w.SourceResBlocks[i], si, curCh, curT, w.SourceResblockKernels[i], w.IsCausal);
            for (int j = 0; j < x.Length; j++) x[j] += si[j];

            // The numKernels resblocks all consume the same x independently (different kernel
            // sizes, disjoint weights/outputs) -- previously run sequentially even though each
            // one's own Conv1dDilated calls already parallelize internally over channels. Running
            // them concurrently stacks extra parallelism on top, which matters most at later
            // stages where curCh has shrunk (fewer channels = less internal parallel width to
            // fill available cores with on its own).
            var rbResults = new float[numKernels][];
            int stageIdx = i;
            System.Threading.Tasks.Parallel.For(0, numKernels, j =>
            {
                int rbIdx = stageIdx * numKernels + j;
                rbResults[j] = HifiResBlockForward(w.ResBlocks[rbIdx], x, curCh, curT, w.ResblockKernels[j], w.IsCausal);
            });
            var xs = rbResults[0];
            for (int j = 1; j < numKernels; j++)
            {
                var rbOut = rbResults[j];
                for (int k2 = 0; k2 < xs.Length; k2++) xs[k2] += rbOut[k2];
            }
            float invK = 1f / numKernels;
            for (int j = 0; j < xs.Length; j++) xs[j] *= invK;
            x = xs;
        }

        LeakyReluInPlace(x, 0.01f);
        x = w.IsCausal
            ? CausalConv1dLeftPad(x, curCh, curT, w.ConvPostWeight, w.ConvPostBias, nFft + 2, kernel: w.ConvPostKernel)
            : Conv1dSamePad(x, curCh, curT, w.ConvPostWeight, w.ConvPostBias, nFft + 2, kernel: w.ConvPostKernel);
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

        var wav = InverseStft(spec, phase, specCh, outT, nFft, hop, sampleLen);
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

    private static float[] HifiResBlockForward(IHifiResBlockWeights rw, float[] x, int ch, int t, int kernel, bool causal = false)
    {
        var cur = (float[])x.Clone();
        int[] dilations = [1, 3, 5];
        for (int i = 0; i < 3; i++)
        {
            var xt = (float[])cur.Clone();
            SnakeInPlace(xt, rw.Alpha1[i], ch, t);
            xt = causal
                ? CausalConv1dDilatedLeftPad(xt, rw.Convs1Weight[i], rw.Convs1Bias[i], ch, ch, t, kernel, dilations[i])
                : Conv1dDilated(xt, rw.Convs1Weight[i], rw.Convs1Bias[i], ch, ch, t, kernel, dilations[i]);

            SnakeInPlace(xt, rw.Alpha2[i], ch, t);
            xt = causal
                ? CausalConv1dDilatedLeftPad(xt, rw.Convs2Weight[i], rw.Convs2Bias[i], ch, ch, t, kernel, dilation: 1)
                : Conv1dDilated(xt, rw.Convs2Weight[i], rw.Convs2Bias[i], ch, ch, t, kernel, dilation: 1);

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

    private static float[] RealStft(float[] signal, int sigLen, int nFft, int hop, int frames)
    {
        int specBins = nFft / 2 + 1;
        int outCh = specBins * 2;
        var output = new float[outCh * frames];
        int pad = nFft / 2;

        var window = HannWindow(nFft);
        System.Threading.Tasks.Parallel.For(0, frames, f =>
        {
            int start = f * hop - pad;
            for (int k = 0; k < specBins; k++)
            {
                float real = 0f, imag = 0f;
                float step = -2f * MathF.PI * k / nFft;
                for (int n = 0; n < nFft; n++)
                {
                    int idx = start + n;
                    float s;
                    if (idx < 0) s = signal[Math.Min(-idx, sigLen - 1)];
                    else if (idx >= sigLen) s = signal[Math.Max(0, 2 * sigLen - 2 - idx)];
                    else s = signal[idx];
                    s *= window[n];
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

    private static float[] InverseStft(float[] spec, float[] phase, int specBins, int frames, int nFft, int hop, int targetLen)
    {
        int rawLen = (frames - 1) * hop + nFft;
        var full = new float[rawLen];
        var norm = new float[rawLen];
        var window = HannWindow(nFft);

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
                    float term = rC * MathF.Cos(angle) - iC * MathF.Sin(angle);
                    val += (k > 0 && k < specBins - 1) ? 2f * term : term;
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
                if (si < rawLen)
                {
                    full[si] += local[n];
                    norm[si] += window[n] * window[n];
                }
            }
        }
        for (int i = 0; i < rawLen; i++)
            if (norm[i] > 1e-8f) full[i] /= norm[i];

        int pad = nFft / 2;
        var output = new float[targetLen];
        for (int i = 0; i < targetLen; i++)
        {
            int src = pad + i;
            output[i] = (src < rawLen) ? full[src] : 0f;
        }
        return output;
    }

    private static float[] HannWindow(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / n));
        return w;
    }

    /// <summary>Real causal Conv1d, left-zero-pad by (kernel-1) then valid conv (output length == input length). Channel-first [inCh,T] flat layout, weight [outCh,inCh,kernel].</summary>
    private static float[] CausalConv1dLeftPad(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
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

    /// <summary>Real causal Conv1d, right-zero-pad by (kernel-1) then valid conv (output length == input length). Same layout as <see cref="CausalConv1dLeftPad"/>.</summary>
    private static float[] CausalConv1dRightPad(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
    {
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
                    int shift = k; // no left pad; taps k=0..kernel-1 read input[ti..ti+kernel-1], right edge implicitly zero via AxpyShifted's clamped range
                    AxpyShifted(inRow, weight[wBase + k], outRow, shift, t);
                }
            }
            Array.Copy(outRow, 0, output, oc * t, t);
        });
        return output;
    }

    /// <summary>Real causal dilated Conv1d, left-zero-pad by (kernel-1)*dilation then valid conv (output length == input length) -- CosyVoice's real `ResBlock` convention (confirmed `causal_type=left` for every dilation in `examples/cosyvoice.cpp`'s `ResBlock::OnLoad`), distinct from Chatterbox's real symmetric `get_padding` formula used by <see cref="Conv1dDilated"/>.</summary>
    private static float[] CausalConv1dDilatedLeftPad(float[] input, float[] weight, float[] bias, int inCh, int outCh, int t, int kernel, int dilation)
    {
        int pad = (kernel - 1) * dilation;
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

public interface IHifiResBlockWeights
{
    float[][] Convs1Weight { get; }
    float[][] Convs1Bias { get; }
    float[][] Convs2Weight { get; }
    float[][] Convs2Bias { get; }
    float[][] Alpha1 { get; }
    float[][] Alpha2 { get; }
}

public interface IF0PredictorWeights
{
    float[][] ConvWeight { get; }
    float[][] ConvBias { get; }
    float[] ClassifierWeight { get; }
    float[] ClassifierBias { get; }
}

public interface IHiFTVocoderWeights
{
    int[] UpsampleRates { get; }
    int[] UpsampleKernels { get; }
    int[] ResblockKernels { get; }
    int[] SourceResblockKernels { get; }
    int BaseChannels { get; }
    int NbHarmonics { get; }
    int IstftNFft { get; }
    int IstftHopLen { get; }
    int SampleRate { get; }
    int ConvPreKernel { get; }
    int ConvPostKernel { get; }
    float[] ConvPreWeight { get; }
    float[] ConvPreBias { get; }
    float[] ConvPostWeight { get; }
    float[] ConvPostBias { get; }
    float[][] UpWeight { get; }
    float[][] UpBias { get; }
    float[][] SourceDownWeight { get; }
    float[][] SourceDownBias { get; }
    IHifiResBlockWeights[] SourceResBlocks { get; }
    IHifiResBlockWeights[] ResBlocks { get; }
    IF0PredictorWeights F0Predictor { get; }
    float[] MSourceLinearWeight { get; }
    float[] MSourceLinearBias { get; }

    /// <summary>
    /// Whether `conv_pre`/`conv_post`/the resblock dilated convs use CosyVoice's real one-sided
    /// causal padding (`conv_pre`=right-pad, `conv_post`=left-pad, resblock convs=left-pad --
    /// confirmed via `examples/cosyvoice.cpp`'s `CausalHiFTGenerator::OnLoad`/`ResBlock::OnLoad`)
    /// rather than Chatterbox's real plain symmetric "same" padding (confirmed via
    /// `examples/chatterbox-tts-py/chatterbox/models/s3gen/hifigan.py`'s real `Conv1d(...,
    /// padding=3)`/`get_padding` -- genuinely non-causal, a different real HiFiGAN variant
    /// despite the shared lineage). Defaults to <c>false</c> (Chatterbox's real convention,
    /// this shared kernel's original and still-correct target) so existing implementers don't
    /// need to change; CosyVoice's real weight classes override this to <c>true</c>.
    /// </summary>
    bool IsCausal => false;
}
