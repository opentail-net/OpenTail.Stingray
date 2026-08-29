
namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// istftnet.py Decoder.forward + Generator.forward, ported directly from
/// examples/kokoro-py/istftnet.py. Uses real weights from <see cref="DecoderWeights"/>
/// and <see cref="GeneratorWeights"/> (both already loaded by <see cref="KokoroWeights"/>).
///
/// Decoder.forward(asr, F0_curve, N, s_dec):
///   F0 = F0_conv(F0_curve)   [1,2T] stride-2 -> [1,T]
///   N  = N_conv(N_curve)     same
///   x  = cat([asr,F0,N])     [514,T] -> encode AdainResBlk1d(514->1024)
///   asr_res = asr_res(asr)   [512->64, T]
///   decode[0..3]: while res==True, x = cat([x, asr_res, F0, N]) -> block(x, s)
///     res flips False after the upsampling block (block 3).
///   -> generator(x, s_dec, F0_curve)  [original 2x-rate F0_curve]
///
/// Generator.forward(x, s, f0):
///   f0 nearest-upsample by prod(upsample_rates)*hop_size (=300) ->
///   SineGen cumulative-phase 9-harmonic source -> Linear(9->1)+tanh ->
///   STFT(nFft=20, hop=5) -> har [22, stftFrames]
///   for each of 2 upsample stages:
///     LeakyReLU(x,0.1) -> noise_convs[i](har) -> noise_res[i](AdaINResBlock1) ->
///     ups[i](ConvTranspose1d) -> (last stage: ReflectionPad left 1) ->
///     x += noise -> average 3 resblocks (AdaIN+Snake1D)
///   LeakyReLU(x, 0.01) -> conv_post(128->22, k=7) ->
///   spec=exp(x[:11]), phase=sin(x[11:]) -> iSTFT overlap-add
/// </summary>
public static class KokoroDecoder
{
    // -----------------------------------------------------------------------
    // Decoder.forward
    // -----------------------------------------------------------------------

    /// <summary>
    /// asr is channel-first [512, T]. f0Curve and nCurve are channel-first [1, 2T].
    /// sDec is ref_s[:128]. Returns raw waveform samples.
    /// </summary>
    public static float[] Forward(
        DecoderWeights dw,
        KokoroWeights kw,
        float[] asr,      // [512, T]
        float[] f0Curve,  // [1, 2T]  (output of F0Ntrain, 2x frame rate)
        float[] nCurve,   // [1, 2T]
        float[] sDec,     // [128]
        int t)            // T
    {
        int styleDim = kw.StyleDim; // 128

        // F0_conv / N_conv: Conv1d(1,1, k=3, stride=2, padding=1)  [1,2T] -> [1,T]
        int t2 = t * 2;
        float[] f0Down = Conv1dStride2(f0Curve, dw.F0ConvWeight, dw.F0ConvBias, inT: t2);
        float[] nDown  = Conv1dStride2(nCurve,  dw.NConvWeight,  dw.NConvBias,  inT: t2);

        // cat([asr, f0Down, nDown]) -> [514, T]
        float[] catIn = Cat3ChannelFirst(asr, 512, f0Down, 1, nDown, 1, t);

        // encode: AdainResBlk1d(514 -> 1024)
        float[] x = KokoroAdainResBlk1d.Forward(dw.Encode, catIn, dimIn: 514, dimOut: 1024, t: t, style: sDec, styleDim: styleDim);

        // asr_res: Conv1d(512->64, k=1)
        float[] asrRes = Conv1dK1(asr, dw.AsrResWeight, dw.AsrResBias, inCh: 512, outCh: 64, inT: t);

        // 4 decode blocks. In the reference, res starts True and flips False after the upsample
        // block (block 3, the only one with PoolWeight). While res=True, concat [x, asrRes, F0, N].
        bool res = true;
        int curT = t;
        for (int i = 0; i < DecoderWeights.DecodeStackSize; i++)
        {
            var blk = dw.Decode[i];
            bool hasUpsample = blk.PoolWeight is not null;

            int dimIn;
            if (res)
            {
                // cat([x(1024), asrRes(64), f0Down(1), nDown(1)]) = 1090
                x = Cat4ChannelFirst(x, 1024, asrRes, 64, f0Down, 1, nDown, 1, curT);
                dimIn = 1090;
            }
            else
            {
                dimIn = 1024;
            }

            int dimOut = (i < DecoderWeights.DecodeStackSize - 1) ? 1024 : 512;
            x = KokoroAdainResBlk1d.Forward(blk, x, dimIn: dimIn, dimOut: dimOut, t: curT, style: sDec, styleDim: styleDim);

            if (hasUpsample)
            {
                curT *= 2;
                res = false;
            }
        }

        // Generator
        return GeneratorForward(dw.Generator, kw, x, sDec, f0Curve, curT, styleDim);
    }

    // -----------------------------------------------------------------------
    // Generator.forward
    // -----------------------------------------------------------------------

    private static float[] GeneratorForward(
        GeneratorWeights gw,
        KokoroWeights kw,
        float[] x,       // [512, curT]
        float[] s,       // [128]
        float[] f0Curve, // [1, 2T] original 2x-rate F0
        int curT,
        int styleDim)
    {
        int[] upsampleRates   = kw.UpsampleRates;        // [10, 6]
        int[] upsampleKernels = kw.UpsampleKernelSizes;  // [20, 12]
        int[] resblockKernels = kw.ResblockKernelSizes;  // [3, 7, 11]
        int[] resblockDils    = kw.ResblockDilations;    // [1,3,5, 1,3,5, 1,3,5]
        int genNFft  = kw.IstftNFft;  // 20
        int genHop   = kw.IstftHopSize; // 5
        int numUp    = upsampleRates.Length;   // 2
        int numKernels = resblockKernels.Length; // 3

        // Upsample F0 by prod(upsampleRates)*genHop = 300 (nearest)
        int totalUp = genHop;
        foreach (int r in upsampleRates) totalUp *= r; // 300

        int f0Len   = f0Curve.Length; // 2T
        int f0UpLen = f0Len * totalUp;
        float[] f0Up = NearestUpsample1D(f0Curve, f0UpLen);

        // SineGen: 9-harmonic cumulative-phase sine source
        float[] harSource = SineGen(f0Up, f0UpLen, kw.SampleRate, harmonicNum: 8,
                                    sineAmp: 0.1f, noiseStd: 0.003f, voicedThreshold: 10f);

        // SourceModuleHnNSF: Linear(9->1) + Tanh
        float[] harMerged = LinearTanh9to1(harSource, f0UpLen, gw.MSourceWeight, gw.MSourceBias);

        // STFT of harMerged: TorchSTFT(filter=genNFft, hop=genHop, win=genNFft)
        int stftFrames = (f0UpLen - genNFft) / genHop + 1;
        if (stftFrames <= 0) stftFrames = 1;
        float[] har = ComputeHarSTFT(harMerged, f0UpLen, genNFft, genHop, stftFrames);
        // har [genNFft+2=22, stftFrames] channel-first: [:11]=mag, [11:]=phase

        int initialChannel = 512; // upsample_initial_channel

        for (int i = 0; i < numUp; i++)
        {
            int chIn  = initialChannel >> i;       // 512, 256
            int chOut = initialChannel >> (i + 1); // 256, 128

            // LeakyReLU(x, 0.1)
            LeakyReluInPlace(x, LeakySlope01);

            // noise_convs[i](har)
            float[] xSource;
            int xSourceT;
            if (i + 1 < numUp)
            {
                int strideFf0 = 1;
                for (int k = i + 1; k < numUp; k++) strideFf0 *= upsampleRates[k]; // 6
                int noiseKernel = strideFf0 * 2; // 12
                int noisePad    = (strideFf0 + 1) / 2; // 3
                xSource  = Conv1dStrided(har, genNFft + 2, stftFrames, gw.NoiseConvWeight[i], gw.NoiseConvBias[i], chOut, noiseKernel, strideFf0, noisePad);
                xSourceT = xSource.Length / chOut;
            }
            else
            {
                xSource  = Conv1dK1(har, gw.NoiseConvWeight[i], gw.NoiseConvBias[i], genNFft + 2, chOut, stftFrames);
                xSourceT = stftFrames;
            }

            // noise_res[i]: AdaINResBlock1 (Snake1D activated). Reference (istftnet.py Generator.__init__):
            // kernel_size=7 for every stage except the last, which uses kernel_size=11.
            int noiseResKernel = (i + 1 < numUp) ? 7 : 11;
            xSource  = AdaINResBlock1Forward(gw.NoiseRes[i], xSource, chOut, xSourceT, s, styleDim, noiseResKernel);
            xSourceT = xSource.Length / chOut;

            // ups[i]: ConvTranspose1d(chIn, chOut, k=upsampleKernels[i], stride=upsampleRates[i], pad=(k-u)/2)
            int upK   = upsampleKernels[i];
            int upS   = upsampleRates[i];
            int upPad = (upK - upS) / 2;
            x    = ConvTranspose1d(x, gw.UpWeight[i], gw.UpBias[i], chIn, chOut, curT, upK, upS, upPad);
            curT = x.Length / chOut;

            // Last stage only: ReflectionPad1d((1, 0))
            if (i == numUp - 1)
            {
                x    = ReflectionPadLeft1(x, chOut, curT);
                curT += 1;
            }

            // Align xSource length to curT (conv output may differ by 1 sample)
            xSourceT = xSource.Length / chOut;
            if (xSourceT != curT)
            {
                var aligned = new float[chOut * curT];
                int copyT = Math.Min(xSourceT, curT);
                for (int c = 0; c < chOut; c++)
                    Array.Copy(xSource, c * xSourceT, aligned, c * curT, copyT);
                xSource = aligned;
            }

            // x += xSource
            for (int j = 0; j < x.Length; j++) x[j] += xSource[j];

            // Average of numKernels resblocks
            float[]? xs = null;
            for (int j = 0; j < numKernels; j++)
            {
                int rbIdx = i * numKernels + j;
                // resblockKernelSizes (reference: [3, 7, 11]) selects j's kernel size; dilations
                // are [1,3,5] for every (stage, kernel) pair per the reference's HiFi-GAN defaults.
                var rbOut = AdaINResBlock1Forward(gw.ResBlocks[rbIdx], x, chOut, curT, s, styleDim, resblockKernels[j]);
                if (xs == null) xs = rbOut;
                else for (int k2 = 0; k2 < xs.Length; k2++) xs[k2] += rbOut[k2];
            }
            float invK = 1f / numKernels;
            for (int j = 0; j < xs!.Length; j++) xs[j] *= invK;
            x = xs;
        }

        // Final LeakyReLU (F.leaky_relu default slope = 0.01)
        LeakyReluInPlace(x, 0.01f);

        // conv_post: Conv1d(128, genNFft+2=22, k=7, padding=3)
        x = Conv1d(x, gw.ConvPostWeight, gw.ConvPostBias, inCh: 128, outCh: genNFft + 2, inT: curT, kernel: 7, padding: 3);
        int outT = x.Length / (genNFft + 2);

        // spec = exp(x[:11]), phase = sin(x[11:22])
        int specCh  = genNFft / 2 + 1; // 11
        int phaseCh = genNFft / 2 + 1; // 11
        var spec  = new float[specCh  * outT];
        var phase = new float[phaseCh * outT];
        for (int c = 0; c < specCh;  c++)
            for (int ti = 0; ti < outT; ti++)
                spec[c * outT + ti] = MathF.Exp(x[c * outT + ti]);
        for (int c = 0; c < phaseCh; c++)
            for (int ti = 0; ti < outT; ti++)
                phase[c * outT + ti] = MathF.Sin(x[(specCh + c) * outT + ti]);

        return InverseSTFT(spec, phase, specCh, outT, genNFft, genHop);
    }

    // -----------------------------------------------------------------------
    // AdaINResBlock1 (istftnet.py, Snake1D activated -- used inside Generator only)
    // for (c1,c2,n1,n2,a1,a2): xt=n1(x,s); xt=Snake(xt,a1); xt=c1(xt)
    //                            xt=n2(xt,s); xt=Snake(xt,a2); xt=c2(xt); x=xt+x
    // -----------------------------------------------------------------------

    // kernel is the resblock's kernel_size (istftnet.py AdainResBlock1: 3, 7, or 11 depending on
    // which resblock/noise_res this is -- see Generator.__init__). Dilations are always [1,3,5]
    // for convs1 (per the reference's HiFi-GAN defaults) and always 1 for convs2.
    private static float[] AdaINResBlock1Forward(ResBlockWeights w, float[] x, int ch, int t, float[] s, int styleDim, int kernel)
    {
        var cur = (float[])x.Clone();
        int[] dilatDilations = [1, 3, 5];
        for (int i = 0; i < 3; i++)
        {
            int dil = dilatDilations[i];

            float[] xt = AdaIn1d(cur, s, w.Adain1Weight[i], w.Adain1Bias[i], ch, styleDim, t);
            Snake1DInPlace(xt, w.Alpha1[i], ch, t);
            xt = Conv1dDilated(xt, w.Convs1Weight[i], w.Convs1Bias[i], ch, ch, t, kernel: kernel, dilation: dil);

            xt = AdaIn1d(xt, s, w.Adain2Weight[i], w.Adain2Bias[i], ch, styleDim, t);
            Snake1DInPlace(xt, w.Alpha2[i], ch, t);
            xt = Conv1dDilated(xt, w.Convs2Weight[i], w.Convs2Bias[i], ch, ch, t, kernel: kernel, dilation: 1);

            for (int j = 0; j < cur.Length; j++) cur[j] += xt[j];
        }
        return cur;
    }

    private static void Snake1DInPlace(float[] x, float[] alpha, int ch, int t)
    {
        // alpha shape stored as [channels] (was [1,channels,1] in PyTorch)
        for (int c = 0; c < ch; c++)
        {
            float a    = alpha[c];
            float invA = (MathF.Abs(a) > 1e-8f) ? 1f / a : 1f;
            int   row  = c * t;
            for (int ti = 0; ti < t; ti++)
            {
                float v = x[row + ti];
                float sv = MathF.Sin(a * v);
                x[row + ti] = v + invA * sv * sv;
            }
        }
    }

    private static float[] AdaIn1d(float[] x, float[] style, float[] fcWeight, float[] fcBias, int ch, int styleDim, int t)
    {
        var h = new float[2 * ch];
        for (int o = 0; o < 2 * ch; o++)
        {
            float sum = fcBias[o];
            int wBase = o * styleDim;
            for (int d = 0; d < styleDim; d++) sum += fcWeight[wBase + d] * style[d];
            h[o] = sum;
        }
        var output = new float[ch * t];
        for (int c = 0; c < ch; c++)
        {
            double mean = 0;
            for (int ti = 0; ti < t; ti++) mean += x[c * t + ti];
            mean /= t;
            double var = 0;
            for (int ti = 0; ti < t; ti++) { double d = x[c * t + ti] - mean; var += d * d; }
            var /= t;
            float invStd = (float)(1.0 / Math.Sqrt(var + 1e-5));
            float gamma = h[c], beta = h[ch + c];
            for (int ti = 0; ti < t; ti++)
                output[c * t + ti] = (1f + gamma) * ((float)((x[c * t + ti] - mean) * invStd)) + beta;
        }
        return output;
    }

    // -----------------------------------------------------------------------
    // SineGen (_f02sine, non-pulse mode, fixed-seed for determinism)
    // -----------------------------------------------------------------------

    private static float[] SineGen(float[] f0Up, int len, int sr, int harmonicNum,
                                    float sineAmp, float noiseStd, float voicedThreshold)
    {
        int dim = harmonicNum + 1; // 9
        var harSource = new float[len * dim]; // [len, 9]

        // Random initial phase for harmonics 1..8 (harmonic 0 = fundamental, no noise per reference)
        var rng = new Random(0);
        var randIni = new double[dim];
        randIni[0] = 0.0;
        for (int h = 1; h < dim; h++) randIni[h] = rng.NextDouble();

        var cumPhase = new double[dim];
        for (int h = 0; h < dim; h++) cumPhase[h] = randIni[h];

        for (int n = 0; n < len; n++)
        {
            float f0  = f0Up[n];
            bool voiced = f0 > voicedThreshold;
            float uv = voiced ? 1f : 0f;

            for (int h = 0; h < dim; h++)
            {
                double rad = (double)(f0 * (h + 1)) / sr;
                cumPhase[h] = (cumPhase[h] + rad) % 1.0;
                harSource[n * dim + h] = (float)Math.Sin(cumPhase[h] * 2.0 * Math.PI) * sineAmp;
            }

            // noise_amp = uv*noise_std + (1-uv)*sine_amp/3; then sine*uv + noise
            float noiseAmp = uv * noiseStd + (1f - uv) * sineAmp / 3f;
            for (int h = 0; h < dim; h++)
            {
                float u1 = MathF.Max(1e-7f, (float)rng.NextDouble());
                float u2 = (float)rng.NextDouble();
                float noise = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2) * noiseAmp;
                harSource[n * dim + h] = harSource[n * dim + h] * uv + noise;
            }
        }
        return harSource;
    }

    private static float[] LinearTanh9to1(float[] sines, int len, float[] weight, float[] bias)
    {
        // weight [1, 9] stored flat, bias [1]
        var output = new float[len];
        float b = bias[0];
        for (int n = 0; n < len; n++)
        {
            float sum = b;
            int row = n * 9;
            for (int h = 0; h < 9; h++) sum += weight[h] * sines[row + h];
            output[n] = MathF.Tanh(sum);
        }
        return output;
    }

    // -----------------------------------------------------------------------
    // STFT / iSTFT
    // -----------------------------------------------------------------------

    private static float[] ComputeHarSTFT(float[] signal, int sigLen, int nFft, int hopLen, int frames)
    {
        int specBins = nFft / 2 + 1; // 11
        int outCh    = specBins * 2;  // 22
        var har = new float[outCh * frames];

        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++)
            window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / nFft));

        for (int f = 0; f < frames; f++)
        {
            int start = f * hopLen;
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
                har[k * frames + f]              = MathF.Sqrt(real * real + imag * imag);
                har[(specBins + k) * frames + f] = MathF.Atan2(imag, real);
            }
        }
        return har;
    }

    private static float[] InverseSTFT(float[] spec, float[] phase, int specBins, int frames, int nFft, int hopLen)
    {
        // spec[c,f] = magnitude, phase[c,f] = phase angle
        // Reconstruct X[k,f] = spec*exp(i*phase), then IDFT, overlap-add with Hann window
        int outputLen = (frames - 1) * hopLen + nFft;
        var output = new float[outputLen];
        var norm   = new float[outputLen];

        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++)
            window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / nFft));

        for (int f = 0; f < frames; f++)
        {
            int start = f * hopLen;
            for (int n = 0; n < nFft; n++)
            {
                float val = 0f;
                for (int k = 0; k < specBins; k++)
                {
                    float mag = spec[k * frames + f];
                    float ph  = phase[k * frames + f];
                    float rC  = mag * MathF.Cos(ph);
                    float iC  = mag * MathF.Sin(ph);
                    float angle = 2f * MathF.PI * k * n / nFft;
                    val += rC * MathF.Cos(angle) - iC * MathF.Sin(angle);
                    if (k > 0 && k < specBins - 1) // conjugate mirror bins
                        val += rC * MathF.Cos(angle) - iC * MathF.Sin(angle);
                }
                val /= nFft;
                int si = start + n;
                if (si < outputLen)
                {
                    output[si] += val * window[n];
                    norm[si]   += window[n] * window[n];
                }
            }
        }
        for (int i = 0; i < outputLen; i++)
            if (norm[i] > 1e-8f) output[i] /= norm[i];

        return output;
    }

    // -----------------------------------------------------------------------
    // Conv / Transpose primitives
    // -----------------------------------------------------------------------

    private const float LeakySlope01 = 0.1f;

    private static float[] Cat3ChannelFirst(float[] a, int aCh, float[] b, int bCh, float[] c, int cCh, int t)
    {
        var output = new float[(aCh + bCh + cCh) * t];
        Array.Copy(a, 0, output, 0,        aCh * t);
        Array.Copy(b, 0, output, aCh * t,  bCh * t);
        Array.Copy(c, 0, output, (aCh + bCh) * t, cCh * t);
        return output;
    }

    private static float[] Cat4ChannelFirst(float[] a, int aCh, float[] b, int bCh, float[] c, int cCh, float[] d, int dCh, int t)
    {
        int total = (aCh + bCh + cCh + dCh) * t;
        var output = new float[total];
        Array.Copy(a, 0, output, 0,                  aCh * t);
        Array.Copy(b, 0, output, aCh * t,             bCh * t);
        Array.Copy(c, 0, output, (aCh + bCh) * t,     cCh * t);
        Array.Copy(d, 0, output, (aCh+bCh+cCh) * t,  dCh * t);
        return output;
    }

    // F0_conv / N_conv: Conv1d(1,1, k=3, stride=2, padding=1)
    private static float[] Conv1dStride2(float[] input, float[] weight, float[] bias, int inT)
    {
        const int stride = 2, kernel = 3, padding = 1;
        int outT = (inT + 2 * padding - kernel) / stride + 1;
        var output = new float[outT];
        float b = bias[0];
        for (int ot = 0; ot < outT; ot++)
        {
            float sum = b;
            int center = ot * stride - padding;
            for (int k = 0; k < kernel; k++)
            {
                int src = center + k;
                if ((uint)src < (uint)inT) sum += input[src] * weight[k];
            }
            output[ot] = sum;
        }
        return output;
    }

    // 1x1 conv. Parallelized over output channels + vectorized: for a fixed (oc,ic) the whole
    // output row is `output += weight[ic] * input[ic]`, a single TensorPrimitives.MultiplyAdd
    // over a contiguous span instead of a per-timestep strided scalar sum -- this is the same
    // reordering fix applied to ChatterboxCfmDecoder.cs/ChatterboxVocoder.cs (see those files'
    // comments for the full rationale); Kokoro's Generator runs these at full 24kHz sample rate
    // through several resblock stages, same as Chatterbox's vocoder was before that fix (14.7x
    // there).
    private static float[] Conv1dK1(float[] input, float[] weight, float[] bias, int inCh, int outCh, int inT)
    {
        var output = new float[outCh * inT];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = output.AsSpan(oc * inT, inT);
            outRow.Fill(bias[oc]);
            int wBase = oc * inCh;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * inT, inT);
                TensorPrimitives.MultiplyAdd(inRow, weight[wBase + ic], outRow, outRow);
            }
        });
        return output;
    }

    // Strided Conv1d (noise_convs[0])
    private static float[] Conv1dStrided(float[] input, int inCh, int inT, float[] weight, float[] bias,
                                          int outCh, int kernel, int stride, int padding)
    {
        int outT = (inT + 2 * padding - kernel) / stride + 1;
        if (outT <= 0) outT = 1;
        var output = new float[outCh * outT];
        for (int oc = 0; oc < outCh; oc++)
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
        }
        return output;
    }

    // Standard Conv1d stride=1. Same scale-and-shift-add vectorization as Conv1dK1 above.
    private static float[] Conv1d(float[] input, float[] weight, float[] bias,
                                   int inCh, int outCh, int inT, int kernel, int padding)
    {
        var output = new float[outCh * inT];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = new float[inT];
            Array.Fill(outRow, bias[oc]);
            int wOcBase = oc * inCh * kernel;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * inT, inT);
                int wBase = wOcBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int shift = k - padding;
                    AxpyShifted(inRow, weight[wBase + k], outRow, shift, inT);
                }
            }
            Array.Copy(outRow, 0, output, oc * inT, inT);
        });
        return output;
    }

    // Dilated Conv1d (AdaINResBlock1 convs). Same vectorization.
    private static float[] Conv1dDilated(float[] input, float[] weight, float[] bias,
                                          int inCh, int outCh, int inT, int kernel, int dilation)
    {
        int padding = (kernel * dilation - dilation) / 2;
        var output = new float[outCh * inT];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = new float[inT];
            Array.Fill(outRow, bias[oc]);
            int wOcBase = oc * inCh * kernel;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * inT, inT);
                int wBase = wOcBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int shift = k * dilation - padding;
                    AxpyShifted(inRow, weight[wBase + k], outRow, shift, inT);
                }
            }
            Array.Copy(outRow, 0, output, oc * inT, inT);
        });
        return output;
    }

    /// <summary>output[ti] += scale * input[ti + shift] for every ti where ti+shift is in [0,t).</summary>
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

    // ConvTranspose1d for ups[i].
    // PyTorch ConvTranspose1d weight layout: [inCh, outCh, kernel]. Loop order is oc-outer (each
    // output channel reads from all input channels/positions) so this parallelizes the same way
    // as the other conv helpers, unlike the original ic-outer scatter which would need per-output
    // locking or atomics to run concurrently.
    private static float[] ConvTranspose1d(float[] input, float[] weight, float[] bias,
                                            int inCh, int outCh, int inT,
                                            int kernel, int stride, int padding)
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

    // ReflectionPad1d((1,0)): prepend reflected sample to each channel
    private static float[] ReflectionPadLeft1(float[] input, int ch, int t)
    {
        int newT = t + 1;
        var output = new float[ch * newT];
        for (int c = 0; c < ch; c++)
        {
            int srcBase = c * t;
            int dstBase = c * newT;
            output[dstBase] = input[srcBase + Math.Min(1, t - 1)]; // reflect index 1
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
}
