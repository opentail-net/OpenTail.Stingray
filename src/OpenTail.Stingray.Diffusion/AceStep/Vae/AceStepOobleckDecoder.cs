
namespace OpenTail.Stingray.Diffusion.AceStep.Vae;

/// <summary>
/// Real `AutoencoderOobleck` decoder (`diffusers.models.autoencoders.autoencoder_oobleck`), the
/// VAE ACE-Step 1.5 uses to turn 25Hz stereo latents into 48kHz stereo PCM. Transcribed directly
/// from the real `diffusers` 0.40.0 source (already installed in this environment), tensor
/// names/shapes confirmed against the real checkpoint's own safetensors header -- see
/// docs/064-acestep-implementation-plan.md's "Corrections and confirmations" section.
///
/// <para><b>NOT the same architecture as this project's existing Stable Audio 3 `AcousticVae`</b>
/// (Stability's bespoke transformer-resampling design) -- confirmed different, do not conflate.
/// This IS structurally similar to `Parler.DacDecoder`/`Primitives.EncodecDecoderKernels`
/// (weight-normalized convs, residual units with dilations 1/3/9), but uses a real TWO-parameter
/// Snake activation (`alpha` AND `beta`, both LOG-SCALE -- `exp()` must be applied to both before
/// use) rather than DAC's single-parameter Snake or EnCodec's ELU. Written self-contained here
/// rather than force-reusing `EncodecDecoderKernels` (CLAUDE.md rule 7: DRY once duplication is
/// proven across a second real, verified user of the SAME activation formula -- Oobleck's Snake
/// differs from both existing variants).</para>
///
/// <para>Real decoder structure (`OobleckDecoder`, confirmed against the real checkpoint's tensor
/// shapes): `conv1(k=7,pad=3): decoder_input_channels(64) -&gt; decoder_channels(128)*channel_
/// multiples[-1](16) = 2048` -&gt; 5x `OobleckDecoderBlock` (snake1 -&gt; ConvTranspose1d(k=2*
/// stride, pad=ceil(stride/2)) -&gt; 3x `OobleckResidualUnit` at dilations 1/3/9), channel
/// progression `2048-&gt;1024-&gt;512-&gt;256-&gt;128-&gt;128` using strides = the REVERSE of the
/// real config's `downsampling_ratios` (`[2,4,4,6,10]` -&gt; decoder strides `[10,6,4,4,2]`) --
/// &gt; final `snake1(128)` -&gt; `conv2(k=7,pad=3,NO bias): 128 -&gt; audio_channels(2)`.</para>
/// </summary>
public sealed class AceStepOobleckDecoderWeights
{
    public required float[] Conv1Weight { get; init; }
    public required float[] Conv1Bias { get; init; }
    public required OobleckDecoderBlockWeights[] Blocks { get; init; }
    public required float[] FinalSnakeAlpha { get; init; }
    public required float[] FinalSnakeBeta { get; init; }
    public required float[] Conv2Weight { get; init; } // no bias, real config

    public static AceStepOobleckDecoderWeights Load(SafetensorsLoader loader)
    {
        var blocks = new OobleckDecoderBlockWeights[AceStepConfig.VaeDownsamplingRatios.Length];
        for (int i = 0; i < blocks.Length; i++)
        {
            string p = $"decoder.block.{i}";
            blocks[i] = new OobleckDecoderBlockWeights
            {
                Snake1Alpha = loader.ReadF32($"{p}.snake1.alpha"),
                Snake1Beta = loader.ReadF32($"{p}.snake1.beta"),
                ConvTWeight = FoldWeightNorm(loader, $"{p}.conv_t1.weight_g", $"{p}.conv_t1.weight_v"),
                ConvTBias = loader.ReadF32($"{p}.conv_t1.bias"),
                ResUnits =
                [
                    LoadResUnit(loader, $"{p}.res_unit1"),
                    LoadResUnit(loader, $"{p}.res_unit2"),
                    LoadResUnit(loader, $"{p}.res_unit3"),
                ],
            };
        }

        return new AceStepOobleckDecoderWeights
        {
            Conv1Weight = FoldWeightNorm(loader, "decoder.conv1.weight_g", "decoder.conv1.weight_v"),
            Conv1Bias = loader.ReadF32("decoder.conv1.bias"),
            Blocks = blocks,
            FinalSnakeAlpha = loader.ReadF32("decoder.snake1.alpha"),
            FinalSnakeBeta = loader.ReadF32("decoder.snake1.beta"),
            Conv2Weight = FoldWeightNorm(loader, "decoder.conv2.weight_g", "decoder.conv2.weight_v"),
        };
    }

    private static OobleckResidualUnitWeights LoadResUnit(SafetensorsLoader loader, string p) => new()
    {
        Snake1Alpha = loader.ReadF32($"{p}.snake1.alpha"),
        Snake1Beta = loader.ReadF32($"{p}.snake1.beta"),
        Conv1Weight = FoldWeightNorm(loader, $"{p}.conv1.weight_g", $"{p}.conv1.weight_v"),
        Conv1Bias = loader.ReadF32($"{p}.conv1.bias"),
        Snake2Alpha = loader.ReadF32($"{p}.snake2.alpha"),
        Snake2Beta = loader.ReadF32($"{p}.snake2.beta"),
        Conv2Weight = FoldWeightNorm(loader, $"{p}.conv2.weight_g", $"{p}.conv2.weight_v"),
        Conv2Bias = loader.ReadF32($"{p}.conv2.bias"),
    };

    /// <summary>Folds `weight_g`(`[outCh,1,1]`)*`weight_v`(`[outCh,inCh,K]`)/||v[outCh,:,:]||_2 into a plain conv weight -- same PyTorch `weight_norm` (dim=0) convention as `Parler.DacWeights`/`Primitives.EncodecDecoderWeights`'s identically-shaped helpers.</summary>
    private static float[] FoldWeightNorm(SafetensorsLoader loader, string weightGName, string weightVName)
    {
        var g = loader.ReadF32(weightGName);
        var v = loader.ReadF32(weightVName);
        int[] vShape = loader.GetShape(weightVName);
        int outCh = vShape[0];
        int perChannel = v.Length / outCh;

        var folded = new float[v.Length];
        for (int o = 0; o < outCh; o++)
        {
            double sumSq = 0;
            int baseIdx = o * perChannel;
            for (int j = 0; j < perChannel; j++) { double vv = v[baseIdx + j]; sumSq += vv * vv; }
            float norm = (float)Math.Sqrt(sumSq);
            float scale = norm > 1e-12f ? g[o] / norm : 0f;
            for (int j = 0; j < perChannel; j++) folded[baseIdx + j] = v[baseIdx + j] * scale;
        }
        return folded;
    }
}

public sealed class OobleckDecoderBlockWeights
{
    public required float[] Snake1Alpha { get; init; }
    public required float[] Snake1Beta { get; init; }
    public required float[] ConvTWeight { get; init; } // ConvTranspose1d, [inCh, outCh, K] flat
    public required float[] ConvTBias { get; init; }
    public required OobleckResidualUnitWeights[] ResUnits { get; init; } // 3, dilations 1/3/9
}

public sealed class OobleckResidualUnitWeights
{
    public required float[] Snake1Alpha { get; init; }
    public required float[] Snake1Beta { get; init; }
    public required float[] Conv1Weight { get; init; } // k=7, dilated
    public required float[] Conv1Bias { get; init; }
    public required float[] Snake2Alpha { get; init; }
    public required float[] Snake2Beta { get; init; }
    public required float[] Conv2Weight { get; init; } // k=1
    public required float[] Conv2Bias { get; init; }
}

public static class AceStepOobleckDecoder
{
    /// <summary>Real decoder channel-per-stage progression, derived from the real config's `channel_multiples=[1,2,4,8,16]` (prepended with 1) the same way `OobleckDecoder.__init__` computes `input_dim`/`output_dim` per block -- NOT a simple halving-each-stage pattern (the last stage keeps 128-&gt;128).</summary>
    private static int[] ChannelsPerStage()
    {
        int channels = AceStepConfig.VaeDecoderChannels; // 128
        var mult = new int[AceStepConfig.VaeChannelMultiples.Length + 1];
        mult[0] = 1;
        Array.Copy(AceStepConfig.VaeChannelMultiples, 0, mult, 1, AceStepConfig.VaeChannelMultiples.Length);
        int stages = AceStepConfig.VaeDownsamplingRatios.Length;
        var result = new int[stages + 1];
        for (int i = 0; i <= stages; i++) result[i] = channels * mult[stages - i];
        return result;
    }

    /// <summary>Real decoder upsample strides: the REVERSE of the config's `downsampling_ratios`.</summary>
    private static int[] UpsamplingRatios() => [.. AceStepConfig.VaeDownsamplingRatios.Reverse()];

    /// <summary>Full real decode: `[latentChannels(64), T]` latent -&gt; `[audioChannels(2), T*hopLength]` stereo PCM (interleave channels yourself if a single interleaved buffer is needed). NOT tanh-clamped -- real Oobleck returns the raw final conv output.</summary>
    public static float[] Decode(AceStepOobleckDecoderWeights w, float[] latent, int latentLen)
    {
        var channelsPerStage = ChannelsPerStage();
        var ratios = UpsamplingRatios();

        var x = FullConv1d(latent, AceStepConfig.VaeDecoderInputChannels, channelsPerStage[0], latentLen, w.Conv1Weight, w.Conv1Bias, kernel: 7, dilation: 1, padding: 3);

        int ch = channelsPerStage[0];
        int curT = latentLen;
        for (int i = 0; i < w.Blocks.Length; i++)
        {
            int outCh = channelsPerStage[i + 1];
            int stride = ratios[i];
            (x, curT) = DecoderBlock(x, ch, outCh, curT, w.Blocks[i], stride);
            ch = outCh;
        }

        x = Snake(x, ch, curT, w.FinalSnakeAlpha, w.FinalSnakeBeta);
        var pcm = FullConv1d(x, ch, AceStepConfig.VaeAudioChannels, curT, w.Conv2Weight, bias: null, kernel: 7, dilation: 1, padding: 3);
        return pcm; // [audioChannels, curT]
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, OobleckDecoderBlockWeights w, int stride)
    {
        var y = Snake(x, inCh, t, w.Snake1Alpha, w.Snake1Beta);
        int kernel = 2 * stride;
        int padding = (int)MathF.Ceiling(stride / 2f);
        var (up, outT) = ConvTranspose1d(y, inCh, outCh, t, w.ConvTWeight, w.ConvTBias, kernel, stride, padding);

        var cur = up;
        cur = ResidualUnit(cur, outCh, outT, w.ResUnits[0], dilation: 1);
        cur = ResidualUnit(cur, outCh, outT, w.ResUnits[1], dilation: 3);
        cur = ResidualUnit(cur, outCh, outT, w.ResUnits[2], dilation: 9);
        return (cur, outT);
    }

    private static float[] ResidualUnit(float[] x, int channels, int t, OobleckResidualUnitWeights w, int dilation)
    {
        int pad = (7 - 1) * dilation / 2;
        var y = Snake(x, channels, t, w.Snake1Alpha, w.Snake1Beta);
        y = FullConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias, kernel: 7, dilation: dilation, padding: pad);
        y = Snake(y, channels, t, w.Snake2Alpha, w.Snake2Beta);
        y = FullConv1d(y, channels, channels, t, w.Conv2Weight, w.Conv2Bias, kernel: 1, dilation: 1, padding: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i]; // real OobleckResidualUnit: plain identity shortcut
        return output;
    }

    /// <summary>Real Oobleck `Snake1d`: `x + (1/(exp(beta)+1e-9)) * sin(exp(alpha)*x)^2` -- both `alpha` and `beta` are stored in LOG-SCALE (`logscale=True`, real default), confirmed from `diffusers/models/autoencoders/autoencoder_oobleck.py`. Do NOT use the raw stored values directly -- omitting the `exp()` would be a silent, wrong-but-plausible-shaped bug.</summary>
    private static float[] Snake(float[] x, int channels, int t, float[] logAlpha, float[] logBeta)
    {
        var output = new float[x.Length];
        for (int c = 0; c < channels; c++)
        {
            float alpha = MathF.Exp(logAlpha[c]);
            float beta = MathF.Exp(logBeta[c]);
            float invBeta = 1f / (beta + 1e-9f);
            int baseIdx = c * t;
            for (int i = 0; i < t; i++)
            {
                float v = x[baseIdx + i];
                float s = MathF.Sin(alpha * v);
                output[baseIdx + i] = v + invBeta * s * s;
            }
        }
        return output;
    }

    /// <summary>Real FULL (non-depthwise) Conv1d, symmetric ("same"-style) padding -- same im2col+GEMM technique as `Parler.DacDecoder`/`Primitives.EncodecDecoderKernels`'s identically-shaped helpers. `bias` may be null (real `conv2` has `bias=False`).</summary>
    private static unsafe float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[]? bias, int kernel, int dilation, int padding)
    {
        int rowLen = inCh * kernel;
        var col = new float[t * rowLen];
        Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - padding + k * dilation;
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
                float b = bias?[oc] ?? 0f;
                float* wOc = weightPtrLocal + oc * rowLen;
                float* outBase = outputPtrLocal + oc * t;
                for (int ti = 0; ti < t; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colPtrLocal + ti * rowLen, rowLen);
            });
        }
        return output;
    }

    /// <summary>Real ConvTranspose1d, weight layout `[inCh, outCh, kernel]` flat row-major, WITH real padding/cropping applied inline (real `nn.ConvTranspose1d(..., padding=ceil(stride/2))`, unlike EnCodec's own transpose conv which trims separately) -- standard PyTorch semantics: `outT = (t-1)*stride - 2*padding + kernel`.</summary>
    private static (float[] Data, int T) ConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride, int padding)
    {
        int outT = (t - 1) * stride - 2 * padding + kernel;
        var output = new float[outCh * outT];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int dstBase = oc * outT;
            for (int ti = 0; ti < outT; ti++) output[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = x[srcBase + ti];
                    int outStart = ti * stride - padding;

                    int kStart = outStart < 0 ? -outStart : 0;
                    int kEnd = outStart + kernel > outT ? outT - outStart : kernel;
                    if (kStart >= kEnd) continue;

                    var wSpan = weight.AsSpan(wBase + kStart, kEnd - kStart);
                    var dstSpan = output.AsSpan(dstBase + outStart + kStart, kEnd - kStart);
                    TensorPrimitives.MultiplyAdd(wSpan, v, dstSpan, dstSpan);
                }
            }
        });
        return (output, outT);
    }
}
