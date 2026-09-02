namespace OpenTail.Stingray.Diffusion.AceStep.Vae;

/// <summary>
/// Real `AutoencoderOobleck` encoder (`OobleckEncoder`), the counterpart to
/// <see cref="AceStepOobleckDecoder"/>. Transcribed directly from the real `diffusers` 0.40.0
/// source -- see docs/064-acestep-implementation-plan.md.
///
/// <para>Ported specifically to derive a real `silence_latent` (VAE-encoded true audio silence)
/// self-sufficiently, without needing an external asset this project doesn't have: the real
/// `diffusers` ACE-Step pipeline ships a `silence_latent` buffer for plain text-to-music generation
/// (no reference audio) and for the timbre encoder's "no reference" input, but that buffer is not
/// present in this project's downloaded raw `Ace-Step1.5` checkpoint (confirmed by inspecting its
/// real safetensors header). This project's checkpoint DOES include the real VAE encoder weights
/// (`encoder.*`, 183 tensors), so encoding a real all-zero (true silence) waveform through this
/// class reproduces the same real "VAE-encoded audio-silence" the reference describes -- see
/// <see cref="AceStepFlowScheduler"/>/<see cref="Conditioning.AceStepConditionEncoder"/> for where
/// the result gets used.</para>
///
/// <para><b>Real encoder structure</b> (`OobleckEncoder`): `conv1(k=7,pad=3): audioChannels(2) -&gt;
/// encoderHiddenSize(128)` -&gt; 5x `OobleckEncoderBlock` (3x `OobleckResidualUnit` at dilations
/// 1/3/9 FIRST, then `snake1` -&gt; strided `Conv1d(k=2*stride,pad=ceil(stride/2))` -- note this is
/// the REVERSE order from the decoder block, which upsamples-then-residual-units) -&gt; final
/// `snake1(2048)` -&gt; `conv2(k=3,pad=1): 2048 -&gt; encoderHiddenSize(128)`. Output is 128 channels
/// = `[mean(64), logScale(64)]` (`OobleckDiagonalGaussianDistribution`); this class returns only the
/// deterministic `mode()` (the mean half), matching how the real pipeline derives a fixed
/// `silence_latent` buffer (no stochastic sampling).</para>
/// </summary>
public sealed class AceStepOobleckEncoderWeights
{
    public required float[] Conv1Weight { get; init; }
    public required float[] Conv1Bias { get; init; }
    public required OobleckEncoderBlockWeights[] Blocks { get; init; }
    public required float[] FinalSnakeAlpha { get; init; }
    public required float[] FinalSnakeBeta { get; init; }
    public required float[] Conv2Weight { get; init; }
    public required float[] Conv2Bias { get; init; }

    public static AceStepOobleckEncoderWeights Load(SafetensorsLoader loader)
    {
        var blocks = new OobleckEncoderBlockWeights[AceStepConfig.VaeDownsamplingRatios.Length];
        for (int i = 0; i < blocks.Length; i++)
        {
            string p = $"encoder.block.{i}";
            blocks[i] = new OobleckEncoderBlockWeights
            {
                ResUnits =
                [
                    LoadResUnit(loader, $"{p}.res_unit1"),
                    LoadResUnit(loader, $"{p}.res_unit2"),
                    LoadResUnit(loader, $"{p}.res_unit3"),
                ],
                Snake1Alpha = loader.ReadF32($"{p}.snake1.alpha"),
                Snake1Beta = loader.ReadF32($"{p}.snake1.beta"),
                ConvWeight = FoldWeightNorm(loader, $"{p}.conv1.weight_g", $"{p}.conv1.weight_v"),
                ConvBias = loader.ReadF32($"{p}.conv1.bias"),
            };
        }

        return new AceStepOobleckEncoderWeights
        {
            Conv1Weight = FoldWeightNorm(loader, "encoder.conv1.weight_g", "encoder.conv1.weight_v"),
            Conv1Bias = loader.ReadF32("encoder.conv1.bias"),
            Blocks = blocks,
            FinalSnakeAlpha = loader.ReadF32("encoder.snake1.alpha"),
            FinalSnakeBeta = loader.ReadF32("encoder.snake1.beta"),
            Conv2Weight = FoldWeightNorm(loader, "encoder.conv2.weight_g", "encoder.conv2.weight_v"),
            Conv2Bias = loader.ReadF32("encoder.conv2.bias"),
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

public sealed class OobleckEncoderBlockWeights
{
    public required OobleckResidualUnitWeights[] ResUnits { get; init; } // 3, dilations 1/3/9
    public required float[] Snake1Alpha { get; init; }
    public required float[] Snake1Beta { get; init; }
    public required float[] ConvWeight { get; init; } // strided downsample Conv1d
    public required float[] ConvBias { get; init; }
}

public static class AceStepOobleckEncoder
{
    /// <summary>Real per-stage channel progression: `encoderHiddenSize * ([1]+channelMultiples)[i]`.</summary>
    private static int[] ChannelsPerStage()
    {
        int hidden = AceStepConfig.VaeDecoderChannels; // == encoder_hidden_size, both 128 in the real config
        var mult = new int[AceStepConfig.VaeChannelMultiples.Length + 1];
        mult[0] = 1;
        Array.Copy(AceStepConfig.VaeChannelMultiples, 0, mult, 1, AceStepConfig.VaeChannelMultiples.Length);
        var result = new int[mult.Length];
        for (int i = 0; i < mult.Length; i++) result[i] = hidden * mult[i];
        return result;
    }

    /// <summary>Encodes real `[audioChannels(2), T]` PCM into the deterministic `mode()` (mean-only, no stochastic sampling) latent `[decoderInputChannels(64), T/hopLength]` -- matches how the real pipeline derives a fixed `silence_latent` buffer. Real, non-mean half of the encoder's raw 128-channel output (the log-scale/variance half) is real but unused here since only the mean is needed for a deterministic conditioning latent.</summary>
    public static float[] EncodeMode(AceStepOobleckEncoderWeights w, float[] pcm, int audioChannels, int sampleCount)
    {
        var channelsPerStage = ChannelsPerStage();

        var x = AceStepOobleckKernels.FullConv1d(pcm, audioChannels, channelsPerStage[0], sampleCount, w.Conv1Weight, w.Conv1Bias, kernel: 7, dilation: 1, padding: 3);

        int ch = channelsPerStage[0];
        int curT = sampleCount;
        for (int i = 0; i < w.Blocks.Length; i++)
        {
            int outCh = channelsPerStage[i + 1];
            int stride = AceStepConfig.VaeDownsamplingRatios[i];
            (x, curT) = EncoderBlock(x, ch, outCh, curT, w.Blocks[i], stride);
            ch = outCh;
        }

        x = AceStepOobleckKernels.Snake(x, ch, curT, w.FinalSnakeAlpha, w.FinalSnakeBeta);
        var raw = AceStepOobleckKernels.FullConv1d(x, ch, AceStepConfig.VaeDecoderChannels, curT, w.Conv2Weight, w.Conv2Bias, kernel: 3, dilation: 1, padding: 1);

        // raw is [128, curT] == [mean(64), logScale(64)] channel-major -- keep only the mean half.
        int latentDim = AceStepConfig.VaeDecoderInputChannels; // 64
        var mode = new float[latentDim * curT];
        Array.Copy(raw, 0, mode, 0, latentDim * curT);
        return mode;
    }

    private static (float[] Data, int T) EncoderBlock(float[] x, int inCh, int outCh, int t, OobleckEncoderBlockWeights w, int stride)
    {
        var cur = x;
        cur = AceStepOobleckKernels.ResidualUnit(cur, inCh, t, w.ResUnits[0], dilation: 1);
        cur = AceStepOobleckKernels.ResidualUnit(cur, inCh, t, w.ResUnits[1], dilation: 3);
        cur = AceStepOobleckKernels.ResidualUnit(cur, inCh, t, w.ResUnits[2], dilation: 9);
        cur = AceStepOobleckKernels.Snake(cur, inCh, t, w.Snake1Alpha, w.Snake1Beta);

        int kernel = 2 * stride;
        int padding = (int)MathF.Ceiling(stride / 2f);
        return AceStepOobleckKernels.StridedConv1d(cur, inCh, outCh, t, w.ConvWeight, w.ConvBias, kernel, stride, padding);
    }
}
