namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 Flow-VAE vocoder (`MiniMaxMusic3Vocoder`), transcribed directly from the
/// real `diffusers` 0.40.0 source (`diffusers/models/autoencoders/minimax_music3_vocoder.py`,
/// already installed in this environment -- see docs/066-minimax-music3-future-plan.md for the
/// full archaeology). A real, single-parameter-Snake DAC-style decoder -- structurally the same
/// shape as this project's own `Audio.Parler.DacDecoder`, a real second caller of that shape, but
/// kept self-contained here (CLAUDE.md rule 7: DRY once this class is itself verified, not
/// speculatively across a different domain namespace in the same change that ports it).
///
/// <para><b>Real "folded stereo" detail</b> (confirmed from the real `forward`): the decoder
/// network itself is MONO (`latent_channels // 2` input channels) -- stereo is produced by
/// reshaping the real `[batch, latentChannels(128), length]` latent into
/// `[batch*2, latentChannels/2(64), length]` (the left/right channel-streams folded into the batch
/// dimension) BEFORE the decoder runs, then un-folded back to `[batch, 2, samples]` after. This is
/// NOT a stereo-aware decoder architecture; it is a mono decoder run twice, each half getting its
/// own 64 latent channels.</para>
///
/// <para><b>Real channel progression</b> (`decoder_hidden_dim=1536`, halving per stage across
/// `upsampling_ratios=(8,8,4,2)`, hop=512): `1536 -&gt; 768 -&gt; 384 -&gt; 192 -&gt; 96`, confirmed
/// directly against the real checkpoint's own tensor shapes (`blocks.0.conv_t1` is `[1536-&gt;768,
/// stride 8]`, etc.).</para>
/// </summary>
public sealed class MiniMaxMusic3VocoderWeights
{
    public required float[] DecInProjWeight { get; init; } // Conv1d k=1, no weight_norm, WITH bias
    public required float[] DecInProjBias { get; init; }
    public required float[] ConvInWeight { get; init; } // weight_norm, k=7
    public required float[] ConvInBias { get; init; }
    public required VocoderBlockWeights[] Blocks { get; init; } // 4 real blocks, strides (8,8,4,2)
    public required float[] SnakeOutAlpha { get; init; }
    public required float[] ConvOutWeight { get; init; } // weight_norm, k=7, out_channels=1
    public required float[] ConvOutBias { get; init; }

    public static MiniMaxMusic3VocoderWeights Load(SafetensorsLoader loader)
    {
        var ratios = MiniMaxMusic3Config.VocoderUpsamplingRatios;
        var blocks = new VocoderBlockWeights[ratios.Length];
        for (int i = 0; i < ratios.Length; i++)
            blocks[i] = LoadBlock(loader, $"blocks.{i}");

        return new MiniMaxMusic3VocoderWeights
        {
            DecInProjWeight = loader.ReadF32("dec_in_proj.weight"),
            DecInProjBias = loader.ReadF32("dec_in_proj.bias"),
            ConvInWeight = FoldWeightNorm(loader, "conv_in.weight_g", "conv_in.weight_v"),
            ConvInBias = loader.ReadF32("conv_in.bias"),
            Blocks = blocks,
            SnakeOutAlpha = loader.ReadF32("snake_out.alpha"),
            ConvOutWeight = FoldWeightNorm(loader, "conv_out.weight_g", "conv_out.weight_v"),
            ConvOutBias = loader.ReadF32("conv_out.bias"),
        };
    }

    private static VocoderBlockWeights LoadBlock(SafetensorsLoader loader, string p) => new()
    {
        Snake1Alpha = loader.ReadF32($"{p}.snake1.alpha"),
        ConvTWeight = FoldWeightNorm(loader, $"{p}.conv_t1.weight_g", $"{p}.conv_t1.weight_v"),
        ConvTBias = loader.ReadF32($"{p}.conv_t1.bias"),
        ResUnits =
        [
            LoadResUnit(loader, $"{p}.res_unit1"),
            LoadResUnit(loader, $"{p}.res_unit2"),
            LoadResUnit(loader, $"{p}.res_unit3"),
        ],
    };

    private static VocoderResUnitWeights LoadResUnit(SafetensorsLoader loader, string p) => new()
    {
        Snake1Alpha = loader.ReadF32($"{p}.snake1.alpha"),
        Conv1Weight = FoldWeightNorm(loader, $"{p}.conv1.weight_g", $"{p}.conv1.weight_v"),
        Conv1Bias = loader.ReadF32($"{p}.conv1.bias"),
        Snake2Alpha = loader.ReadF32($"{p}.snake2.alpha"),
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

public sealed class VocoderBlockWeights
{
    public required float[] Snake1Alpha { get; init; }
    public required float[] ConvTWeight { get; init; }
    public required float[] ConvTBias { get; init; }
    public required VocoderResUnitWeights[] ResUnits { get; init; } // 3, dilations 1/3/9
}

public sealed class VocoderResUnitWeights
{
    public required float[] Snake1Alpha { get; init; }
    public required float[] Conv1Weight { get; init; } // k=7, dilated
    public required float[] Conv1Bias { get; init; }
    public required float[] Snake2Alpha { get; init; }
    public required float[] Conv2Weight { get; init; } // k=1
    public required float[] Conv2Bias { get; init; }
}

public static class MiniMaxMusic3Vocoder
{
    /// <summary>Real decode: `[latentChannels(128), T]` flow-matched latent -&gt; `[2, T*hopLength(512)]` stereo waveform in [-1,1] (channel-major; the "folded stereo" split happens internally -- see class doc comment).</summary>
    public static float[] Decode(MiniMaxMusic3VocoderWeights w, float[] latent, int latentLen)
    {
        int fullLatentDim = MiniMaxMusic3Config.VocoderLatentChannels; // 128
        int halfLatentDim = fullLatentDim / 2; // 64

        // Fold [128, T] -> two [64, T] channel-streams (left, right), decoded independently.
        var results = new float[2][];
        for (int ch = 0; ch < 2; ch++)
        {
            var streamLatent = new float[halfLatentDim * latentLen];
            for (int c = 0; c < halfLatentDim; c++)
                Array.Copy(latent, (ch * halfLatentDim + c) * latentLen, streamLatent, c * latentLen, latentLen);
            results[ch] = DecodeMono(w, streamLatent, latentLen);
        }

        int samples = results[0].Length;
        var stereo = new float[2 * samples];
        Array.Copy(results[0], 0, stereo, 0, samples);
        Array.Copy(results[1], 0, stereo, samples, samples);
        return stereo;
    }

    private static float[] DecodeMono(MiniMaxMusic3VocoderWeights w, float[] latent, int latentLen)
    {
        int decoderInputDim = MiniMaxMusic3Config.VocoderDecoderInputDim; // 1024
        int decoderHiddenDim = MiniMaxMusic3Config.VocoderDecoderHiddenDim; // 1536
        int halfLatentDim = MiniMaxMusic3Config.VocoderLatentChannels / 2;

        var projected = FullConv1d(latent, halfLatentDim, decoderInputDim, latentLen, w.DecInProjWeight, w.DecInProjBias, kernel: 1, dilation: 1, padding: 0);
        var x = FullConv1d(projected, decoderInputDim, decoderHiddenDim, latentLen, w.ConvInWeight, w.ConvInBias, kernel: 7, dilation: 1, padding: 3);

        int ch = decoderHiddenDim;
        int curT = latentLen;
        var ratios = MiniMaxMusic3Config.VocoderUpsamplingRatios;
        for (int i = 0; i < w.Blocks.Length; i++)
        {
            int outCh = ch / 2;
            (x, curT) = DecoderBlock(x, ch, outCh, curT, w.Blocks[i], ratios[i]);
            ch = outCh;
        }

        x = Snake(x, ch, curT, w.SnakeOutAlpha);
        var raw = FullConv1d(x, ch, 1, curT, w.ConvOutWeight, w.ConvOutBias, kernel: 7, dilation: 1, padding: 3);

        var waveform = new float[raw.Length];
        for (int i = 0; i < raw.Length; i++) waveform[i] = MathF.Tanh(raw[i]);
        return waveform;
    }

    private static (float[] Data, int T) DecoderBlock(float[] x, int inCh, int outCh, int t, VocoderBlockWeights w, int stride)
    {
        var y = Snake(x, inCh, t, w.Snake1Alpha);
        int kernel = 2 * stride;
        int padding = (int)MathF.Ceiling(stride / 2f);
        var (up, outT) = ConvTranspose1d(y, inCh, outCh, t, w.ConvTWeight, w.ConvTBias, kernel, stride, padding);

        var cur = up;
        cur = ResidualUnit(cur, outCh, outT, w.ResUnits[0], dilation: 1);
        cur = ResidualUnit(cur, outCh, outT, w.ResUnits[1], dilation: 3);
        cur = ResidualUnit(cur, outCh, outT, w.ResUnits[2], dilation: 9);
        return (cur, outT);
    }

    private static float[] ResidualUnit(float[] x, int channels, int t, VocoderResUnitWeights w, int dilation)
    {
        int pad = (7 - 1) * dilation / 2;
        var y = Snake(x, channels, t, w.Snake1Alpha);
        y = FullConv1d(y, channels, channels, t, w.Conv1Weight, w.Conv1Bias, kernel: 7, dilation: dilation, padding: pad);
        y = Snake(y, channels, t, w.Snake2Alpha);
        y = FullConv1d(y, channels, channels, t, w.Conv2Weight, w.Conv2Bias, kernel: 1, dilation: 1, padding: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i];
        return output;
    }

    /// <summary>Real single-parameter Snake (`x + (alpha+1e-9)^-1 * sin(alpha*x)^2`), `alpha` used
    /// DIRECTLY (no `exp()` -- confirmed real, unlike ACE-Step's Oobleck two-parameter LOG-SCALE
    /// Snake; real `alpha` init is `torch.ones`, matching the standard DAC convention).</summary>
    private static float[] Snake(float[] x, int channels, int t, float[] alpha)
    {
        var output = new float[x.Length];
        for (int c = 0; c < channels; c++)
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
        }
        return output;
    }

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
