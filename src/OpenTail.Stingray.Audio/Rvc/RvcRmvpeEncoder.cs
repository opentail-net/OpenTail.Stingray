
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real RMVPE pitch-extractor forward pass, transcribed directly from
/// `examples/audio.cpp/src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp`'s real
/// `build_rmvpe_feature_graph`/`build_gru_chunk_graph`/`build_rmvpe_head_graph` (not guessed) --
/// see `docs/audio-review-progress.md`'s RMVPE section for the full architecture derivation.
/// Operates on a real log-mel spectrogram (128 bins), channel-first as a single-channel "image"
/// [1, melBins, frames]: BatchNorm2d -> 5-level ResNet-U-Net encoder (skip connections saved
/// pre-pool) -> 4-level bottleneck -> 5-level ResNet-U-Net decoder (ConvTranspose2d upsample +
/// skip-concat) -> final 3-channel conv -> reshape to [frames, 384] -> bidirectional GRU (real,
/// manually-unrolled cell, matching the reference exactly rather than a black-box RNN) -> Linear
/// (512-&gt;360) -> Sigmoid (real per-class probabilities, not a softmax distribution).
/// </summary>
public static class RvcRmvpeEncoder
{
    /// <summary>Returns [frames, 360] sigmoid pitch-class probabilities. <paramref name="mel"/> is
    /// frame-major [frames, 128] (matches this codebase's existing mel-extractor convention).</summary>
    public static float[][] Forward(RvcRmvpeWeights w, float[][] mel)
    {
        int frames = mel.Length;
        // Real reference orientation (rmvpe_pitch_extractor.cpp): feature.input tensor shape is
        // [1, melBins, frames], transposed via TransposeModule({0,2,1}) to [1, frames, melBins]
        // before being reshaped into the U-Net's [1,1,H,W] image -- i.e. H=frames, W=melBins, NOT
        // H=melBins/W=frames. Convolution kernels are not symmetric under an H/W swap, so getting
        // this backwards silently produces near-collapsed (tiny-variance) output instead of an
        // outright crash -- found via a real end-to-end salience-stats mismatch against the C++
        // reference (ours ~50x smaller mean/std/max than the reference's real trace).
        var image = new float[1][,];
        image[0] = new float[frames, RvcRmvpeWeights.MelBins];
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < RvcRmvpeWeights.MelBins; m++)
                image[0][f, m] = mel[f][m];

        var x = BatchNorm2d(image, w.EncoderInputBn);
        Trace("after-input-bn", x);

        var skips = new List<float[][,]>(5);
        int channels = 1;
        int outChannels = 16;
        for (int level = 0; level < 5; level++)
        {
            x = ResUNetLevel(x, w.EncoderLevels[level], channels, outChannels);
            channels = outChannels;
            skips.Add(x);
            x = AvgPool2x2(x, channels);
            outChannels *= 2;
            Trace($"after-encoder-level{level}", x);
        }

        channels = 256;
        int interOut = 512;
        for (int level = 0; level < 4; level++)
        {
            x = ResUNetLevel(x, w.IntermediateLevels[level], channels, interOut);
            channels = interOut;
        }
        Trace("after-bottleneck", x);

        channels = 512;
        for (int level = 0; level < 5; level++)
        {
            int decOut = channels / 2;
            x = ConvTranspose2dPyTorch2x(x, w.DecoderLevels[level].UpsampleWeight, channels, decOut);
            x = BatchNorm2d(x, w.DecoderLevels[level].UpsampleBn);
            ReluInPlace(x);
            Trace($"decoder-upsample{level}", x);
            var skip = skips[^1];
            skips.RemoveAt(skips.Count - 1);
            Trace($"decoder-skip{level}", skip);
            x = ConcatChannels(x, decOut, skip, decOut);
            channels = decOut * 2;
            x = ResUNetLevelFromBlocks(x, w.DecoderLevels[level].Conv2Blocks, channels, decOut);
            channels = decOut;
            Trace($"decoder-level{level}", x);
        }

        Trace("after-decoder", x);
        // Final projection: Conv2d(16->3, 3x3, pad 1).
        var final = Conv2dSamePad3x3(x, channels, w.CnnWeight, w.CnnBias, outCh: 3);
        Trace("after-final-cnn", final);

        // final is [3, H=frames, W=melBins]. Reference: transpose {0,2,1,3} on [1,3,frames,melBins]
        // -> [1,frames,3,melBins] -> flatten to [frames, 3*melBins=384] (channel-then-mel order).
        int melBins = RvcRmvpeWeights.MelBins;
        var flat = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            var row = new float[RvcRmvpeWeights.FeatureDim];
            for (int c = 0; c < 3; c++)
                for (int m = 0; m < melBins; m++)
                    row[c * melBins + m] = final[c][f, m];
            flat[f] = row;
        }

        TraceFlat("flat", flat);
        var (fwdSeq, _) = GruUnroll(flat, w.GruForward, reverse: false);
        var (revSeq, _) = GruUnroll(flat, w.GruReverse, reverse: true);
        TraceFlat("gru-fwd", fwdSeq);
        TraceFlat("gru-rev", revSeq);

        var output = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            var combined = new float[2 * RvcRmvpeWeights.GruHiddenDim];
            Array.Copy(fwdSeq[f], 0, combined, 0, RvcRmvpeWeights.GruHiddenDim);
            Array.Copy(revSeq[f], 0, combined, RvcRmvpeWeights.GruHiddenDim, RvcRmvpeWeights.GruHiddenDim);
            var logits = LinearBias(combined, w.FcOutWeight, w.FcOutBias, inDim: combined.Length, outDim: RvcRmvpeWeights.NumPitchClasses);
            for (int c = 0; c < logits.Length; c++) logits[c] = Sigmoid(logits[c]);
            output[f] = logits;
        }
        return output;
    }

    private static readonly bool TraceEnabled = Environment.GetEnvironmentVariable("STINGRAY_RVC_TRACE") == "1";

    private static void Trace(string label, float[][,] x)
    {
        if (!TraceEnabled) return;
        double sum = 0, sumSq = 0; int n = 0;
        foreach (var plane in x)
        {
            int h = plane.GetLength(0), w2 = plane.GetLength(1);
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w2; j++)
                {
                    sum += plane[i, j];
                    sumSq += (double)plane[i, j] * plane[i, j];
                    n++;
                }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[RVC-RMVPE-CS] {label} channels={x.Length} mean={mean:F5} std={std:F5}");
    }

    private static void TraceFlat(string label, float[][] x)
    {
        if (!TraceEnabled) return;
        double sum = 0, sumSq = 0; int n = 0;
        foreach (var row in x)
            foreach (var v in row)
            {
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[RVC-RMVPE-CS] {label} rows={x.Length} dim={x[0].Length} mean={mean:F5} std={std:F5}");
    }

    private static float[][,] ResUNetLevel(float[][,] x, RvcResUNetLevel level, int inChannels, int outChannels)
    {
        int ch = inChannels;
        foreach (var block in level.Blocks)
        {
            x = ResBlock(x, block, ch, outChannels);
            ch = outChannels;
        }
        return x;
    }

    private static float[][,] ResUNetLevelFromBlocks(float[][,] x, RvcResBlock[] blocks, int inChannels, int outChannels)
    {
        int ch = inChannels;
        foreach (var block in blocks)
        {
            x = ResBlock(x, block, ch, outChannels);
            ch = outChannels;
        }
        return x;
    }

    /// <summary>Real RMVPE `conv_block_res` (NOT a standard ResNet BasicBlock -- confirmed against
    /// the reference's own `conv_bn_relu`/`conv_block_res` C++ functions): conv-BN-ReLU applied
    /// TWICE (both conv stages get a ReLU, including the second one), then a plain residual add
    /// with NO ReLU afterward -- the opposite placement from the textbook BasicBlock (which
    /// omits the second ReLU and instead applies one ReLU after the add). Getting this backwards
    /// doesn't crash or resize anything, it just quietly changes the learned feature statistics --
    /// found via a real per-stage mean/std trace against the C++ reference showing divergence
    /// starting at the very first ResUNet level's output.</summary>
    private static float[][,] ResBlock(float[][,] x, RvcResBlock block, int inChannels, int outChannels)
    {
        var h = Conv2dSamePad3x3(x, inChannels, block.Conv0Weight, bias: null, outCh: outChannels);
        h = BatchNorm2d(h, block.Bn1);
        ReluInPlace(h);
        h = Conv2dSamePad3x3(h, outChannels, block.Conv3Weight, bias: null, outCh: outChannels);
        h = BatchNorm2d(h, block.Bn4);
        ReluInPlace(h);

        float[][,] residual;
        if (block.ShortcutWeight is not null)
            residual = Conv2d1x1(x, inChannels, block.ShortcutWeight, block.ShortcutBias!, outCh: outChannels);
        else
            residual = x;

        int height = h[0].GetLength(0), width = h[0].GetLength(1);
        var output = new float[outChannels][,];
        for (int c = 0; c < outChannels; c++)
        {
            output[c] = new float[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    output[c][i, j] = h[c][i, j] + residual[c][i, j];
        }
        return output;
    }

    /// <summary>Real "same" padding (pad=1) 3x3 conv2d, no bias unless provided. Weight real PyTorch layout [outCh, inCh, 3, 3].</summary>
    private static float[][,] Conv2dSamePad3x3(float[][,] x, int inCh, float[] weight, float[]? bias, int outCh)
    {
        int height = x[0].GetLength(0), width = x[0].GetLength(1);
        var output = new float[outCh][,];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var plane = new float[height, width];
            float b = bias is null ? 0f : bias[oc];
            int wBase = oc * inCh * 9;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    float sum = b;
                    for (int ic = 0; ic < inCh; ic++)
                    {
                        int wIcBase = wBase + ic * 9;
                        for (int kh = 0; kh < 3; kh++)
                        {
                            int ih = i - 1 + kh;
                            if ((uint)ih >= (uint)height) continue;
                            for (int kw = 0; kw < 3; kw++)
                            {
                                int iw = j - 1 + kw;
                                if ((uint)iw >= (uint)width) continue;
                                sum += weight[wIcBase + kh * 3 + kw] * x[ic][ih, iw];
                            }
                        }
                    }
                    plane[i, j] = sum;
                }
            }
            output[oc] = plane;
        });
        return output;
    }

    private static float[][,] Conv2d1x1(float[][,] x, int inCh, float[] weight, float[] bias, int outCh)
    {
        int height = x[0].GetLength(0), width = x[0].GetLength(1);
        var output = new float[outCh][,];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var plane = new float[height, width];
            int wBase = oc * inCh;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    float sum = bias[oc];
                    for (int ic = 0; ic < inCh; ic++)
                        sum += weight[wBase + ic] * x[ic][i, j];
                    plane[i, j] = sum;
                }
            output[oc] = plane;
        });
        return output;
    }

    private static float[][,] AvgPool2x2(float[][,] x, int channels)
    {
        int height = x[0].GetLength(0), width = x[0].GetLength(1);
        int outH = height / 2, outW = width / 2;
        var output = new float[channels][,];
        for (int c = 0; c < channels; c++)
        {
            var plane = new float[outH, outW];
            for (int i = 0; i < outH; i++)
                for (int j = 0; j < outW; j++)
                    plane[i, j] = (x[c][2 * i, 2 * j] + x[c][2 * i + 1, 2 * j] + x[c][2 * i, 2 * j + 1] + x[c][2 * i + 1, 2 * j + 1]) * 0.25f;
            output[c] = plane;
        }
        return output;
    }

    /// <summary>Real PyTorch ConvTranspose2d(kernel=2, stride=2), matching the reference's exact "oversized-then-sliced" trick: computes a (2*size+1)-sized full transpose-conv output then keeps only [1, 2*size), producing an exact 2x upsample.</summary>
    private static float[][,] ConvTranspose2dPyTorch2x(float[][,] x, float[] weight, int inCh, int outCh)
    {
        int height = x[0].GetLength(0), width = x[0].GetLength(1);
        int outH = height * 2, outW = width * 2;
        var output = new float[outCh][,];
        for (int oc = 0; oc < outCh; oc++) output[oc] = new float[outH, outW];

        // Real PyTorch ConvTranspose2d weight layout: [inCh, outCh, kh, kw]. Each input pixel
        // (i,j) scatters into a 2x2 output block at (2i, 2j) via the kernel -- this is the
        // standard, mathematically exact ConvTranspose2d(kernel=2, stride=2, pad=0) result, size
        // exactly 2x the input with no extra row/column. The reference computes an oversized
        // (2*size+1) output then slices [1, 2*size) to drop a leading row/column -- that's an
        // artifact specific to GGML's own conv_transpose_2d_p0 kernel implementation, not a
        // property of the real math itself, so this direct from-scratch scatter (which never
        // produces that extra row/column to begin with) does NOT need the same -1 shift.
        for (int ic = 0; ic < inCh; ic++)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    float v = x[ic][i, j];
                    if (v == 0f) continue;
                    for (int oc = 0; oc < outCh; oc++)
                    {
                        int wBase = (ic * outCh + oc) * 4;
                        for (int kh = 0; kh < 2; kh++)
                        {
                            int oh = 2 * i + kh;
                            for (int kw = 0; kw < 2; kw++)
                            {
                                int ow = 2 * j + kw;
                                output[oc][oh, ow] += v * weight[wBase + kh * 2 + kw];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    private static float[][,] ConcatChannels(float[][,] a, int aChannels, float[][,] b, int bChannels)
    {
        var output = new float[aChannels + bChannels][,];
        Array.Copy(a, output, aChannels);
        Array.Copy(b, 0, output, aChannels, bChannels);
        return output;
    }

    private static float[][,] BatchNorm2d(float[][,] x, RvcBatchNorm2d bn, float eps = 1e-5f)
    {
        int channels = x.Length;
        int height = x[0].GetLength(0), width = x[0].GetLength(1);
        var output = new float[channels][,];
        for (int c = 0; c < channels; c++)
        {
            var plane = new float[height, width];
            float invStd = 1f / MathF.Sqrt(bn.RunningVar[c] + eps);
            float scale = bn.Weight[c] * invStd;
            float shift = bn.Bias[c] - bn.RunningMean[c] * scale;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    plane[i, j] = x[c][i, j] * scale + shift;
            output[c] = plane;
        }
        return output;
    }

    private static void ReluInPlace(float[][,] x)
    {
        foreach (var plane in x)
        {
            int h = plane.GetLength(0), w = plane.GetLength(1);
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    if (plane[i, j] < 0f) plane[i, j] = 0f;
        }
    }

    /// <summary>Real, manually-unrolled PyTorch GRU cell (matching the reference exactly, not a black-box RNN op): standard r/z/n gate equations, run forward or in reverse time order.</summary>
    private static (float[][] Sequence, float[] FinalHidden) GruUnroll(float[][] input, RvcGruWeights gw, bool reverse)
    {
        int frames = input.Length;
        int hiddenDim = RvcRmvpeWeights.GruHiddenDim;
        var hidden = new float[hiddenDim];
        var sequence = new float[frames][];

        for (int step = 0; step < frames; step++)
        {
            int t = reverse ? frames - 1 - step : step;
            var xi = LinearBias(input[t], gw.WeightIh, gw.BiasIh, inDim: input[t].Length, outDim: 3 * hiddenDim);
            var hi = LinearBias(hidden, gw.WeightHh, gw.BiasHh, inDim: hiddenDim, outDim: 3 * hiddenDim);

            var next = new float[hiddenDim];
            for (int d = 0; d < hiddenDim; d++)
            {
                float r = Sigmoid(xi[d] + hi[d]);
                float z = Sigmoid(xi[hiddenDim + d] + hi[hiddenDim + d]);
                float n = MathF.Tanh(xi[2 * hiddenDim + d] + r * hi[2 * hiddenDim + d]);
                next[d] = n + z * (hidden[d] - n);
            }
            hidden = next;
            sequence[t] = hidden;
        }
        return (sequence, hidden);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] LinearBias(float[] input, float[] weight, float[] bias, int inDim, int outDim)
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
}
