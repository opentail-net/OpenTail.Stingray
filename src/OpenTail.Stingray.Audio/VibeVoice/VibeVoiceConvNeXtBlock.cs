
namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real ConvNeXt-1D causal codec primitives for VibeVoice ASR's speech tokenizer (acoustic +
/// semantic VAE encoders), transcribed directly from
/// `examples/audio.cpp/src/models/vibevoice_asr/speech_tokenizer.cpp`'s `tokenizer_block`/
/// `sconv1d`/`sconv_depthwise1d`/`channel_rms_norm`/`scale_channels` (not guessed) -- despite the
/// generic `mixer_layer` config field name, this is a genuine modern ConvNeXt-1D design (the same
/// family used by newer neural audio codecs), NOT Mamba/SSM. Real causal Conv1d padding: all
/// `(kernel-1)*dilation-(stride-1)` padding goes on the LEFT (causal), plus a real extra RIGHT
/// padding computed to make the output frame count land exactly (the same
/// `ideal_length - length` formula used by Encodec/DAC-family codecs, confirmed via
/// `extra_padding_for_conv1d`). Channels-major layout throughout: `[C][T]` (row per channel).
/// </summary>
public static class VibeVoiceConvNeXtBlock
{
    /// <summary>One real ConvNeXt-1D block: `residual=x; h=ChannelRMSNorm(x) ->
    /// causal DepthwiseConv1d(stride=1) -> per-channel LayerScale(gamma) -> x=residual+h;
    /// residual=x; h=ChannelRMSNorm(x,ffnNorm) -> Linear(dim->4*dim) -> GELU(exact erf) ->
    /// Linear(4*dim->dim) -> per-channel LayerScale(ffnGamma) -> x=residual+h`.</summary>
    public static float[][] Forward(float[][] channelsMajor, VibeVoiceConvNeXtBlockWeights w, float eps)
    {
        int channels = channelsMajor.Length;
        int frames = channelsMajor[0].Length;

        var normed = ChannelRmsNorm(channelsMajor, w.NormWeight, eps);
        var mixed = CausalDepthwiseConv1d(normed, w.MixerWeight, w.MixerBias, kernel: w.MixerWeight[0].Length);
        ScaleChannelsInPlace(mixed, w.Gamma);
        var afterMixer = AddChannelsMajor(channelsMajor, mixed);

        var ffnNormed = ChannelRmsNorm(afterMixer, w.FfnNormWeight, eps);
        // Transpose to frame-major for the pointwise FFN (Linear operates per-frame across channels).
        var frameMajor = Transpose(ffnNormed, channels, frames);
        var hidden = new float[frames][];
        int ffnDim = w.FfnLinear1Weight.Length / channels;
        for (int t = 0; t < frames; t++)
        {
            var h = LinearRow(frameMajor[t], w.FfnLinear1Weight, w.FfnLinear1Bias, channels, ffnDim);
            for (int c = 0; c < ffnDim; c++) h[c] = GeluErf(h[c]);
            hidden[t] = LinearRow(h, w.FfnLinear2Weight, w.FfnLinear2Bias, ffnDim, channels);
        }
        var ffnOut = Transpose(hidden, frames, channels); // back to channels-major
        ScaleChannelsInPlace(ffnOut, w.FfnGamma);
        return AddChannelsMajor(afterMixer, ffnOut);
    }

    /// <summary>Real STREAMING ConvNeXt-1D block, ported from `tokenizer_block_streaming` (not
    /// guessed): identical to <see cref="Forward"/> except the depthwise mixer conv uses
    /// <see cref="SConvDepthwise1dStreaming"/> (real per-call cache) instead of the whole-clip
    /// causal zero-padded version -- the FFN half has no conv, so it's unchanged.</summary>
    public static float[][] ForwardStreaming(float[][] channelsMajor, VibeVoiceConvNeXtBlockWeights w, float eps, ref float[][]? cache)
    {
        int channels = channelsMajor.Length;
        int frames = channelsMajor[0].Length;

        var normed = ChannelRmsNorm(channelsMajor, w.NormWeight, eps);
        var mixed = SConvDepthwise1dStreaming(normed, w.MixerWeight, w.MixerBias, stride: 1, ref cache);
        ScaleChannelsInPlace(mixed, w.Gamma);
        var afterMixer = AddChannelsMajor(channelsMajor, mixed);

        var ffnNormed = ChannelRmsNorm(afterMixer, w.FfnNormWeight, eps);
        var frameMajor = Transpose(ffnNormed, channels, frames);
        var hidden = new float[frames][];
        int ffnDim = w.FfnLinear1Weight.Length / channels;
        for (int t = 0; t < frames; t++)
        {
            var h = LinearRow(frameMajor[t], w.FfnLinear1Weight, w.FfnLinear1Bias, channels, ffnDim);
            for (int c = 0; c < ffnDim; c++) h[c] = GeluErf(h[c]);
            hidden[t] = LinearRow(h, w.FfnLinear2Weight, w.FfnLinear2Bias, ffnDim, channels);
        }
        var ffnOut = Transpose(hidden, frames, channels);
        ScaleChannelsInPlace(ffnOut, w.FfnGamma);
        return AddChannelsMajor(afterMixer, ffnOut);
    }

    /// <summary>Real causal Conv1d (full, not depthwise): weight `[outCh][inCh][kernel]`, causal
    /// left-pad `(kernel-1)*dilation-(stride-1)` plus the real Encodec/DAC-style extra right pad to
    /// land the output frame count exactly, matching `sconv1d`.</summary>
    public static (float[][] Output, int OutFrames) CausalConv1d(float[][] channelsMajorIn, float[][][] weight, float[]? bias, int stride, int dilation = 1)
    {
        int inCh = channelsMajorIn.Length;
        int frames = channelsMajorIn[0].Length;
        int outCh = weight.Length;
        int kernel = weight[0][0].Length;
        int padTotal = (kernel - 1) * dilation - (stride - 1);
        int extra = ExtraPaddingForConv1d(frames, kernel, stride, padTotal);
        var padded = PadFrames(channelsMajorIn, padTotal, extra);
        int paddedFrames = padded[0].Length;
        int outFrames = (paddedFrames - (kernel - 1) * dilation - 1) / stride + 1;

        var output = new float[outCh][];
        for (int oc = 0; oc < outCh; oc++)
        {
            var row = new float[outFrames];
            float b = bias?[oc] ?? 0f;
            for (int t = 0; t < outFrames; t++)
            {
                float sum = b;
                int start = t * stride;
                for (int ic = 0; ic < inCh; ic++)
                {
                    var wRow = weight[oc][ic];
                    var srcRow = padded[ic];
                    for (int k = 0; k < kernel; k++)
                        sum += wRow[k] * srcRow[start + k * dilation];
                }
                row[t] = sum;
            }
            output[oc] = row;
        }
        return (output, outFrames);
    }

    /// <summary>Real causal DepthwiseConv1d (stride=1 only, as used inside every ConvNeXt block):
    /// weight `[channels][kernel]`, same causal+extra padding convention as
    /// <see cref="CausalConv1d"/>.</summary>
    public static float[][] CausalDepthwiseConv1d(float[][] channelsMajor, float[][] weight, float[]? bias, int kernel, int dilation = 1)
    {
        int channels = channelsMajor.Length;
        int frames = channelsMajor[0].Length;
        int padTotal = (kernel - 1) * dilation; // stride=1 -> (stride-1)=0
        int extra = ExtraPaddingForConv1d(frames, kernel, 1, padTotal);
        var padded = PadFrames(channelsMajor, padTotal, extra);
        int paddedFrames = padded[0].Length;
        int outFrames = paddedFrames - (kernel - 1) * dilation;

        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[outFrames];
            float b = bias?[c] ?? 0f;
            var wRow = weight[c];
            var srcRow = padded[c];
            for (int t = 0; t < outFrames; t++)
            {
                float sum = b;
                for (int k = 0; k < kernel; k++)
                    sum += wRow[k] * srcRow[t + k * dilation];
                row[t] = sum;
            }
            output[c] = row;
        }
        return output;
    }

    /// <summary>
    /// Real STREAMING full Conv1d, ported from `tokenizer_audio.cpp`'s `sconv1d_streaming` (not
    /// guessed): unlike <see cref="CausalConv1d"/>'s zero-padding, the causal left-context comes
    /// from a real per-call CACHE (the previous call's own trailing input frames), concatenated
    /// before a plain unpadded (`padding=0`) convolution -- no `ExtraPaddingForConv1d` trick
    /// needed since the cache supplies exact context. `cache` is `null` on the very first call
    /// (real reference: `VibeVoiceTokenizerStreamingState`'s caches lazily zero-initialize on
    /// first use) and is updated in place to the new real cache (the trailing
    /// `contextFrames = (kernel-1)*dilation-(stride-1)` frames of `concat(cache, input)`) for the
    /// next call.</summary>
    public static float[][] SConv1dStreaming(float[][] channelsMajorIn, float[][][] weight, float[]? bias, int stride, ref float[][]? cache, int dilation = 1)
    {
        int inCh = channelsMajorIn.Length;
        int outCh = weight.Length;
        int kernel = weight[0][0].Length;
        int contextFrames = (kernel - 1) * dilation - (stride - 1);
        var full = ConcatFrames(cache ??= ZeroCache(inCh, contextFrames), channelsMajorIn);
        int fullLen = full[0].Length;
        int outFrames = (fullLen - (kernel - 1) * dilation - 1) / stride + 1;

        var output = new float[outCh][];
        for (int oc = 0; oc < outCh; oc++)
        {
            var row = new float[outFrames];
            float b = bias?[oc] ?? 0f;
            for (int t = 0; t < outFrames; t++)
            {
                float sum = b;
                int start = t * stride;
                for (int ic = 0; ic < inCh; ic++)
                {
                    var wRow = weight[oc][ic];
                    var srcRow = full[ic];
                    for (int k = 0; k < kernel; k++)
                        sum += wRow[k] * srcRow[start + k * dilation];
                }
                row[t] = sum;
            }
            output[oc] = row;
        }
        cache = LastFrames(full, contextFrames);
        return output;
    }

    /// <summary>Real STREAMING DepthwiseConv1d, ported from `sconv_depthwise1d_streaming` (not
    /// guessed) -- same cache convention as <see cref="SConv1dStreaming"/>.</summary>
    public static float[][] SConvDepthwise1dStreaming(float[][] channelsMajor, float[][] weight, float[]? bias, int stride, ref float[][]? cache, int dilation = 1)
    {
        int channels = channelsMajor.Length;
        int kernel = weight[0].Length;
        int contextFrames = (kernel - 1) * dilation - (stride - 1);
        var full = ConcatFrames(cache ??= ZeroCache(channels, contextFrames), channelsMajor);
        int fullLen = full[0].Length;
        int outFrames = (fullLen - (kernel - 1) * dilation - 1) / stride + 1;

        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[outFrames];
            float b = bias?[c] ?? 0f;
            var wRow = weight[c];
            var srcRow = full[c];
            for (int t = 0; t < outFrames; t++)
            {
                float sum = b;
                int start = t * stride;
                for (int k = 0; k < kernel; k++)
                    sum += wRow[k] * srcRow[start + k * dilation];
                row[t] = sum;
            }
            output[c] = row;
        }
        cache = LastFrames(full, contextFrames);
        return output;
    }

    /// <summary>Real STREAMING `ConvTranspose1d`, ported from `sconv_transpose1d_streaming` (not
    /// guessed): the previous chunk's real trailing INPUT frames (not output) are prepended before
    /// running the SAME real unpadded transpose-conv + all-from-the-right trim
    /// (`kTokenizerConvTransposeTrimRightRatio=1.0`, matching the non-streaming
    /// `CausalConvTranspose1d`'s real crop convention) that the whole-clip decoder uses, then only
    /// the LAST `inputFrames*stride` samples of the trimmed result are kept (discarding the
    /// portion influenced by the cache's own extra context) -- this is what makes chunk boundaries
    /// causally consistent with a whole-clip decode. `cache` is updated to the trailing
    /// `kernel-1` frames of `concat(cache, input)` (the raw INPUT, matching the regular streaming
    /// convs' cache convention, not the output).</summary>
    public static float[][] SConvTranspose1dStreaming(float[][] input, float[][][] weight, float[] bias, int stride, ref float[][]? cache)
    {
        int inChannels = input.Length;
        int inFrames = input[0].Length;
        int outChannels = weight[0].Length;
        int kernel = weight[0][0].Length;
        int contextFrames = kernel - 1;

        var full = ConcatFrames(cache ??= ZeroCache(inChannels, contextFrames), input);
        int fullInLen = full[0].Length;
        int rawLen = (fullInLen - 1) * stride + kernel;

        var raw = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            raw[oc] = new float[rawLen];
            float b = bias[oc];
            for (int o = 0; o < rawLen; o++) raw[oc][o] = b;
        }
        for (int ic = 0; ic < inChannels; ic++)
        {
            var inRow = full[ic];
            for (int i = 0; i < fullInLen; i++)
            {
                float v = inRow[i];
                if (v == 0f) continue;
                int baseOut = i * stride;
                for (int oc = 0; oc < outChannels; oc++)
                {
                    var wRow = weight[ic][oc];
                    var outRow = raw[oc];
                    for (int k = 0; k < kernel; k++) outRow[baseOut + k] += wRow[k] * v;
                }
            }
        }

        int paddingTotal = kernel - stride;
        if (paddingTotal < 0) throw new InvalidOperationException("VibeVoice streaming tokenizer ConvTranspose1d padding_total is negative.");
        // Real kTokenizerConvTransposeTrimRightRatio=1.0 -> paddingRight=paddingTotal, paddingLeft=0.
        int unpaddedFrames = rawLen - paddingTotal;
        if (unpaddedFrames <= 0) throw new InvalidOperationException("VibeVoice streaming tokenizer ConvTranspose1d unpad removed all frames.");

        int outputFrames = inFrames * stride;
        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            var unpadded = raw[oc]; // paddingLeft=0, so unpadded is just raw[0..unpaddedFrames)
            var row = new float[outputFrames];
            Array.Copy(unpadded, unpaddedFrames - outputFrames, row, 0, outputFrames);
            output[oc] = row;
        }
        cache = LastFrames(full, contextFrames);
        return output;
    }

    private static float[][] ZeroCache(int channels, int frames)
    {
        var cache = new float[channels][];
        for (int c = 0; c < channels; c++) cache[c] = new float[frames];
        return cache;
    }

    private static float[][] ConcatFrames(float[][] a, float[][] b)
    {
        int channels = a.Length;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[a[c].Length + b[c].Length];
            Array.Copy(a[c], 0, row, 0, a[c].Length);
            Array.Copy(b[c], 0, row, a[c].Length, b[c].Length);
            output[c] = row;
        }
        return output;
    }

    private static float[][] LastFrames(float[][] channelsMajor, int frames)
    {
        int channels = channelsMajor.Length;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[frames];
            Array.Copy(channelsMajor[c], channelsMajor[c].Length - frames, row, 0, frames);
            output[c] = row;
        }
        return output;
    }

    /// <summary>Real Encodec/DAC-family extra right-padding formula, matching
    /// `extra_padding_for_conv1d` exactly.</summary>
    public static int ExtraPaddingForConv1d(int length, int kernel, int stride, int padTotal)
    {
        double nFrames = ((double)(length - kernel + padTotal) / stride) + 1.0;
        int ceilFrames = (int)Math.Ceiling(nFrames);
        int idealLength = (ceilFrames - 1) * stride + (kernel - padTotal);
        return idealLength - length;
    }

    private static float[][] PadFrames(float[][] channelsMajor, int left, int right)
    {
        int channels = channelsMajor.Length;
        int frames = channelsMajor[0].Length;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[left + frames + right];
            Array.Copy(channelsMajor[c], 0, row, left, frames);
            output[c] = row;
        }
        return output;
    }

    /// <summary>Real channel RMSNorm: normalizes across the CHANNEL axis per frame (not per
    /// channel across time), affine weight only (no bias), matching `channel_rms_norm`'s
    /// transpose-to-[B,T,C]-then-RMSNorm-then-transpose-back convention.</summary>
    internal static float[][] ChannelRmsNorm(float[][] channelsMajor, float[] weight, float eps)
    {
        int channels = channelsMajor.Length;
        int frames = channelsMajor[0].Length;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++) output[c] = new float[frames];

        for (int t = 0; t < frames; t++)
        {
            double sumSq = 0;
            for (int c = 0; c < channels; c++) { float v = channelsMajor[c][t]; sumSq += (double)v * v; }
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / channels + eps));
            for (int c = 0; c < channels; c++)
                output[c][t] = channelsMajor[c][t] * invRms * weight[c];
        }
        return output;
    }

    private static void ScaleChannelsInPlace(float[][] channelsMajor, float[] gamma)
    {
        for (int c = 0; c < channelsMajor.Length; c++)
        {
            var row = channelsMajor[c];
            float g = gamma[c];
            for (int t = 0; t < row.Length; t++) row[t] *= g;
        }
    }

    private static float[][] AddChannelsMajor(float[][] a, float[][] b)
    {
        var output = new float[a.Length][];
        for (int c = 0; c < a.Length; c++)
        {
            var row = new float[a[c].Length];
            for (int t = 0; t < row.Length; t++) row[t] = a[c][t] + b[c][t];
            output[c] = row;
        }
        return output;
    }

    private static float[][] Transpose(float[][] input, int dim0, int dim1)
    {
        var output = new float[dim1][];
        for (int j = 0; j < dim1; j++) output[j] = new float[dim0];
        for (int i = 0; i < dim0; i++)
            for (int j = 0; j < dim1; j++)
                output[j][i] = input[i][j];
        return output;
    }

    private static float[] LinearRow(float[] input, float[] weight, float[]? bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias?[o] ?? 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }

    private static float GeluErf(float x) => 0.5f * x * (1f + Erf(x * 0.70710678f));

    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}

/// <summary>Weights for one <see cref="VibeVoiceConvNeXtBlock"/>. Depthwise mixer weight is
/// `[channels][kernel]` (flattened from the real PyTorch `[channels,1,kernel]` layout); FFN linear
/// weights are real PyTorch `[out,in]` row-major.</summary>
public sealed class VibeVoiceConvNeXtBlockWeights
{
    public float[] NormWeight { get; set; } = [];
    public float[][] MixerWeight { get; set; } = []; // [channels][kernel]
    public float[]? MixerBias { get; set; }
    public float[] Gamma { get; set; } = [];
    public float[] FfnNormWeight { get; set; } = [];
    public float[] FfnLinear1Weight { get; set; } = []; // [4*dim, dim]
    public float[]? FfnLinear1Bias { get; set; }
    public float[] FfnLinear2Weight { get; set; } = []; // [dim, 4*dim]
    public float[]? FfnLinear2Bias { get; set; }
    public float[] FfnGamma { get; set; } = [];
}
