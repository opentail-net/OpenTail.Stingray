using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared front-end kernels of the wav2vec2 family (HF <c>Wav2Vec2ForCTC</c> in <see cref="Wav2Vec2.Wav2Vec2CtcModel"/>,
/// and RVC's HuBERT content encoder in <see cref="Rvc.RvcHubertEncoder"/>): the conv feature extractor's valid conv1d
/// and per-channel GroupNorm, and the weight-normalized grouped positional conv. Activations are frame-major
/// <c>[t, channels]</c>; convolutions run as im2col + <see cref="PackedLinearF32"/> GEMMs with the PyTorch
/// <c>[out, in, kernel]</c> weight used as is (im2col row order <c>[c0k0..c0kK, c1k0..]</c>).
/// </summary>
public static class Wav2Vec2FrontendKernels
{
    /// <summary>Valid (unpadded) conv1d: <c>[t, inCh]</c> → <c>[tOut, outCh]</c>, tOut = (t − kernel) / stride + 1.</summary>
    public static float[] Conv1dValid(float[] x, int t, int inCh, int kernel, int stride, PackedLinearF32 w, out int tOut)
    {
        tOut = (t - kernel) / stride + 1;
        int kIn = inCh * kernel, to = tOut;
        var cols = new float[(long)tOut * kIn];
        Parallel.For(0, to, o =>
        {
            var row = cols.AsSpan(o * kIn, kIn);
            int start = o * stride;
            for (int c = 0; c < inCh; c++)
                for (int j = 0; j < kernel; j++) row[c * kernel + j] = x[(start + j) * inCh + c];
        });
        var y = new float[(long)tOut * w.OutDim];
        w.Forward(cols, y, tOut);
        return y;
    }

    /// <summary>GroupNorm with num_groups == channels (each channel normalized over time), eps 1e-5, affine.</summary>
    public static void GroupNormPerChannel(float[] x, int t, int ch, float[] w, float[] b, float eps = 1e-5f)
    {
        Parallel.For(0, ch, c =>
        {
            double sum = 0, sq = 0;
            for (int i = 0; i < t; i++) sum += x[i * ch + c];
            double mean = sum / t;
            for (int i = 0; i < t; i++) { double d = x[i * ch + c] - mean; sq += d * d; }
            float inv = (float)(1.0 / Math.Sqrt(sq / t + eps));
            for (int i = 0; i < t; i++) x[i * ch + c] = (float)(x[i * ch + c] - mean) * inv * w[c] + b[c];
        });
    }

    /// <summary>Packs a grouped conv weight <c>[channels, channels/groups, kernel]</c> into one GEMM per group.</summary>
    public static PackedLinearF32[] PackGroupedConv(float[] weight, int channels, int groups, int kernel)
    {
        int gc = channels / groups, per = gc * gc * kernel;
        var packed = new PackedLinearF32[groups];
        for (int g = 0; g < groups; g++)
            packed[g] = new PackedLinearF32(weight.AsSpan(g * per, per).ToArray(), null, gc, gc * kernel);
        return packed;
    }

    /// <summary>
    /// The positional conv (<c>Wav2Vec2PositionalConvEmbedding</c> before its GELU): grouped conv1d, padding kernel/2 on
    /// both sides, trailing frame dropped for an even kernel (<c>Wav2Vec2SamePadLayer</c>), plus bias. <c>[t, h]</c> → <c>[t, h]</c>.
    /// </summary>
    public static float[] GroupedConvSamePad(float[] hs, int t, int h, PackedLinearF32[] groups, float[] bias, int kernel)
    {
        int ng = groups.Length, gc = h / ng, pad = kernel / 2, kIn = gc * kernel;
        var outp = new float[t * h];
        var cols = new float[(long)t * kIn];
        var y = new float[t * gc];
        for (int g = 0; g < ng; g++)
        {
            int g0 = g * gc;
            Parallel.For(0, t, o =>
            {
                var row = cols.AsSpan(o * kIn, kIn);
                for (int c = 0; c < gc; c++)
                    for (int j = 0; j < kernel; j++)
                    {
                        int src = o + j - pad;
                        row[c * kernel + j] = src < 0 || src >= t ? 0f : hs[src * h + g0 + c];
                    }
            });
            groups[g].Forward(cols, y, t);
            for (int o = 0; o < t; o++) y.AsSpan(o * gc, gc).CopyTo(outp.AsSpan(o * h + g0, gc));
        }
        for (int o = 0; o < t; o++) TensorPrimitives.Add(outp.AsSpan(o * h, h), bias, outp.AsSpan(o * h, h));
        return outp;
    }
}
