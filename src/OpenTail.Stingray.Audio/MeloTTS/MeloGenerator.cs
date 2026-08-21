using System;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// MeloTTS's HiFi-GAN vocoder (`dec`, `models.py`'s `Generator`), 44.1kHz. Structurally
/// DIFFERENT from Piper's `PiperHifiGanDecoder` despite both being "HiFi-GAN": this checkpoint
/// uses the classic `ResBlock1` (`modules.py`) -- 3 conv PAIRS per resblock, each pair at a fixed
/// dilation from (1,3,5) for the first conv of the pair and dilation 1 for the second -- NOT
/// Piper's simpler per-resblock-kernel single-dilation-pair layout. Confirmed via direct ONNX
/// Conv node attribute inspection (`dilations`/`kernel_shape`/`pads` on every `/dec/resblocks.*`
/// node), not assumed from the module name. Low-level conv/upsample primitives ARE shared via
/// <see cref="HifiGanKernels"/> (extracted from Piper's implementation this iteration).
///
/// Topology (confirmed via ONNX inspection, see `MeloOnnxWeights`'s `Dec*` fields doc comment):
/// conv_pre (k=7, 192-&gt;512) + cond(g) (1x1, 256-&gt;512) -&gt; 5x [LeakyReLU -&gt; ConvTranspose1d
/// upsample -&gt; 3-way-averaged ResBlock1 stack] -&gt; LeakyReLU -&gt; conv_post (k=7, 16-&gt;1, no bias)
/// -&gt; Tanh. Upsample kernels [16,16,8,2,2], strides [8,8,2,2,2] (total factor 512).
/// </summary>
public static class MeloGenerator
{
    private const float LeakyReluAlpha = 0.1f;
    private static readonly int[] ResblockKernels = [3, 7, 11];
    // EVERY resblock (regardless of kernel size) internally loops j=0,1,2 with convs1[j] dilation
    // taken from THIS SAME fixed tuple (models.py: `ResBlock1(ch, k, dilation=(1,3,5))` -- the
    // dilation tuple does not vary with the resblock's kernel size k); convs2[j] is always
    // dilation 1. Confirmed via real ONNX Conv node attribute inspection across resblocks 0/1/2
    // (kernel 3/7/11 respectively, all showing dilations [1],[3],[5] for convs1.0/1/2).
    private static readonly int[] ResblockDilations = [1, 3, 5];
    private static readonly int[] UpKernels = [16, 16, 8, 2, 2];
    private static readonly int[] UpStrides = [8, 8, 2, 2, 2];

    /// <summary>z is the flow's output, channel-first [192, T]. g is the speaker embedding [GinChannels]. Returns mono waveform samples.</summary>
    public static float[] Forward(MeloOnnxWeights w, float[] z, int t, float[] g)
    {
        int ch = 512;
        var x = HifiGanKernels.Conv1dSamePad(z, w.HiddenDim, t, w.DecConvPreWeight, w.DecConvPreBias, ch, kernel: 7);

        // dec.cond: standard Conv1d(gin_channels, upsample_initial_channel, 1) -- [out,in] layout
        // (a real Conv1d export, NOT the spk_emb_linear MatMul-transpose quirk seen elsewhere).
        int gin = w.GinChannels;
        var cond = new float[ch];
        for (int o = 0; o < ch; o++)
        {
            float sum = w.DecCondBias[o];
            int wBase = o * gin;
            for (int i = 0; i < gin; i++) sum += w.DecCondWeight[wBase + i] * g[i];
            cond[o] = sum;
        }
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < ch; c++)
                x[c * t + ti] += cond[c];

        for (int stage = 0; stage < 5; stage++)
        {
            for (int i = 0; i < x.Length; i++) x[i] = HifiGanKernels.LeakyRelu(x[i], LeakyReluAlpha);

            int outCh = ch / 2;
            int newT = t * UpStrides[stage];
            x = HifiGanKernels.ConvTranspose1d(x, ch, t, w.DecUpsWeight[stage], w.DecUpsBias[stage], outCh, UpKernels[stage], UpStrides[stage]);
            ch = outCh;
            t = newT;

            float[]? sum = null;
            for (int k = 0; k < 3; k++)
            {
                int rbIndex = stage * 3 + k;
                var rbOut = ResBlock1Forward(x, ch, t, w.DecResblocks[rbIndex], ResblockKernels[k]);
                if (sum is null) sum = rbOut;
                else for (int i = 0; i < sum.Length; i++) sum[i] += rbOut[i];
            }
            for (int i = 0; i < sum!.Length; i++) sum[i] /= 3f;
            x = sum;
        }

        for (int i = 0; i < x.Length; i++) x[i] = HifiGanKernels.LeakyRelu(x[i], LeakyReluAlpha);
        var post = HifiGanKernels.Conv1dSamePad(x, ch, t, w.DecConvPostWeight, null, 1, kernel: 7);
        for (int i = 0; i < post.Length; i++) post[i] = MathF.Tanh(post[i]);
        return post;
    }

    /// <summary>
    /// ResBlock1.forward: for j in 0,1,2: xt = leaky_relu(x); xt = convs1[j](xt) [dilation =
    /// ResblockDilations[j] = 1,3,5]; xt = leaky_relu(xt); xt = convs2[j](xt) [dilation=1]; x = xt
    /// + x. NOT Piper's simpler 2-conv resblock, and NOT indexed by the resblock's own kernel
    /// group `k` -- every resblock reuses the same (1,3,5) dilation tuple internally regardless
    /// of its kernel size (see <see cref="ResblockDilations"/>'s doc comment; an earlier version
    /// of this method took a single `dilation` parameter equal to `ResblockDilations[k]`, which
    /// applied the SAME dilation to all 3 internal j-iterations instead of cycling through
    /// 1,3,5 -- caught via golden bisection: cosine ~1.0 through conv_pre/cond/upsample, then
    /// dropping to ~0.86 at the very first resblock's own output).
    /// </summary>
    private static float[] ResBlock1Forward(float[] input, int ch, int t, MeloResBlockWeights rb, int kernel)
    {
        var x = input;
        for (int j = 0; j < 3; j++)
        {
            int dilation = ResblockDilations[j];
            var xt = new float[ch * t];
            for (int i = 0; i < xt.Length; i++) xt[i] = HifiGanKernels.LeakyRelu(x[i], LeakyReluAlpha);
            xt = HifiGanKernels.Conv1dDilated(xt, ch, t, rb.Convs1Weight[j], rb.Convs1Bias[j], ch, kernel, dilation);
            for (int i = 0; i < xt.Length; i++) xt[i] = HifiGanKernels.LeakyRelu(xt[i], LeakyReluAlpha);
            xt = HifiGanKernels.Conv1dDilated(xt, ch, t, rb.Convs2Weight[j], rb.Convs2Bias[j], ch, kernel, dilation: 1);

            var next = new float[ch * t];
            for (int i = 0; i < next.Length; i++) next[i] = xt[i] + x[i];
            x = next;
        }
        return x;
    }
}
