
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
    /// <summary>z is the flow's output, channel-first [192, T]. g is the speaker embedding [GinChannels]. Returns mono waveform samples.</summary>
    public static unsafe float[] Forward(MeloOnnxWeights w, float[] z, int t, float[] g)
    {
        int ch = 512;
        var x = HifiGanKernels.Conv1dSamePad(z, w.HiddenDim, t, w.DecConvPreWeight, w.DecConvPreBias, ch, kernel: 7);

        // dec.cond: standard Conv1d(gin_channels, upsample_initial_channel, 1) -- [out,in] layout
        int gin = w.GinChannels;
        var cond = new float[ch];
        fixed (float* wP = w.DecCondWeight, gP = g, cP = cond)
        {
            SimdKernels.MatVecF32(cP, wP, gP, ch, gin);
        }
        System.Numerics.Tensors.TensorPrimitives.Add(cond, w.DecCondBias, cond);
        for (int c = 0; c < ch; c++)
        {
            var span = x.AsSpan(c * t, t);
            System.Numerics.Tensors.TensorPrimitives.Add(span, cond[c], span);
        }

        for (int stage = 0; stage < 5; stage++)
        {
            LeakyReluInPlace(x, LeakyReluAlpha);

            int outCh = ch / 2;
            int newT = t * UpStrides[stage];
            x = HifiGanKernels.ConvTranspose1d(x, ch, t, w.DecUpsWeight[stage], w.DecUpsBias[stage], outCh, UpKernels[stage], UpStrides[stage]);
            ch = outCh;
            t = newT;

            float[] rbOut0 = null!, rbOut1 = null!, rbOut2 = null!;
            Parallel.Invoke(
                () => rbOut0 = ResBlock1Forward(x, ch, t, w.DecResblocks[stage * 3 + 0], ResblockKernels[0]),
                () => rbOut1 = ResBlock1Forward(x, ch, t, w.DecResblocks[stage * 3 + 1], ResblockKernels[1]),
                () => rbOut2 = ResBlock1Forward(x, ch, t, w.DecResblocks[stage * 3 + 2], ResblockKernels[2])
            );

            System.Numerics.Tensors.TensorPrimitives.Add(rbOut0, rbOut1, rbOut0);
            System.Numerics.Tensors.TensorPrimitives.Add(rbOut0, rbOut2, rbOut0);
            System.Numerics.Tensors.TensorPrimitives.Multiply(rbOut0, 1f / 3f, rbOut0);
            x = rbOut0;
        }

        LeakyReluInPlace(x, LeakyReluAlpha);
        var post = HifiGanKernels.Conv1dSamePad(x, ch, t, w.DecConvPostWeight, null, 1, kernel: 7);
        System.Numerics.Tensors.TensorPrimitives.Tanh(post, post);
        return post;
    }

    /// <summary>
    /// ResBlock1.forward: for j in 0,1,2: xt = leaky_relu(x); xt = convs1[j](xt) [dilation =
    /// ResblockDilations[j] = 1,3,5]; xt = leaky_relu(xt); xt = convs2[j](xt) [dilation=1]; x = xt
    /// + x.
    /// </summary>
    private static float[] ResBlock1Forward(float[] input, int ch, int t, MeloResBlockWeights rb, int kernel)
    {
        var x = (float[])input.Clone();
        var xt = new float[ch * t];
        for (int j = 0; j < 3; j++)
        {
            int dilation = ResblockDilations[j];
            Array.Copy(x, xt, x.Length);
            LeakyReluInPlace(xt, LeakyReluAlpha);
            var c1 = HifiGanKernels.Conv1dDilated(xt, ch, t, rb.Convs1Weight[j], rb.Convs1Bias[j], ch, kernel, dilation);
            LeakyReluInPlace(c1, LeakyReluAlpha);
            var c2 = HifiGanKernels.Conv1dDilated(c1, ch, t, rb.Convs2Weight[j], rb.Convs2Bias[j], ch, kernel, dilation: 1);

            System.Numerics.Tensors.TensorPrimitives.Add(x, c2, x);
        }
        return x;
    }

    private static void LeakyReluInPlace(float[] x, float alpha)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0f) x[i] *= alpha;
    }
}
