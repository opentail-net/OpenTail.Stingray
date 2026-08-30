
namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// MMS-TTS's HiFi-GAN vocoder (`decoder`/`VitsHifiGan`), single-speaker (no `cond`/speaker
/// conditioning -- `speaker_embedding_size=0` in the real config.json). Same ResBlock1 topology
/// (3 conv PAIRS per resblock, dilations (1,3,5) cycling regardless of the resblock's own kernel
/// size) as <see cref="OpenTail.Stingray.Audio.MeloTTS.MeloGenerator"/>'s confirmed-via-ONNX-
/// inspection `ResBlock1Forward`, confirmed independently here via the real
/// `transformers.models.vits.modeling_vits.HifiGanResidualBlock`/`VitsHifiGan` source
/// (`resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]` in
/// config.json, `upsample_rates=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`,
/// `upsample_initial_channel=512`).
/// </summary>
public static class MmsTtsHifiGanDecoder
{
    private const float LeakyReluAlpha = 0.1f;

    /// <summary>zp is the flow's output, channel-first [192, T]. Returns mono waveform samples.</summary>
    public static float[] Forward(MmsTtsWeights w, float[] zp, int t)
    {
        int numStages = w.UpsampleRates.Length;
        int numKernels = w.ResblockKernelSizes.Length;
        int ch = w.UpsampleInitialChannel;

        var x = HifiGanKernels.Conv1dSamePad(zp, w.HiddenDim, t, w.DecConvPreWeight, w.DecConvPreBias, ch, kernel: 7);

        for (int stage = 0; stage < numStages; stage++)
        {
            LeakyReluInPlace(x, LeakyReluAlpha);

            int outCh = ch / 2;
            int newT = t * w.UpsampleRates[stage];
            x = HifiGanKernels.ConvTranspose1d(x, ch, t, w.DecUpsWeight[stage], w.DecUpsBias[stage], outCh, w.UpsampleKernelSizes[stage], w.UpsampleRates[stage]);
            ch = outCh;
            t = newT;

            var sum = new float[ch * t];
            for (int k = 0; k < numKernels; k++)
            {
                int rbIndex = stage * numKernels + k;
                var rbOut = ResBlock1Forward((float[])x.Clone(), ch, t, w.DecResblocks[rbIndex], w.ResblockKernelSizes[k]);
                System.Numerics.Tensors.TensorPrimitives.Add(sum, rbOut, sum);
            }
            float invKernels = 1f / numKernels;
            System.Numerics.Tensors.TensorPrimitives.Multiply(sum, invKernels, sum);
            x = sum;
        }

        LeakyReluInPlace(x, LeakyReluAlpha);
        var post = HifiGanKernels.Conv1dSamePad(x, ch, t, w.DecConvPostWeight, null, 1, kernel: 7);
        System.Numerics.Tensors.TensorPrimitives.Tanh(post, post);
        return post;
    }

    /// <summary>ResBlock1: for j in 0,1,2: xt = leaky_relu(x); xt = convs1[j](xt) [dilation=(1,3,5)[j]]; xt = leaky_relu(xt); xt = convs2[j](xt) [dilation=1]; x = xt + x.</summary>
    private static float[] ResBlock1Forward(float[] input, int ch, int t, MmsResBlockWeights rb, int kernel)
    {
        var x = input;
        Span<int> dilations = [1, 3, 5];
        for (int j = 0; j < 3; j++)
        {
            int dilation = dilations[j];
            var xt = (float[])x.Clone();
            LeakyReluInPlace(xt, LeakyReluAlpha);
            xt = HifiGanKernels.Conv1dDilated(xt, ch, t, rb.Convs1Weight[j], rb.Convs1Bias[j], ch, kernel, dilation);
            LeakyReluInPlace(xt, LeakyReluAlpha);
            xt = HifiGanKernels.Conv1dDilated(xt, ch, t, rb.Convs2Weight[j], rb.Convs2Bias[j], ch, kernel, dilation: 1);

            System.Numerics.Tensors.TensorPrimitives.Add(x, xt, x);
        }
        return x;
    }

    private static void LeakyReluInPlace(float[] data, float alpha = 0.1f)
    {
        for (int i = 0; i < data.Length; i++)
            data[i] = data[i] >= 0f ? data[i] : data[i] * alpha;
    }
}
