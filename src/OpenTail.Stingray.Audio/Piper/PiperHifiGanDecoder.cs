
namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's HiFi-GAN vocoder (dec), forward pass. Standard HiFi-GAN generator, confirmed via ONNX
/// node/attribute inspection: conv_pre (k=7) -&gt; 3x [LeakyRelu -&gt; ConvTranspose1d upsample -&gt;
/// 3-way-averaged "resblock2"-style residual stack (2 convs per block, dilations [1,2], kernels
/// [3,5,7])] -&gt; LeakyRelu -&gt; conv_post (k=7, no bias) -&gt; Tanh.
/// Upsample stages: stride/kernel (8,16), (8,16), (4,8) -- total upsample factor 256, matching
/// Piper's 22050 Hz / ~86 Hz mel frame rate.
/// </summary>
public static class PiperHifiGanDecoder
{
    private const float LeakyReluAlpha = 0.1f;
    private static readonly int[] ResblockKernels = [3, 5, 7];
    // Confirmed via ONNX node attribute inspection: NOT uniform [1,2] across kernel sizes.
    private static readonly int[][] ResblockDilations = [[1, 2], [2, 6], [3, 12]];
    private static readonly int[] UpKernels = [16, 16, 8];
    private static readonly int[] UpStrides = [8, 8, 4];

    /// <summary>zp is the flow's output, channel-first [192, T]. Returns mono waveform samples.</summary>
    public static float[] Forward(PiperOnnxWeights w, float[] zp, int t)
    {
        int ch = 256;
        var x = HifiGanKernels.Conv1dSamePad(zp, w.HiddenDim, t, w.DecConvPreWeight, w.DecConvPreBias, ch, kernel: 7);

        for (int stage = 0; stage < 3; stage++)
        {
            LeakyReluInPlace(x, LeakyReluAlpha);

            int outCh = ch / 2;
            int newT = t * UpStrides[stage];
            x = HifiGanKernels.ConvTranspose1d(x, ch, t, w.DecUpsWeight[stage], w.DecUpsBias[stage], outCh, UpKernels[stage], UpStrides[stage]);
            ch = outCh;
            t = newT;

            var sum = new float[ch * t];
            for (int k = 0; k < 3; k++)
            {
                int rbIndex = stage * 3 + k;
                var rbOut = ResBlockForward((float[])x.Clone(), ch, t, w.DecResblocks[rbIndex], ResblockKernels[k], ResblockDilations[k]);
                System.Numerics.Tensors.TensorPrimitives.Add(sum, rbOut, sum);
            }
            System.Numerics.Tensors.TensorPrimitives.Multiply(sum, 1f / 3f, sum);
            x = sum;
        }

        LeakyReluInPlace(x, LeakyReluAlpha);
        var post = HifiGanKernels.Conv1dSamePad(x, ch, t, w.DecConvPostWeight, null, 1, kernel: 7);
        System.Numerics.Tensors.TensorPrimitives.Tanh(post, post);
        return post;
    }

    private static float[] ResBlockForward(float[] input, int ch, int t, PiperResBlockWeights rb, int kernel, int[] dilations)
    {
        var x = input;
        for (int layer = 0; layer < 2; layer++)
        {
            var y = (float[])x.Clone();
            LeakyReluInPlace(y, LeakyReluAlpha);
            y = HifiGanKernels.Conv1dDilated(y, ch, t, rb.ConvWeight[layer], rb.ConvBias[layer], ch, kernel, dilations[layer]);
            System.Numerics.Tensors.TensorPrimitives.Add(x, y, x);
        }
        return x;
    }

    private static void LeakyReluInPlace(float[] data, float alpha = 0.1f)
    {
        for (int i = 0; i < data.Length; i++)
            data[i] = data[i] >= 0f ? data[i] : data[i] * alpha;
    }
}
