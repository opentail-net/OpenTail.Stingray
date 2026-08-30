
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 vocoder forward (`HifiganGenerator.forward`, real FiLM-style global speaker
/// conditioning). Reuses <see cref="HifiGanKernels"/> (same shared conv/upsample primitives
/// already used by Piper/MeloTTS/CosyVoice/MMS-TTS) for the conv/upsample math, and the same
/// ResBlock1 topology already confirmed and implemented for MMS-TTS
/// (<see cref="MmsTts.MmsTtsHifiGanDecoder"/>) -- only the FiLM conditioning is new here.
/// </summary>
public static class XttsVocoder
{
    private const float LeakyReluAlpha = 0.1f;

    /// <summary>x is channel-first [1024, T] (real GPT-trunk latents, already upsampled to the vocoder's real input rate -- see <see cref="XttsHifiDecoder"/>'s doc comment for that preprocessing). speakerEmbedding is the real 512-dim d-vector (global, not per-timestep -- from <see cref="XttsResNetEncoder"/>). Returns mono waveform samples.</summary>
    public static float[] Forward(XttsVocoderWeights w, float[] x, int t, float[] speakerEmbedding)
    {
        var o = HifiGanKernels.Conv1dSamePad(x, XttsVocoderWeights.InChannels, t, w.ConvPreWeight, w.ConvPreBias, XttsVocoderWeights.UpsampleInitialChannel, kernel: 7);

        // Real `o = o + cond_layer(g)`: g is a single [512] vector (T=1), cond_layer's Conv1d
        // output is also effectively a single [upsample_initial_channel] vector, broadcast-added
        // to every timestep of `o` (PyTorch's standard T=1 broadcast).
        var condOnce = LinearVec(speakerEmbedding, w.CondLayerWeight, w.CondLayerBias, XttsVocoderWeights.UpsampleInitialChannel);
        AddBroadcastInPlace(o, XttsVocoderWeights.UpsampleInitialChannel, t, condOnce);

        int ch = XttsVocoderWeights.UpsampleInitialChannel;
        int numStages = XttsVocoderWeights.UpsampleRates.Length;
        int numKernels = XttsVocoderWeights.ResblockKernelSizes.Length;

        for (int stage = 0; stage < numStages; stage++)
        {
            for (int i = 0; i < o.Length; i++) o[i] = HifiGanKernels.LeakyRelu(o[i], LeakyReluAlpha);

            int outCh = ch / 2;
            int newT = t * XttsVocoderWeights.UpsampleRates[stage];
            o = HifiGanKernels.ConvTranspose1d(o, ch, t, w.UpsWeight[stage], w.UpsBias[stage], outCh, XttsVocoderWeights.UpsampleKernelSizes[stage], XttsVocoderWeights.UpsampleRates[stage]);
            ch = outCh;
            t = newT;

            // Real `cond_in_each_up_layer=True`: add this stage's own speaker projection, same broadcast-over-time semantics as cond_layer above.
            var condStage = LinearVec(speakerEmbedding, w.CondsWeight[stage], w.CondsBias[stage], ch);
            AddBroadcastInPlace(o, ch, t, condStage);

            float[]? sum = null;
            for (int k = 0; k < numKernels; k++)
            {
                int rbIndex = stage * numKernels + k;
                var rbOut = ResBlock1Forward(o, ch, t, w.ResBlocks[rbIndex], XttsVocoderWeights.ResblockKernelSizes[k]);
                if (sum is null) sum = rbOut;
                else for (int i = 0; i < sum.Length; i++) sum[i] += rbOut[i];
            }
            for (int i = 0; i < sum!.Length; i++) sum[i] /= numKernels;
            o = sum;
        }

        for (int i = 0; i < o.Length; i++) o[i] = HifiGanKernels.LeakyRelu(o[i], LeakyReluAlpha);
        // Real conv_post: no bias (conv_post_bias=False).
        var post = HifiGanKernels.Conv1dSamePad(o, ch, t, w.ConvPostWeight, null, XttsVocoderWeights.OutChannels, kernel: 7);
        for (int i = 0; i < post.Length; i++) post[i] = MathF.Tanh(post[i]);
        return post;
    }

    /// <summary>Real ResBlock1: for j in 0,1,2: xt=leaky_relu(x); xt=convs1[j](xt)[dilation=(1,3,5)[j]]; xt=leaky_relu(xt); xt=convs2[j](xt)[dilation=1]; x=xt+x.</summary>
    private static float[] ResBlock1Forward(float[] input, int ch, int t, XttsVocResBlockWeights rb, int kernel)
    {
        var x = input;
        Span<int> dilations = [1, 3, 5];
        for (int j = 0; j < 3; j++)
        {
            int dilation = dilations[j];
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

    private static float[] LinearVec(float[] x, float[] weight, float[] bias, int outDim)
    {
        int inDim = x.Length;
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * x[i];
            output[o] = sum;
        }
        return output;
    }

    private static void AddBroadcastInPlace(float[] x, int ch, int t, float[] addPerChannel)
    {
        for (int c = 0; c < ch; c++)
        {
            float v = addPerChannel[c];
            int baseIdx = c * t;
            for (int ti = 0; ti < t; ti++) x[baseIdx + ti] += v;
        }
    }
}
