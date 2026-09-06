
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>Real forward pass for OmniVoice's semantic-to-acoustic bridge stage -- see
/// <see cref="OmniVoiceSemanticBridgeWeights"/>'s doc comment for the architecture derivation.
/// Operates on channel-major `[C*T]` flat arrays (index `c*T+t`).</summary>
public static class OmniVoiceSemanticBridge
{
    /// <summary>Input/output both channel-major [768, frames] (no time-axis change -- this stage
    /// is stride-1 throughout).</summary>
    public static float[] Forward(OmniVoiceSemanticBridgeWeights w, float[] xChannelMajor, int frames)
    {
        int dim = OmniVoiceSemanticBridgeWeights.HiddenDim;
        var x = Conv1dSamePad(xChannelMajor, dim, frames, w.ConvWeight, bias: null, outCh: dim, kernel: OmniVoiceSemanticBridgeWeights.KernelSize, dilation: 1);

        foreach (var block in w.Blocks)
        {
            for (int r = 0; r < OmniVoiceSemanticBridgeWeights.ResUnitsPerBlock; r++)
                x = ResidualUnit(x, dim, frames, block.ResConv1[r], block.ResConv2[r], dilation: 1);
            x = Conv1dSamePad(x, dim, frames, block.ConvWeight, block.ConvBias, outCh: dim, kernel: 3, dilation: 1);
        }
        return x;
    }

    private static float[] ResidualUnit(float[] x, int dim, int frames, float[] conv1Weight, float[] conv2Weight, int dilation)
    {
        var h = new float[x.Length];
        for (int i = 0; i < x.Length; i++) h[i] = Elu(x[i]);
        h = Conv1dSamePad(h, dim, frames, conv1Weight, bias: null, outCh: dim, kernel: OmniVoiceSemanticBridgeWeights.UnitKernelSize, dilation: dilation);
        for (int i = 0; i < h.Length; i++) h[i] = Elu(h[i]);
        h = Conv1dSamePad(h, dim, frames, conv2Weight, bias: null, outCh: dim, kernel: 1, dilation: 1);
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] + h[i];
        return output;
    }

    private static float Elu(float x) => x > 0f ? x : MathF.Exp(x) - 1f;

    /// <summary>Real "same"-padding (stride 1) dilated Conv1d over a [C][T] channel-major tensor.
    /// Weight real PyTorch layout [outCh, inCh, kernel].</summary>
    private static float[] Conv1dSamePad(float[] x, int inCh, int frames, float[] weight, float[]? bias, int outCh, int kernel, int dilation)
    {
        int pad = ((kernel - 1) / 2) * dilation;
        var output = new float[outCh * frames];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias is null ? 0f : bias[oc];
            int wBase = oc * inCh * kernel;
            for (int t = 0; t < frames; t++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int it = t - pad + k * dilation;
                        if ((uint)it >= (uint)frames) continue;
                        sum += weight[wIcBase + k] * x[ic * frames + it];
                    }
                }
                output[oc * frames + t] = sum;
            }
        }
        return output;
    }
}
