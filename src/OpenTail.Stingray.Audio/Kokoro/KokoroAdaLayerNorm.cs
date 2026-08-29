
namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// StyleTTS2 AdaLayerNorm (modules.py AdaLayerNorm): a LayerNorm with NO learned per-channel
/// affine of its own -- instead gamma/beta are computed from the style vector via a single
/// Linear(style_dim, channels*2), then applied identically at every timestep (the style
/// vector doesn't vary over time, so gamma/beta are constant across T). Net of the reference's
/// double-transpose bookkeeping (which is a no-op for our fixed rank-3 case, see
/// docs/audio-review-progress.md), this reduces to: per-timestep LayerNorm over the channel
/// dim (eps=1e-5, AdaLayerNorm's own default -- distinct from ALBERT's 1e-12 and distinct from
/// modules.py's other custom LayerNorm class), then affine with `(1+gamma)*x + beta`.
/// </summary>
public static class KokoroAdaLayerNorm
{
    private const float Eps = 1e-5f;

    /// <summary>
    /// input/output are channel-first [channels, T]. fcWeight is `fc.weight` [2*channels, styleDim]
    /// row-major (GGUF reversed-dims convention), fcBias is `fc.bias` [2*channels].
    /// </summary>
    public static float[] Forward(float[] input, float[] style, float[] fcWeight, float[] fcBias, int channels, int styleDim, int t)
    {
        var h = new float[2 * channels];
        for (int o = 0; o < 2 * channels; o++)
        {
            float sum = fcBias[o];
            int wBase = o * styleDim;
            for (int d = 0; d < styleDim; d++)
                sum += fcWeight[wBase + d] * style[d];
            h[o] = sum;
        }
        // gamma = h[0:channels], beta = h[channels:2*channels] (torch.chunk(h,2,dim=1) on the channel axis).
        var output = new float[channels * t];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < channels; c++) mean += input[c * t + ti];
            mean /= channels;

            double variance = 0;
            for (int c = 0; c < channels; c++)
            {
                double d = input[c * t + ti] - mean;
                variance += d * d;
            }
            variance /= channels;
            float invStd = (float)(1.0 / Math.Sqrt(variance + Eps));

            for (int c = 0; c < channels; c++)
            {
                float normed = (float)((input[c * t + ti] - mean) * invStd);
                float gamma = h[c];
                float beta = h[channels + c];
                output[c * t + ti] = (1f + gamma) * normed + beta;
            }
        }
        return output;
    }
}
