
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>F5-TTS's `TimestepEmbedding`: SinusPositionEmbedding(256) -> Linear(256,1024) -> SiLU -> Linear(1024,1024).</summary>
public static class F5TimestepEmbedding
{
    public static float[] Forward(F5TtsWeights w, float timestep)
    {
        int freqDim = F5TtsWeights.TimeFreqDim;
        int halfDim = freqDim / 2;

        // SinusPositionEmbedding.forward(x, scale=1000): emb = log(10000)/(halfDim-1); freqs =
        // exp(arange(halfDim) * -emb); out = cat(sin(scale*x*freqs), cos(scale*x*freqs)).
        var sinusEmbed = new float[freqDim];
        float embConst = MathF.Log(10000f) / (halfDim - 1);
        for (int k = 0; k < halfDim; k++)
        {
            float freq = MathF.Exp(-k * embConst);
            float angle = 1000f * timestep * freq;
            sinusEmbed[k] = MathF.Sin(angle);
            sinusEmbed[halfDim + k] = MathF.Cos(angle);
        }

        var h = F5Kernels.Linear(sinusEmbed, 1, freqDim, w.TimeMlp0Weight, w.TimeMlp0Bias, F5TtsWeights.HiddenDim);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.SiLU(h[i]);
        return F5Kernels.Linear(h, 1, F5TtsWeights.HiddenDim, w.TimeMlp2Weight, w.TimeMlp2Bias, F5TtsWeights.HiddenDim);
    }
}
