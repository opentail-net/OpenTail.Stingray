namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real weights for VibeVoice's connector (acoustic OR semantic -- same architecture,
/// different prefix/checkpoint tensors), ported from `connector.cpp`'s `load_connector` (not
/// guessed): `fc1.weight/bias` (`[hidden,input]`/`[hidden]`), `norm.weight` (`[hidden]`, RMSNorm,
/// no bias), `fc2.weight/bias` (`[hidden,hidden]`/`[hidden]`).</summary>
public sealed class VibeVoiceConnectorWeights
{
    public required int InputDim { get; init; }
    public required int HiddenSize { get; init; }
    public required float[] Fc1Weight { get; init; }
    public required float[] Fc1Bias { get; init; }
    public required float[] NormWeight { get; init; }
    public required float[] Fc2Weight { get; init; }
    public required float[] Fc2Bias { get; init; }

    public static VibeVoiceConnectorWeights Load(string prefix, int inputDim, int hiddenSize, Func<string, float[]> get) => new()
    {
        InputDim = inputDim,
        HiddenSize = hiddenSize,
        Fc1Weight = get($"{prefix}.fc1.weight"),
        Fc1Bias = get($"{prefix}.fc1.bias"),
        NormWeight = get($"{prefix}.norm.weight"),
        Fc2Weight = get($"{prefix}.fc2.weight"),
        Fc2Bias = get($"{prefix}.fc2.bias"),
    };
}

/// <summary>
/// Real forward pass for VibeVoice's connector, ported from `connector.cpp`'s
/// `build_vibevoice_connector` (not guessed): `Linear(input-&gt;hidden) -&gt; RMSNorm(no bias,
/// eps=1e-6) -&gt; Linear(hidden-&gt;hidden)`. No activation function anywhere in this block -- a
/// real, deliberately simple bridge between the tokenizer encoders' latent features and the text
/// decoder's embedding space (projects each per-frame latent independently; no cross-frame
/// mixing).
/// </summary>
public static class VibeVoiceConnector
{
    private const float RmsNormEps = 1e-6f;

    /// <summary>Projects `[inputDim][frames]` channel-major latent features into
    /// `[hiddenSize][frames]` embedding space.</summary>
    public static float[][] Project(VibeVoiceConnectorWeights w, float[][] channelMajorFeatures)
    {
        int frames = channelMajorFeatures[0].Length;
        var output = new float[w.HiddenSize][];
        for (int i = 0; i < w.HiddenSize; i++) output[i] = new float[frames];

        for (int t = 0; t < frames; t++)
        {
            var frame = new float[w.InputDim];
            for (int c = 0; c < w.InputDim; c++) frame[c] = channelMajorFeatures[c][t];

            var hidden = Linear(frame, w.Fc1Weight, w.Fc1Bias, w.InputDim, w.HiddenSize);
            hidden = RmsNorm(hidden, w.NormWeight);
            var final = Linear(hidden, w.Fc2Weight, w.Fc2Bias, w.HiddenSize, w.HiddenSize);

            for (int c = 0; c < w.HiddenSize; c++) output[c][t] = final[c];
        }
        return output;
    }

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }

    private static float[] RmsNorm(float[] x, float[] weight)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + RmsNormEps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }
}
