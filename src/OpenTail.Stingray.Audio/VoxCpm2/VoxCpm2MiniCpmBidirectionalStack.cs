namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Shared bidirectional (non-causal) MiniCPM transformer stack, factored out of
/// <see cref="VoxCpm2LocalEncoder"/> since VoxCPM2's local encoder AND its DiT estimator's
/// decoder (`generator.cpp`'s `VoxCPM2DiTEstimatorRuntime`) both call the SAME real
/// `minicpm_transformer(..., is_causal=false)` helper over the SAME real config shape
/// (`hidden_dim=1024`, `ffn_dim=4096`, `num_heads=16`, `num_layers=12`, `kv_channels=128`,
/// `num_key_value_heads=2` -- confirmed identical for both `encoder_config` and `dit_config` in
/// the real checkpoint's `config.json`), just with different learned weights and different input
/// sequence construction. Real NEOX+longrope RoPE (see <see cref="VoxCpm2LocalEncoder"/>'s doc
/// comment for the full derivation) is shared too.
/// </summary>
public static class VoxCpm2MiniCpmBidirectionalStack
{
    public const int HiddenDim = 1024;
    public const int NumHeads = 16;
    public const int NumKvHeads = 2;
    public const int HeadDim = 128;
    public const int FfnDim = 4096;
    public const float RopeTheta = 10000f;
    public const float RmsNormEps = 1e-5f;

    public static unsafe float[][] Run(float[][] hidden, VoxCpm2MiniCpmLayerWeights[] layers, float[] finalNorm, int seqLen)
    {
        int halfDim = HeadDim / 2;
        var cos = new float[seqLen * halfDim];
        var sin = new float[seqLen * halfDim];
        fixed (float* cosPtr = cos, sinPtr = sin, freqPtr = VoxCpm2LocalEncoder.RopeShortFactor)
            SimdKernels.BuildRopeTable(cosPtr, sinPtr, seqLen, HeadDim, RopeTheta, freqPtr);

        foreach (var layer in layers)
            hidden = Layer(hidden, layer, cos, sin, seqLen);

        return RmsNormRows(hidden, finalNorm);
    }

    private static float[][] Layer(float[][] input, VoxCpm2MiniCpmLayerWeights layer, float[] cos, float[] sin, int seqLen)
    {
        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;
        int kvRepeats = NumHeads / NumKvHeads;
        float scale = 1f / MathF.Sqrt(HeadDim);
        int halfDim = HeadDim / 2;

        var normed = RmsNormRows(input, layer.InputNorm);
        var q = new float[seqLen][];
        var k = new float[seqLen][];
        var v = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            q[t] = Linear(normed[t], layer.QProjWeight, bias: [], HiddenDim, qOut);
            k[t] = Linear(normed[t], layer.KProjWeight, bias: [], HiddenDim, kvOut);
            v[t] = Linear(normed[t], layer.VProjWeight, bias: [], HiddenDim, kvOut);
            ApplyRopeNeox(q[t], NumHeads, HeadDim, cos, sin, t, halfDim);
            ApplyRopeNeox(k[t], NumKvHeads, HeadDim, cos, sin, t, halfDim);
        }

        var context = new float[seqLen][];
        for (int ti = 0; ti < seqLen; ti++)
        {
            var ctxOut = new float[qOut];
            for (int h = 0; h < NumHeads; h++)
            {
                int hOff = h * HeadDim;
                int kvHOff = (h / kvRepeats) * HeadDim;
                var scores = new float[seqLen];
                for (int tj = 0; tj < seqLen; tj++)
                {
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[ti][hOff + d] * k[tj][kvHOff + d];
                    scores[tj] = dot * scale;
                }
                Softmax(scores);
                for (int tj = 0; tj < seqLen; tj++)
                {
                    float p = scores[tj];
                    for (int d = 0; d < HeadDim; d++) ctxOut[hOff + d] += p * v[tj][kvHOff + d];
                }
            }
            context[ti] = ctxOut;
        }

        var attnOut = new float[seqLen][];
        var x = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            attnOut[t] = Linear(context[t], layer.OProjWeight, bias: [], qOut, HiddenDim);
            x[t] = Add(input[t], attnOut[t]);
        }

        var ffnNormed = RmsNormRows(x, layer.PostNorm);
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var gate = Linear(ffnNormed[t], layer.GateProjWeight, bias: [], HiddenDim, FfnDim);
            SiluInPlace(gate);
            var up = Linear(ffnNormed[t], layer.UpProjWeight, bias: [], HiddenDim, FfnDim);
            for (int i = 0; i < gate.Length; i++) gate[i] *= up[i];
            var down = Linear(gate, layer.DownProjWeight, bias: [], FfnDim, HiddenDim);
            output[t] = Add(x[t], down);
        }
        return output;
    }

    private static void ApplyRopeNeox(float[] x, int numHeads, int headDim, float[] cos, float[] sin, int position, int halfDim)
    {
        int cosBase = position * halfDim;
        for (int h = 0; h < numHeads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < halfDim; i++)
            {
                float c = cos[cosBase + i];
                float s = sin[cosBase + i];
                int idx0 = hOff + i;
                int idx1 = hOff + i + halfDim;
                float x0 = x[idx0];
                float x1 = x[idx1];
                x[idx0] = x0 * c - x1 * s;
                x[idx1] = x0 * s + x1 * c;
            }
        }
    }

    internal static float[][] RmsNormRows(float[][] rows, float[] weight)
    {
        var output = new float[rows.Length][];
        for (int t = 0; t < rows.Length; t++) output[t] = RmsNorm(rows[t], weight);
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

    private static float[] Add(float[] a, float[] b)
    {
        var output = new float[a.Length];
        for (int i = 0; i < a.Length; i++) output[i] = a[i] + b[i];
        return output;
    }

    internal static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        bool hasBias = bias.Length > 0;
        for (int o = 0; o < outDim; o++)
        {
            float sum = hasBias ? bias[o] : 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }

    private static void Softmax(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    internal static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-v));
        }
    }
}
