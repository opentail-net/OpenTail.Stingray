
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real weights for CosyVoice3's flow-conditioning path: the `input_embedding` speech-token
/// table (6561-entry, real speech-codec vocab -- distinct from the LLM's 6761-entry vocab,
/// which includes ~200 special/stop tokens on top; `6761-6561=200` matches `stop_token_ids`'s
/// real length exactly, a real consistency check, not a coincidence), the `PreLookaheadLayer`
/// convs, and the CamPlus-embedding affine projection (`spk_embed_affine_layer`, 192-&gt;80).
/// Real tensor names/shapes confirmed via `list-tensors` before writing this loader.
/// </summary>
public sealed class CosyVoice3FlowEncoderWeights
{
    public const int SpeechVocabSize = 6561;
    public const int EmbedDim = 80; // == CosyVoice3DiTWeights.MelDim
    public const int PreLookaheadHidden = 1024;
    public const int PreLookaheadLen = 3;
    public const int Conv1Kernel = 4;
    public const int Conv2Kernel = 3;
    public const int TokenMelRatio = 2;
    public const int SpeakerEmbedDim = 192;

    public float[] InputEmbeddingWeight { get; } // [SpeechVocabSize, EmbedDim] real row-major
    public float[] Conv1Weight { get; } // native [1024,80,4] = [outCh,inCh,kernel]
    public float[] Conv1Bias { get; }
    public float[] Conv2Weight { get; } // native [80,1024,3]
    public float[] Conv2Bias { get; }
    public float[] SpkEmbedAffineWeight { get; } // native [80,192]
    public float[] SpkEmbedAffineBias { get; }

    public CosyVoice3FlowEncoderWeights(GgufModel model)
    {
        InputEmbeddingWeight = GetF32(model, "input_embedding.weight");
        Conv1Weight = GetF32(model, "pre_lookahead_layer.conv1.weight");
        Conv1Bias = GetF32(model, "pre_lookahead_layer.conv1.bias");
        Conv2Weight = GetF32(model, "pre_lookahead_layer.conv2.weight");
        Conv2Bias = GetF32(model, "pre_lookahead_layer.conv2.bias");
        SpkEmbedAffineWeight = GetF32(model, "spk_embed_affine_layer.weight");
        SpkEmbedAffineBias = GetF32(model, "spk_embed_affine_layer.bias");
    }

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"CosyVoice3 flow encoder GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}

/// <summary>
/// Real forward for CosyVoice3's flow-conditioning path (`mu`/`spks` inputs to
/// <see cref="CosyVoice3DiTModel"/>), transcribed from `examples/cosyvoice.cpp`'s
/// `CausalMaskedDiffWithDiT::build_cgraph_encode`/`PreLookaheadLayer::build_cgraph`
/// (`cosyvoice-graph.cpp`).
///
/// <para>Real, deliberate simplification flagged: this operates on speech tokens directly (no
/// reference/prompt speech tokens prepended -- the no-zero-shot-cloning case) and produces
/// `spks` from whatever 192-dim speaker vector is passed in. A real CamPlus x-vector extractor
/// is not yet ported in this codebase (`models/campplus.onnx` remains ONNX-only), so callers
/// without one should pass a zero vector -- the affine layer's real bias still contributes a
/// real, non-fabricated speaker-conditioning offset, just not a real per-speaker embedding.</para>
/// </summary>
public static class CosyVoice3FlowEncoder
{
    /// <summary>
    /// Computes `mu` ([2*T, EmbedDim], real 2x nearest-neighbor upsample per
    /// `token_mel_ratio=2`) and `spks` ([EmbedDim], broadcast by the caller across every frame)
    /// from real speech tokens and a real (or zeroed) 192-dim speaker vector.
    /// </summary>
    public static (float[] Mu, float[] Spks) ComputeMuAndSpks(CosyVoice3FlowEncoderWeights w, int[] speechTokens, float[] speakerVector192)
    {
        int t = speechTokens.Length;
        var tokenEmb = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[CosyVoice3FlowEncoderWeights.EmbedDim];
            int baseIdx = speechTokens[i] * CosyVoice3FlowEncoderWeights.EmbedDim;
            System.Array.Copy(w.InputEmbeddingWeight, baseIdx, row, 0, CosyVoice3FlowEncoderWeights.EmbedDim);
            tokenEmb[i] = row;
        }

        var h = PreLookaheadLayer(w, tokenEmb);

        // Real 2x nearest-neighbor upsample (ggml_repeat_4d + reshape): mu[2t] = mu[2t+1] = h[t].
        int ratio = CosyVoice3FlowEncoderWeights.TokenMelRatio;
        var mu = new float[t * ratio * CosyVoice3FlowEncoderWeights.EmbedDim];
        for (int i = 0; i < t; i++)
            for (int r = 0; r < ratio; r++)
                System.Array.Copy(h[i], 0, mu, (i * ratio + r) * CosyVoice3FlowEncoderWeights.EmbedDim, CosyVoice3FlowEncoderWeights.EmbedDim);

        var l2 = L2Normalize(speakerVector192);
        var spks = Linear(l2, w.SpkEmbedAffineWeight, w.SpkEmbedAffineBias, CosyVoice3FlowEncoderWeights.SpeakerEmbedDim, CosyVoice3FlowEncoderWeights.EmbedDim);

        return (mu, spks);
    }

    /// <summary>Real PreLookaheadLayer: right-pad(T,pre_lookahead_len) -&gt; conv1(80-&gt;1024,k=4,valid) -&gt; LeakyReLU(0.01) -&gt; left-pad(causal,k=3) -&gt; conv2(1024-&gt;80,k=3,valid) -&gt; +residual(original token embeddings).</summary>
    private static float[][] PreLookaheadLayer(CosyVoice3FlowEncoderWeights w, float[][] tokenEmb)
    {
        int t = tokenEmb.Length;
        int padded = t + CosyVoice3FlowEncoderWeights.PreLookaheadLen;
        var padRight = new float[padded][];
        for (int i = 0; i < t; i++) padRight[i] = tokenEmb[i];
        for (int i = t; i < padded; i++) padRight[i] = new float[CosyVoice3FlowEncoderWeights.EmbedDim];

        var c1 = ValidConv1d(padRight, w.Conv1Weight, w.Conv1Bias, inCh: CosyVoice3FlowEncoderWeights.EmbedDim, outCh: CosyVoice3FlowEncoderWeights.PreLookaheadHidden, kernel: CosyVoice3FlowEncoderWeights.Conv1Kernel);
        for (int i = 0; i < c1.Length; i++)
            for (int c = 0; c < c1[i].Length; c++)
                c1[i][c] = c1[i][c] >= 0f ? c1[i][c] : c1[i][c] * 0.01f;

        int padLeft = CosyVoice3FlowEncoderWeights.Conv2Kernel - 1;
        var c1PadLeft = new float[c1.Length + padLeft][];
        for (int i = 0; i < padLeft; i++) c1PadLeft[i] = new float[CosyVoice3FlowEncoderWeights.PreLookaheadHidden];
        for (int i = 0; i < c1.Length; i++) c1PadLeft[padLeft + i] = c1[i];

        var c2 = ValidConv1d(c1PadLeft, w.Conv2Weight, w.Conv2Bias, inCh: CosyVoice3FlowEncoderWeights.PreLookaheadHidden, outCh: CosyVoice3FlowEncoderWeights.EmbedDim, kernel: CosyVoice3FlowEncoderWeights.Conv2Kernel);

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[CosyVoice3FlowEncoderWeights.EmbedDim];
            for (int c = 0; c < row.Length; c++) row[c] = c2[i][c] + tokenEmb[i][c];
            output[i] = row;
        }
        return output;
    }

    private static float[][] ValidConv1d(float[][] input, float[] weight, float[] bias, int inCh, int outCh, int kernel)
    {
        int outT = input.Length - kernel + 1;
        var output = new float[outT][];
        for (int ti = 0; ti < outT; ti++)
        {
            var row = new float[outCh];
            for (int oc = 0; oc < outCh; oc++)
            {
                float sum = bias[oc];
                int wOcBase = oc * inCh * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    var srcRow = input[ti + k];
                    int wBase = wOcBase + k;
                    for (int ic = 0; ic < inCh; ic++)
                        sum += srcRow[ic] * weight[wBase + ic * kernel];
                }
                row[oc] = sum;
            }
            output[ti] = row;
        }
        return output;
    }

    private static float[] L2Normalize(float[] v)
    {
        double sumSq = 0;
        foreach (var x in v) sumSq += (double)x * x;
        float norm = (float)System.Math.Sqrt(sumSq + 1e-6 * 1e-6);
        var result = new float[v.Length];
        for (int i = 0; i < v.Length; i++) result[i] = norm > 1e-12f ? v[i] / norm : 0f;
        return result;
    }

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }
}
