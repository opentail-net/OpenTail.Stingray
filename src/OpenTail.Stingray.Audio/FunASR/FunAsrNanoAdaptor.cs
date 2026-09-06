
namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real weight loader + forward pass for Fun-ASR-Nano-2512's audio adaptor/projector
/// (`model.multi_modal_projector.*`/`model.audio_adaptor.blocks.*` in the real packed GGUF),
/// transcribed directly from `examples/audio.cpp/src/models/fun_asr_nano/adaptor.cpp` (not
/// guessed). Real pipeline: `Linear(512-&gt;2048,bias) -&gt; ReLU -&gt; Linear(2048-&gt;1024,bias)` (bridges
/// the SAN-M encoder's 512-dim output into the Qwen3 LLM's 1024-dim embedding space) -&gt; 2 standard
/// (non-SANM) post-LN transformer blocks (`LayerNorm -&gt; QKV+bias -&gt; non-causal MHA -&gt; out_proj+bias
/// -&gt; residual` then `LayerNorm -&gt; Linear+bias -&gt; ReLU -&gt; Linear+bias -&gt; residual`), real config
/// confirmed directly from the real `FunAudioLLM/Fun-ASR-Nano-2512-hf` `config.json`:
/// `adaptor_config.hidden_size=1024` (== `text_config.hidden_size`), `num_attention_heads=8`,
/// `intermediate_size=256` (== `text_config.hidden_size/4`), `num_hidden_layers=2`,
/// `projector_hidden_size=2048`.
/// </summary>
public sealed class FunAsrNanoAdaptorWeights
{
    public const int EncoderDim = FunAsrNanoEncoderWeights.DModel; // 512
    public const int ProjectorHiddenSize = 2048;
    public const int DModel = 1024; // == Qwen3 LLM hidden_size
    public const int AttentionHeads = 8;
    public const int FfnDim = 256; // == DModel/4
    public const int Layers = 2;
    public const float LayerNormEps = 1e-5f;

    public float[] Projector1Weight { get; } // [2048, 512]
    public float[] Projector1Bias { get; }
    public float[] Projector2Weight { get; } // [1024, 2048]
    public float[] Projector2Bias { get; }
    public FunAsrNanoAdaptorLayerWeights[] Layers1 { get; }

    public FunAsrNanoAdaptorWeights(Rvc.RvcPackedTensorSource source)
    {
        const string projectorRoot = "model.multi_modal_projector.";
        Projector1Weight = source.GetTensor(projectorRoot + "linear_1.weight");
        Projector1Bias = source.GetTensor(projectorRoot + "linear_1.bias");
        Projector2Weight = source.GetTensor(projectorRoot + "linear_2.weight");
        Projector2Bias = source.GetTensor(projectorRoot + "linear_2.bias");

        string blockRoot = source.HasTensor("model.audio_adaptor.blocks.0.self_attn.q_proj.weight")
            ? "model.audio_adaptor.blocks."
            : "model.multi_modal_projector.blocks.";

        Layers1 = new FunAsrNanoAdaptorLayerWeights[Layers];
        for (int i = 0; i < Layers; i++)
        {
            string p = $"{blockRoot}{i}";
            Layers1[i] = new FunAsrNanoAdaptorLayerWeights
            {
                AttnNormWeight = source.GetTensor($"{p}.self_attn_layer_norm.weight"),
                AttnNormBias = source.GetTensor($"{p}.self_attn_layer_norm.bias"),
                QWeight = source.GetTensor($"{p}.self_attn.q_proj.weight"),
                QBias = source.GetTensor($"{p}.self_attn.q_proj.bias"),
                KWeight = source.GetTensor($"{p}.self_attn.k_proj.weight"),
                KBias = source.GetTensor($"{p}.self_attn.k_proj.bias"),
                VWeight = source.GetTensor($"{p}.self_attn.v_proj.weight"),
                VBias = source.GetTensor($"{p}.self_attn.v_proj.bias"),
                OutWeight = source.GetTensor($"{p}.self_attn.out_proj.weight"),
                OutBias = source.GetTensor($"{p}.self_attn.out_proj.bias"),
                FinalNormWeight = source.GetTensor($"{p}.final_layer_norm.weight"),
                FinalNormBias = source.GetTensor($"{p}.final_layer_norm.bias"),
                Fc1Weight = source.GetTensor($"{p}.fc1.weight"),
                Fc1Bias = source.GetTensor($"{p}.fc1.bias"),
                Fc2Weight = source.GetTensor($"{p}.fc2.weight"),
                Fc2Bias = source.GetTensor($"{p}.fc2.bias"),
            };
        }
    }
}

public sealed class FunAsrNanoAdaptorLayerWeights
{
    public float[] AttnNormWeight { get; set; } = [];
    public float[] AttnNormBias { get; set; } = [];
    public float[] QWeight { get; set; } = [];
    public float[] QBias { get; set; } = [];
    public float[] KWeight { get; set; } = [];
    public float[] KBias { get; set; } = [];
    public float[] VWeight { get; set; } = [];
    public float[] VBias { get; set; } = [];
    public float[] OutWeight { get; set; } = [];
    public float[] OutBias { get; set; } = [];
    public float[] FinalNormWeight { get; set; } = [];
    public float[] FinalNormBias { get; set; } = [];
    public float[] Fc1Weight { get; set; } = [];
    public float[] Fc1Bias { get; set; } = [];
    public float[] Fc2Weight { get; set; } = [];
    public float[] Fc2Bias { get; set; } = [];
}

public static class FunAsrNanoAdaptor
{
    /// <summary>encoderOutput is per-frame [T][512] from <see cref="FunAsrNanoEncoder"/>. Returns
    /// per-frame audio embeddings [T][1024] ready to splice into the Qwen3 LLM's token embedding
    /// sequence.</summary>
    public static float[][] Forward(FunAsrNanoAdaptorWeights w, float[][] encoderOutput)
    {
        int t = encoderOutput.Length;
        var hidden = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h1 = LinearRow(encoderOutput[i], w.Projector1Weight, w.Projector1Bias, FunAsrNanoAdaptorWeights.EncoderDim, FunAsrNanoAdaptorWeights.ProjectorHiddenSize);
            for (int c = 0; c < h1.Length; c++) if (h1[c] < 0f) h1[c] = 0f;
            hidden[i] = LinearRow(h1, w.Projector2Weight, w.Projector2Bias, FunAsrNanoAdaptorWeights.ProjectorHiddenSize, FunAsrNanoAdaptorWeights.DModel);
        }

        foreach (var layer in w.Layers1)
            hidden = Block(hidden, layer);
        return hidden;
    }

    private static float[][] Block(float[][] input, FunAsrNanoAdaptorLayerWeights w)
    {
        int t = input.Length;
        int dModel = FunAsrNanoAdaptorWeights.DModel;
        int heads = FunAsrNanoAdaptorWeights.AttentionHeads;
        int headDim = dModel / heads;

        var normed = LayerNormRows(input, dModel, w.AttnNormWeight, w.AttnNormBias);
        var q = LinearRows(normed, w.QWeight, w.QBias, dModel, dModel);
        var k = LinearRows(normed, w.KWeight, w.KBias, dModel, dModel);
        var v = LinearRows(normed, w.VWeight, w.VBias, dModel, dModel);

        var attn = SelfAttention(q, k, v, heads, headDim);
        var attnOut = LinearRows(attn, w.OutWeight, w.OutBias, dModel, dModel);

        var afterAttn = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dModel];
            for (int c = 0; c < dModel; c++) row[c] = input[i][c] + attnOut[i][c];
            afterAttn[i] = row;
        }

        var ffnNorm = LayerNormRows(afterAttn, dModel, w.FinalNormWeight, w.FinalNormBias);
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = LinearRow(ffnNorm[i], w.Fc1Weight, w.Fc1Bias, dModel, FunAsrNanoAdaptorWeights.FfnDim);
            for (int c = 0; c < h.Length; c++) if (h[c] < 0f) h[c] = 0f;
            var ffnOut = LinearRow(h, w.Fc2Weight, w.Fc2Bias, FunAsrNanoAdaptorWeights.FfnDim, dModel);
            var row = new float[dModel];
            for (int c = 0; c < dModel; c++) row[c] = afterAttn[i][c] + ffnOut[c];
            output[i] = row;
        }
        return output;
    }

    private static float[][] SelfAttention(float[][] q, float[][] k, float[][] v, int numHeads, int headDim)
    {
        int t = q.Length;
        int modelSize = numHeads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);
        var output = new float[t][];
        for (int i = 0; i < t; i++) output[i] = new float[modelSize];

        var scores = new float[t];
        for (int h = 0; h < numHeads; h++)
        {
            int hBase = h * headDim;
            for (int i = 0; i < t; i++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][hBase + d] * k[j][hBase + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < t; j++) acc += scores[j] * invSum * v[j][hBase + d];
                    output[i][hBase + d] = acc;
                }
            }
        }
        return output;
    }

    private static float[][] LayerNormRows(float[][] xRows, int dim, float[] weight, float[] bias)
    {
        var output = new float[xRows.Length][];
        for (int i = 0; i < xRows.Length; i++)
        {
            var row = xRows[i];
            double mean = 0;
            for (int c = 0; c < dim; c++) mean += row[c];
            mean /= dim;
            double variance = 0;
            for (int c = 0; c < dim; c++) { double d = row[c] - mean; variance += d * d; }
            variance /= dim;
            float invStd = (float)(1.0 / Math.Sqrt(variance + FunAsrNanoAdaptorWeights.LayerNormEps));
            var outRow = new float[dim];
            for (int c = 0; c < dim; c++) outRow[c] = (float)((row[c] - mean) * invStd) * weight[c] + bias[c];
            output[i] = outRow;
        }
        return output;
    }

    private static float[][] LinearRows(float[][] xRows, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[xRows.Length][];
        for (int i = 0; i < xRows.Length; i++) output[i] = LinearRow(xRows[i], weight, bias, inDim, outDim);
        return output;
    }

    private static float[] LinearRow(float[] input, float[] weight, float[] bias, int inDim, int outDim)
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
