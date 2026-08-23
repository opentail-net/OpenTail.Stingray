using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Real encoder weights pulled from a <see cref="WhisperGgmlModel"/>, using whisper.cpp's
/// tensor naming scheme (examples/whisper.cpp/src/whisper-arch.h, ASR_SYSTEM_ENCODER).
/// The big per-layer attention/MLP matrices are wrapped as <see cref="WhisperLinearWeight"/>,
/// which dispatches to a real hardware F16C kernel when available (see its own doc comment and
/// docs/audio-review-progress.md's ggml/F16C investigation) instead of always going through the
/// eagerly-dequantized float32 copy. Small vectors (biases, LayerNorm weight/bias, positional
/// embedding, conv weights) stay plain float32.
/// </summary>
public sealed class WhisperEncoderWeights
{
    public float[] PositionalEmbedding { get; }
    public float[] Conv1Weight { get; }   // [dModel, numMels, 3]
    public float[] Conv1Bias { get; }     // [dModel]
    public float[] Conv2Weight { get; }   // [dModel, dModel, 3]
    public float[] Conv2Bias { get; }     // [dModel]
    public float[] LnPostWeight { get; }
    public float[] LnPostBias { get; }
    public WhisperEncoderLayerWeights[] Layers { get; }

    public WhisperEncoderWeights(WhisperGgmlModel model)
    {
        PositionalEmbedding = model.GetTensor("encoder.positional_embedding");
        Conv1Weight = model.GetTensor("encoder.conv1.weight");
        Conv1Bias = model.GetTensor("encoder.conv1.bias");
        Conv2Weight = model.GetTensor("encoder.conv2.weight");
        Conv2Bias = model.GetTensor("encoder.conv2.bias");
        LnPostWeight = model.GetTensor("encoder.ln_post.weight");
        LnPostBias = model.GetTensor("encoder.ln_post.bias");

        int dModel = Conv2Bias.Length;

        Layers = new WhisperEncoderLayerWeights[model.AudioLayer];
        for (int i = 0; i < Layers.Length; i++)
        {
            string qName = $"encoder.blocks.{i}.attn.query.weight";
            string kName = $"encoder.blocks.{i}.attn.key.weight";
            string vName = $"encoder.blocks.{i}.attn.value.weight";
            string oName = $"encoder.blocks.{i}.attn.out.weight";
            string m0Name = $"encoder.blocks.{i}.mlp.0.weight";
            string m2Name = $"encoder.blocks.{i}.mlp.2.weight";

            Layers[i] = new WhisperEncoderLayerWeights
            {
                AttnLnWeight = model.GetTensor($"encoder.blocks.{i}.attn_ln.weight"),
                AttnLnBias = model.GetTensor($"encoder.blocks.{i}.attn_ln.bias"),
                QueryWeight = WhisperLinearWeight.FromTensor(model, qName, dModel, dModel, model.GetTensor(qName)),
                QueryBias = model.GetTensor($"encoder.blocks.{i}.attn.query.bias"),
                KeyWeight = WhisperLinearWeight.FromTensor(model, kName, dModel, dModel, model.GetTensor(kName)),
                ValueWeight = WhisperLinearWeight.FromTensor(model, vName, dModel, dModel, model.GetTensor(vName)),
                ValueBias = model.GetTensor($"encoder.blocks.{i}.attn.value.bias"),
                OutWeight = WhisperLinearWeight.FromTensor(model, oName, dModel, dModel, model.GetTensor(oName)),
                OutBias = model.GetTensor($"encoder.blocks.{i}.attn.out.bias"),
                MlpLnWeight = model.GetTensor($"encoder.blocks.{i}.mlp_ln.weight"),
                MlpLnBias = model.GetTensor($"encoder.blocks.{i}.mlp_ln.bias"),
                Mlp0Weight = WhisperLinearWeight.FromTensor(model, m0Name, dModel * 4, dModel, model.GetTensor(m0Name)),
                Mlp0Bias = model.GetTensor($"encoder.blocks.{i}.mlp.0.bias"),
                Mlp2Weight = WhisperLinearWeight.FromTensor(model, m2Name, dModel, dModel * 4, model.GetTensor(m2Name)),
                Mlp2Bias = model.GetTensor($"encoder.blocks.{i}.mlp.2.bias"),
            };
        }
    }
}

public sealed class WhisperEncoderLayerWeights
{
    public required float[] AttnLnWeight;
    public required float[] AttnLnBias;
    public required WhisperLinearWeight QueryWeight;
    public required float[] QueryBias;
    public required WhisperLinearWeight KeyWeight;   // no bias (matches OpenAI Whisper: k_proj bias=False)
    public required WhisperLinearWeight ValueWeight;
    public required float[] ValueBias;
    public required WhisperLinearWeight OutWeight;
    public required float[] OutBias;
    public required float[] MlpLnWeight;
    public required float[] MlpLnBias;
    public required WhisperLinearWeight Mlp0Weight;  // [4*dModel, dModel]
    public required float[] Mlp0Bias;
    public required WhisperLinearWeight Mlp2Weight;  // [dModel, 4*dModel]
    public required float[] Mlp2Bias;
}
