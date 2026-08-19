namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Real encoder weights pulled from a <see cref="WhisperGgmlModel"/>, using whisper.cpp's
/// tensor naming scheme (examples/whisper.cpp/src/whisper-arch.h, ASR_SYSTEM_ENCODER).
/// All 2D weight tensors keep their file storage order [outFeatures, inFeatures] (row-major,
/// inFeatures contiguous), which is exactly what SimdKernels.MatMulBatchedF32 expects.
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

        Layers = new WhisperEncoderLayerWeights[model.AudioLayer];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = new WhisperEncoderLayerWeights
            {
                AttnLnWeight = model.GetTensor($"encoder.blocks.{i}.attn_ln.weight"),
                AttnLnBias = model.GetTensor($"encoder.blocks.{i}.attn_ln.bias"),
                QueryWeight = model.GetTensor($"encoder.blocks.{i}.attn.query.weight"),
                QueryBias = model.GetTensor($"encoder.blocks.{i}.attn.query.bias"),
                KeyWeight = model.GetTensor($"encoder.blocks.{i}.attn.key.weight"),
                ValueWeight = model.GetTensor($"encoder.blocks.{i}.attn.value.weight"),
                ValueBias = model.GetTensor($"encoder.blocks.{i}.attn.value.bias"),
                OutWeight = model.GetTensor($"encoder.blocks.{i}.attn.out.weight"),
                OutBias = model.GetTensor($"encoder.blocks.{i}.attn.out.bias"),
                MlpLnWeight = model.GetTensor($"encoder.blocks.{i}.mlp_ln.weight"),
                MlpLnBias = model.GetTensor($"encoder.blocks.{i}.mlp_ln.bias"),
                Mlp0Weight = model.GetTensor($"encoder.blocks.{i}.mlp.0.weight"),
                Mlp0Bias = model.GetTensor($"encoder.blocks.{i}.mlp.0.bias"),
                Mlp2Weight = model.GetTensor($"encoder.blocks.{i}.mlp.2.weight"),
                Mlp2Bias = model.GetTensor($"encoder.blocks.{i}.mlp.2.bias"),
            };
        }
    }
}

public sealed class WhisperEncoderLayerWeights
{
    public required float[] AttnLnWeight;
    public required float[] AttnLnBias;
    public required float[] QueryWeight;
    public required float[] QueryBias;
    public required float[] KeyWeight;   // no bias (matches OpenAI Whisper: k_proj bias=False)
    public required float[] ValueWeight;
    public required float[] ValueBias;
    public required float[] OutWeight;
    public required float[] OutBias;
    public required float[] MlpLnWeight;
    public required float[] MlpLnBias;
    public required float[] Mlp0Weight;  // [4*dModel, dModel]
    public required float[] Mlp0Bias;
    public required float[] Mlp2Weight;  // [dModel, 4*dModel]
    public required float[] Mlp2Bias;
}
