namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Real decoder weights pulled from a <see cref="WhisperGgmlModel"/>, using whisper.cpp's tensor
/// naming scheme (examples/whisper.cpp/src/whisper-arch.h, ASR_SYSTEM_DECODER / ASR_SYSTEM_CROSS).
/// The LM head is tied to <see cref="TokenEmbeddingWeight"/> (no separate output projection tensor),
/// matching the original OpenAI Whisper decoder.
/// </summary>
public sealed class WhisperDecoderWeights
{
    public float[] PositionalEmbedding { get; }   // learned, [TextCtx, dModel]
    public float[] TokenEmbeddingWeight { get; }  // [vocab, dModel] — used for embed lookup AND tied LM head matmul
    public float[] LnWeight { get; }
    public float[] LnBias { get; }
    public WhisperDecoderLayerWeights[] Layers { get; }

    public WhisperDecoderWeights(WhisperGgmlModel model)
    {
        PositionalEmbedding = model.GetTensor("decoder.positional_embedding");
        TokenEmbeddingWeight = model.GetTensor("decoder.token_embedding.weight");
        LnWeight = model.GetTensor("decoder.ln.weight");
        LnBias = model.GetTensor("decoder.ln.bias");

        Layers = new WhisperDecoderLayerWeights[model.TextLayer];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = new WhisperDecoderLayerWeights
            {
                AttnLnWeight = model.GetTensor($"decoder.blocks.{i}.attn_ln.weight"),
                AttnLnBias = model.GetTensor($"decoder.blocks.{i}.attn_ln.bias"),
                QueryWeight = model.GetTensor($"decoder.blocks.{i}.attn.query.weight"),
                QueryBias = model.GetTensor($"decoder.blocks.{i}.attn.query.bias"),
                KeyWeight = model.GetTensor($"decoder.blocks.{i}.attn.key.weight"),
                ValueWeight = model.GetTensor($"decoder.blocks.{i}.attn.value.weight"),
                ValueBias = model.GetTensor($"decoder.blocks.{i}.attn.value.bias"),
                OutWeight = model.GetTensor($"decoder.blocks.{i}.attn.out.weight"),
                OutBias = model.GetTensor($"decoder.blocks.{i}.attn.out.bias"),

                CrossAttnLnWeight = model.GetTensor($"decoder.blocks.{i}.cross_attn_ln.weight"),
                CrossAttnLnBias = model.GetTensor($"decoder.blocks.{i}.cross_attn_ln.bias"),
                CrossQueryWeight = model.GetTensor($"decoder.blocks.{i}.cross_attn.query.weight"),
                CrossQueryBias = model.GetTensor($"decoder.blocks.{i}.cross_attn.query.bias"),
                CrossKeyWeight = model.GetTensor($"decoder.blocks.{i}.cross_attn.key.weight"),
                CrossValueWeight = model.GetTensor($"decoder.blocks.{i}.cross_attn.value.weight"),
                CrossValueBias = model.GetTensor($"decoder.blocks.{i}.cross_attn.value.bias"),
                CrossOutWeight = model.GetTensor($"decoder.blocks.{i}.cross_attn.out.weight"),
                CrossOutBias = model.GetTensor($"decoder.blocks.{i}.cross_attn.out.bias"),

                MlpLnWeight = model.GetTensor($"decoder.blocks.{i}.mlp_ln.weight"),
                MlpLnBias = model.GetTensor($"decoder.blocks.{i}.mlp_ln.bias"),
                Mlp0Weight = model.GetTensor($"decoder.blocks.{i}.mlp.0.weight"),
                Mlp0Bias = model.GetTensor($"decoder.blocks.{i}.mlp.0.bias"),
                Mlp2Weight = model.GetTensor($"decoder.blocks.{i}.mlp.2.weight"),
                Mlp2Bias = model.GetTensor($"decoder.blocks.{i}.mlp.2.bias"),
            };
        }
    }
}

public sealed class WhisperDecoderLayerWeights
{
    public required float[] AttnLnWeight;
    public required float[] AttnLnBias;
    public required float[] QueryWeight;
    public required float[] QueryBias;
    public required float[] KeyWeight;        // no bias
    public required float[] ValueWeight;
    public required float[] ValueBias;
    public required float[] OutWeight;
    public required float[] OutBias;

    public required float[] CrossAttnLnWeight;
    public required float[] CrossAttnLnBias;
    public required float[] CrossQueryWeight;
    public required float[] CrossQueryBias;
    public required float[] CrossKeyWeight;   // no bias
    public required float[] CrossValueWeight;
    public required float[] CrossValueBias;
    public required float[] CrossOutWeight;
    public required float[] CrossOutBias;

    public required float[] MlpLnWeight;
    public required float[] MlpLnBias;
    public required float[] Mlp0Weight;       // [4*dModel, dModel]
    public required float[] Mlp0Bias;
    public required float[] Mlp2Weight;       // [dModel, 4*dModel]
    public required float[] Mlp2Bias;
}
