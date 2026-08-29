
namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Real decoder weights pulled from a <see cref="WhisperGgmlModel"/>, using whisper.cpp's tensor
/// naming scheme (examples/whisper.cpp/src/whisper-arch.h, ASR_SYSTEM_DECODER / ASR_SYSTEM_CROSS).
/// The LM head is tied to <see cref="TokenEmbeddingWeight"/> (no separate output projection tensor),
/// matching the original OpenAI Whisper decoder.
///
/// <para>The big per-layer self/cross-attention and MLP matrices, and the tied LM head, are
/// wrapped as <see cref="WhisperLinearWeight"/> (real hardware F16C kernel when available -- see
/// its own doc comment and docs/audio-review-progress.md's ggml/F16C investigation).
/// <see cref="TokenEmbeddingWeight"/> keeps its plain float32 form too (needed for direct
/// index-based embedding lookup by token id) alongside the <see cref="TokenEmbeddingWeightLinear"/>
/// wrapper used for the tied LM head's output projection.</para>
/// </summary>
public sealed class WhisperDecoderWeights
{
    public float[] PositionalEmbedding { get; }   // learned, [TextCtx, dModel]
    public float[] TokenEmbeddingWeight { get; }  // [vocab, dModel] — used for embed lookup by index
    public WhisperLinearWeight TokenEmbeddingWeightLinear { get; } // same tensor, used for the tied LM head matmul
    public float[] LnWeight { get; }
    public float[] LnBias { get; }
    public WhisperDecoderLayerWeights[] Layers { get; }

    public WhisperDecoderWeights(WhisperGgmlModel model)
    {
        PositionalEmbedding = model.GetTensor("decoder.positional_embedding");
        TokenEmbeddingWeight = model.GetTensor("decoder.token_embedding.weight");
        LnWeight = model.GetTensor("decoder.ln.weight");
        LnBias = model.GetTensor("decoder.ln.bias");

        int dModel = LnBias.Length;
        int vocabSize = TokenEmbeddingWeight.Length / dModel;
        TokenEmbeddingWeightLinear = WhisperLinearWeight.FromTensor(model, "decoder.token_embedding.weight", vocabSize, dModel, TokenEmbeddingWeight);

        Layers = new WhisperDecoderLayerWeights[model.TextLayer];
        for (int i = 0; i < Layers.Length; i++)
        {
            string qName = $"decoder.blocks.{i}.attn.query.weight";
            string kName = $"decoder.blocks.{i}.attn.key.weight";
            string vName = $"decoder.blocks.{i}.attn.value.weight";
            string oName = $"decoder.blocks.{i}.attn.out.weight";
            string cqName = $"decoder.blocks.{i}.cross_attn.query.weight";
            string ckName = $"decoder.blocks.{i}.cross_attn.key.weight";
            string cvName = $"decoder.blocks.{i}.cross_attn.value.weight";
            string coName = $"decoder.blocks.{i}.cross_attn.out.weight";
            string m0Name = $"decoder.blocks.{i}.mlp.0.weight";
            string m2Name = $"decoder.blocks.{i}.mlp.2.weight";

            Layers[i] = new WhisperDecoderLayerWeights
            {
                AttnLnWeight = model.GetTensor($"decoder.blocks.{i}.attn_ln.weight"),
                AttnLnBias = model.GetTensor($"decoder.blocks.{i}.attn_ln.bias"),
                QueryWeight = WhisperLinearWeight.FromTensor(model, qName, dModel, dModel, model.GetTensor(qName)),
                QueryBias = model.GetTensor($"decoder.blocks.{i}.attn.query.bias"),
                KeyWeight = WhisperLinearWeight.FromTensor(model, kName, dModel, dModel, model.GetTensor(kName)),
                ValueWeight = WhisperLinearWeight.FromTensor(model, vName, dModel, dModel, model.GetTensor(vName)),
                ValueBias = model.GetTensor($"decoder.blocks.{i}.attn.value.bias"),
                OutWeight = WhisperLinearWeight.FromTensor(model, oName, dModel, dModel, model.GetTensor(oName)),
                OutBias = model.GetTensor($"decoder.blocks.{i}.attn.out.bias"),

                CrossAttnLnWeight = model.GetTensor($"decoder.blocks.{i}.cross_attn_ln.weight"),
                CrossAttnLnBias = model.GetTensor($"decoder.blocks.{i}.cross_attn_ln.bias"),
                CrossQueryWeight = WhisperLinearWeight.FromTensor(model, cqName, dModel, dModel, model.GetTensor(cqName)),
                CrossQueryBias = model.GetTensor($"decoder.blocks.{i}.cross_attn.query.bias"),
                CrossKeyWeight = WhisperLinearWeight.FromTensor(model, ckName, dModel, dModel, model.GetTensor(ckName)),
                CrossValueWeight = WhisperLinearWeight.FromTensor(model, cvName, dModel, dModel, model.GetTensor(cvName)),
                CrossValueBias = model.GetTensor($"decoder.blocks.{i}.cross_attn.value.bias"),
                CrossOutWeight = WhisperLinearWeight.FromTensor(model, coName, dModel, dModel, model.GetTensor(coName)),
                CrossOutBias = model.GetTensor($"decoder.blocks.{i}.cross_attn.out.bias"),

                MlpLnWeight = model.GetTensor($"decoder.blocks.{i}.mlp_ln.weight"),
                MlpLnBias = model.GetTensor($"decoder.blocks.{i}.mlp_ln.bias"),
                Mlp0Weight = WhisperLinearWeight.FromTensor(model, m0Name, dModel * 4, dModel, model.GetTensor(m0Name)),
                Mlp0Bias = model.GetTensor($"decoder.blocks.{i}.mlp.0.bias"),
                Mlp2Weight = WhisperLinearWeight.FromTensor(model, m2Name, dModel, dModel * 4, model.GetTensor(m2Name)),
                Mlp2Bias = model.GetTensor($"decoder.blocks.{i}.mlp.2.bias"),
            };
        }
    }
}

public sealed class WhisperDecoderLayerWeights
{
    public required float[] AttnLnWeight;
    public required float[] AttnLnBias;
    public required WhisperLinearWeight QueryWeight;
    public required float[] QueryBias;
    public required WhisperLinearWeight KeyWeight;        // no bias
    public required WhisperLinearWeight ValueWeight;
    public required float[] ValueBias;
    public required WhisperLinearWeight OutWeight;
    public required float[] OutBias;

    public required float[] CrossAttnLnWeight;
    public required float[] CrossAttnLnBias;
    public required WhisperLinearWeight CrossQueryWeight;
    public required float[] CrossQueryBias;
    public required WhisperLinearWeight CrossKeyWeight;   // no bias
    public required WhisperLinearWeight CrossValueWeight;
    public required float[] CrossValueBias;
    public required WhisperLinearWeight CrossOutWeight;
    public required float[] CrossOutBias;

    public required float[] MlpLnWeight;
    public required float[] MlpLnBias;
    public required WhisperLinearWeight Mlp0Weight;       // [4*dModel, dModel]
    public required float[] Mlp0Bias;
    public required WhisperLinearWeight Mlp2Weight;       // [dModel, 4*dModel]
    public required float[] Mlp2Bias;
}
