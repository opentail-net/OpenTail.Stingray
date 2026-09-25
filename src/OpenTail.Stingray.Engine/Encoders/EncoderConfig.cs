using System.Text.Json;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Dimensions and per-family switches for <see cref="TransformerEncoder"/>, read from a HF
/// <c>config.json</c>. Families: BERT (<c>bert</c>, also ELECTRA with <c>embedding_size == hidden_size</c>)
/// and RoBERTa/XLM-R (<c>roberta</c>, <c>xlm-roberta</c>), whose learned positions start at
/// <c>pad_token_id + 1</c> (HF <c>create_position_ids_from_input_ids</c>; this encoder never runs pad
/// tokens, so every real token gets <c>pad + 1 + index</c>).
/// </summary>
public sealed record EncoderConfig
{
    public required string ModelType { get; init; }
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }
    public required int NumHeads { get; init; }
    public required int IntermediateSize { get; init; }
    public required int VocabSize { get; init; }
    public required int MaxPositions { get; init; }
    public required int TypeVocabSize { get; init; }
    public required float LayerNormEps { get; init; }
    public required string HiddenAct { get; init; }
    public required int PositionOffset { get; init; }

    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Longest input the learned position table can take.</summary>
    public int MaxSequenceLength => MaxPositions - PositionOffset;

    public static EncoderConfig FromFile(string configJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configJsonPath));
        var r = doc.RootElement;
        string type = r.GetProperty("model_type").GetString() ?? "";
        int Int(string name, int fallback) => r.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
        int Req(string name) => r.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetInt32()
            : throw new InvalidDataException($"'{configJsonPath}': missing '{name}'.");

        bool robertaPositions = type is "roberta" or "xlm-roberta" or "camembert";
        if (type is not ("bert" or "electra") && !robertaPositions)
            throw new NotSupportedException($"'{configJsonPath}': encoder model_type '{type}' is not supported yet.");
        if (r.TryGetProperty("position_embedding_type", out var pet) && pet.GetString() is { } p && p != "absolute")
            throw new NotSupportedException($"'{configJsonPath}': position_embedding_type '{p}' is not supported.");
        int hidden = Req("hidden_size");
        if (type == "electra" && Int("embedding_size", hidden) != hidden)
            throw new NotSupportedException($"'{configJsonPath}': ELECTRA with embedding_size != hidden_size needs the embeddings projection.");

        return new EncoderConfig
        {
            ModelType = type,
            HiddenSize = hidden,
            NumLayers = Req("num_hidden_layers"),
            NumHeads = Req("num_attention_heads"),
            IntermediateSize = Req("intermediate_size"),
            VocabSize = Req("vocab_size"),
            MaxPositions = Req("max_position_embeddings"),
            TypeVocabSize = Int("type_vocab_size", 0),
            LayerNormEps = r.TryGetProperty("layer_norm_eps", out var eps) ? eps.GetSingle() : 1e-12f,
            HiddenAct = r.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? "gelu" : "gelu",
            PositionOffset = robertaPositions ? Int("pad_token_id", 1) + 1 : 0,
        };
    }
}
