using System.Text.Json;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>Encoder families <see cref="TransformerEncoder"/> implements; they differ in positions,
/// attention bias, projections and FFN, not in the post-LN layer structure.</summary>
public enum EncoderFamily
{
    /// <summary>BERT/ELECTRA and RoBERTa/XLM-R: learned absolute positions, separate q/k/v with biases, GELU FFN.</summary>
    Bert,

    /// <summary>MPNet: RoBERTa-style positions plus a T5-style bucketed relative position bias shared by all layers.</summary>
    MpNet,

    /// <summary>NomicBERT: rotary positions (NeoX halves), fused bias-free Wqkv, SwiGLU FFN without biases.</summary>
    NomicBert,
}

/// <summary>
/// Dimensions and per-family switches for <see cref="TransformerEncoder"/>, read from a HF
/// <c>config.json</c>. RoBERTa/XLM-R/MPNet learned positions start at <c>pad_token_id + 1</c> (HF
/// <c>create_position_ids_from_input_ids</c>; this encoder never runs pad tokens, so every real token gets
/// <c>pad + 1 + index</c>).
/// </summary>
public sealed record EncoderConfig
{
    public required string ModelType { get; init; }
    public required EncoderFamily Family { get; init; }
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

    /// <summary>MPNet <c>relative_attention_num_buckets</c> (0 when unused).</summary>
    public int RelativeAttentionBuckets { get; init; }

    /// <summary>NomicBERT rotary base (<c>rotary_emb_base</c>).</summary>
    public float RopeTheta { get; init; }

    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Longest input the position scheme takes.</summary>
    public int MaxSequenceLength => MaxPositions - PositionOffset;

    public static EncoderConfig FromFile(string configJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configJsonPath));
        var r = doc.RootElement;
        string type = r.GetProperty("model_type").GetString() ?? "";
        int Int(string name, int fallback) => r.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
        int Req(params string[] names)
        {
            foreach (var n in names)
                if (r.TryGetProperty(n, out var v) && v.ValueKind == JsonValueKind.Number) return v.GetInt32();
            throw new InvalidDataException($"'{configJsonPath}': missing '{names[0]}'.");
        }
        bool Bool(string name, bool fallback) => r.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : fallback;
        float Eps() => r.TryGetProperty("layer_norm_eps", out var e) && e.ValueKind == JsonValueKind.Number ? e.GetSingle()
            : r.TryGetProperty("layer_norm_epsilon", out var e2) && e2.ValueKind == JsonValueKind.Number ? e2.GetSingle() : 1e-12f;

        if (type == "nomic_bert")
        {
            if (!(r.TryGetProperty("activation_function", out var af) && af.GetString() == "swiglu"))
                throw new NotSupportedException($"'{configJsonPath}': only the SwiGLU NomicBERT variant is supported.");
            if (Bool("prenorm", false) || Bool("qkv_proj_bias", true) || Bool("mlp_fc1_bias", true) || Bool("mlp_fc2_bias", true)
                || Bool("rotary_emb_interleaved", false) || Bool("parallel_block", false)
                || (r.TryGetProperty("rotary_emb_fraction", out var rf) && rf.GetDouble() != 1.0)
                || (r.TryGetProperty("rotary_scaling_factor", out var rs) && rs.ValueKind != JsonValueKind.Null))
                throw new NotSupportedException($"'{configJsonPath}': NomicBERT variant (prenorm/biases/partial or scaled rotary/parallel block) is not supported.");
            return new EncoderConfig
            {
                ModelType = type,
                Family = EncoderFamily.NomicBert,
                HiddenSize = Req("n_embd", "hidden_size"),
                NumLayers = Req("n_layer", "num_hidden_layers"),
                NumHeads = Req("n_head", "num_attention_heads"),
                IntermediateSize = Req("n_inner", "intermediate_size"),
                VocabSize = Req("vocab_size"),
                MaxPositions = Int("n_positions", Int("max_position_embeddings", 2048)),
                TypeVocabSize = Int("type_vocab_size", 0),
                LayerNormEps = Eps(),
                HiddenAct = "swiglu",
                PositionOffset = 0,
                RopeTheta = r.TryGetProperty("rotary_emb_base", out var rb) && rb.ValueKind == JsonValueKind.Number ? rb.GetSingle() : 10000f,
            };
        }

        bool mpnet = type == "mpnet";
        bool robertaPositions = mpnet || type is "roberta" or "xlm-roberta" or "camembert";
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
            Family = mpnet ? EncoderFamily.MpNet : EncoderFamily.Bert,
            HiddenSize = hidden,
            NumLayers = Req("num_hidden_layers"),
            NumHeads = Req("num_attention_heads"),
            IntermediateSize = Req("intermediate_size"),
            VocabSize = Req("vocab_size"),
            MaxPositions = Req("max_position_embeddings"),
            TypeVocabSize = mpnet ? 0 : Int("type_vocab_size", 0),
            LayerNormEps = Eps(),
            HiddenAct = r.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? "gelu" : "gelu",
            PositionOffset = robertaPositions ? Int("pad_token_id", 1) + 1 : 0,
            RelativeAttentionBuckets = mpnet ? Int("relative_attention_num_buckets", 32) : 0,
        };
    }
}
