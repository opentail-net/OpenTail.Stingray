using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Sentence-embedding model from a HF / sentence-transformers checkpoint directory: <c>tokenizer.json</c>
/// → <see cref="TransformerEncoder"/> → pooling → optional L2 normalize. Pooling, normalization and the
/// max sequence length come from the checkpoint's own sentence-transformers files
/// (<c>modules.json</c>, <c>1_Pooling/config.json</c>, <c>sentence_bert_config.json</c>); a plain
/// <c>BertModel</c> directory without them defaults to CLS pooling and no normalization.
/// </summary>
public sealed class HfEncoderEmbeddingPipeline : IEmbeddingPipeline
{
    private readonly TransformerEncoder _encoder;

    public EncoderTokenizer Tokenizer { get; }
    public string ModelName { get; }
    public int EmbeddingDimensions => _encoder.Config.HiddenSize;
    public PoolingType DefaultPooling { get; }

    /// <summary>Whether the checkpoint's sentence-transformers pipeline ends with a Normalize module.</summary>
    public bool NormalizeByDefault { get; }

    /// <summary>Inputs are truncated to this many tokens, special tokens included.</summary>
    public int MaxSequenceLength { get; }

    private HfEncoderEmbeddingPipeline(string name, EncoderTokenizer tokenizer, TransformerEncoder encoder,
        PoolingType pooling, bool normalize, int maxLen)
    {
        ModelName = name;
        Tokenizer = tokenizer;
        _encoder = encoder;
        DefaultPooling = pooling;
        NormalizeByDefault = normalize;
        MaxSequenceLength = maxLen;
    }

    public static HfEncoderEmbeddingPipeline Load(string modelDir)
    {
        var tokenizer = EncoderTokenizer.FromTokenizerJson(Path.Combine(modelDir, "tokenizer.json"));
        var encoder = TransformerEncoder.Load(modelDir);
        var (pooling, normalize) = ReadSentenceTransformersModules(modelDir);
        int maxLen = encoder.Config.MaxSequenceLength;
        string stConfig = Path.Combine(modelDir, "sentence_bert_config.json");
        if (File.Exists(stConfig))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(stConfig));
            if (doc.RootElement.TryGetProperty("max_seq_length", out var m) && m.ValueKind == JsonValueKind.Number)
                maxLen = Math.Min(maxLen, m.GetInt32());
        }
        string name = Path.GetFileName(Path.TrimEndingDirectorySeparator(modelDir));
        return new HfEncoderEmbeddingPipeline(name, tokenizer, encoder, pooling, normalize, maxLen);
    }

    /// <summary>
    /// Loads a GGUF <c>bert</c>-architecture embedding model (e.g. all-MiniLM-L6-v2 Q8_0). Pooling comes from
    /// <c>bert.pooling_type</c> (llama.cpp numbering, which <see cref="PoolingType"/> shares); output is L2-normalised
    /// by default like llama.cpp's <c>llama-embedding</c>. The WordPiece vocab is read from the GGUF, with BERT-uncased
    /// normalisation assumed (GGUF stores no normalizer flags).
    /// </summary>
    public static HfEncoderEmbeddingPipeline LoadGguf(string ggufPath)
    {
        using var model = OpenTail.Stingray.Core.GgufModel.Open(ggufPath);
        var md = model.Metadata;

        string tokModel = md.TryGetValue("tokenizer.ggml.model", out var tm) ? tm as string ?? "" : "";
        var tokens = ((object[])md["tokenizer.ggml.tokens"]).Select(o => (string)o).ToList();
        int MdInt(string key, int fallback) => md.TryGetValue(key, out var v) ? Convert.ToInt32(v, System.Globalization.CultureInfo.InvariantCulture) : fallback;
        EncoderTokenizer tokenizer;
        switch (tokModel)
        {
            case "bert": // WordPiece (MiniLM, BGE, ...)
                tokenizer = EncoderTokenizer.FromWordPiece(OpenTail.Stingray.Core.BertWordPieceTokenizer.FromGgufVocab(tokens));
                break;
            case "t5": // SentencePiece Unigram (XLM-R family, e.g. Arctic Embed L v2)
            {
                var scores = ((object[])md["tokenizer.ggml.scores"]).Select(o => Convert.ToSingle(o, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
                int[]? types = md.TryGetValue("tokenizer.ggml.token_type", out var tt2)
                    ? ((object[])tt2).Select(o => Convert.ToInt32(o, System.Globalization.CultureInfo.InvariantCulture)).ToArray() : null;
                byte[]? charsmap = md.TryGetValue("tokenizer.ggml.precompiled_charsmap", out var cm) && cm is object[] raw
                    ? raw.Select(o => unchecked((byte)Convert.ToInt32(o, System.Globalization.CultureInfo.InvariantCulture))).ToArray() : null;
                var uni = OpenTail.Stingray.Core.UnigramTokenizer.FromGgufVocab(tokens.ToArray(), scores, MdInt("tokenizer.ggml.unknown_token_id", 3), types, charsmap);
                tokenizer = EncoderTokenizer.FromUnigram(uni, MdInt("tokenizer.ggml.bos_token_id", 0), MdInt("tokenizer.ggml.eos_token_id", 2), MdInt("tokenizer.ggml.padding_token_id", 1));
                break;
            }
            default:
                throw new NotSupportedException($"GGUF encoder tokenizer '{tokModel}' is not supported (only 'bert' WordPiece and 't5' Unigram).");
        }

        // A single-row type table (XLM-R) is stored 1-D, so there is no second dimension to read.
        int typeVocab = model.FindTensor("token_types.weight") is { } tt ? (tt.Dimensions.Length > 1 ? (int)tt.Dimensions[1] : 1) : 0;
        var config = EncoderConfig.FromGguf(md, typeVocab);
        var encoder = TransformerEncoder.LoadGguf(config, model);

        var pooling = md.TryGetValue("bert.pooling_type", out var pt)
            ? (PoolingType)Convert.ToInt32(pt, System.Globalization.CultureInfo.InvariantCulture)
            : PoolingType.Mean;
        if (pooling == PoolingType.None) pooling = PoolingType.Mean;
        string name = md.TryGetValue("general.name", out var gn) && gn is string s && s.Length > 0 ? s : Path.GetFileNameWithoutExtension(ggufPath);
        return new HfEncoderEmbeddingPipeline(name, tokenizer, encoder, pooling, normalize: true, config.MaxSequenceLength);
    }

    private static (PoolingType Pooling, bool Normalize) ReadSentenceTransformersModules(string modelDir)
    {
        string modules = Path.Combine(modelDir, "modules.json");
        if (!File.Exists(modules)) return (PoolingType.Cls, false);

        var pooling = PoolingType.Cls;
        bool normalize = false;
        using var doc = JsonDocument.Parse(File.ReadAllBytes(modules));
        foreach (var module in doc.RootElement.EnumerateArray())
        {
            string type = module.GetProperty("type").GetString() ?? "";
            if (type.EndsWith(".Normalize", StringComparison.Ordinal)) normalize = true;
            if (!type.EndsWith(".Pooling", StringComparison.Ordinal)) continue;
            string cfgPath = Path.Combine(modelDir, module.GetProperty("path").GetString() ?? "", "config.json");
            using var cfg = JsonDocument.Parse(File.ReadAllBytes(cfgPath));
            var c = cfg.RootElement;
            bool Flag(string n) => c.TryGetProperty(n, out var v) && v.ValueKind == JsonValueKind.True;
            int modes = (Flag("pooling_mode_cls_token") ? 1 : 0) + (Flag("pooling_mode_mean_tokens") ? 1 : 0)
                + (Flag("pooling_mode_max_tokens") ? 1 : 0) + (Flag("pooling_mode_mean_sqrt_len_tokens") ? 1 : 0)
                + (Flag("pooling_mode_lasttoken") ? 1 : 0) + (Flag("pooling_mode_weightedmean_tokens") ? 1 : 0);
            if (modes != 1)
                throw new NotSupportedException($"'{cfgPath}': only a single CLS, mean or last-token pooling mode is supported.");
            pooling = Flag("pooling_mode_cls_token") ? PoolingType.Cls
                : Flag("pooling_mode_mean_tokens") ? PoolingType.Mean
                : Flag("pooling_mode_lasttoken") ? PoolingType.LastToken
                : throw new NotSupportedException($"'{cfgPath}': pooling mode not supported.");
        }
        return (pooling, normalize);
    }

    /// <summary>Last hidden state for each text (tokenized with this model's truncation).</summary>
    public float[][] EncodeHidden(IReadOnlyList<string> texts, out EncodedInput[] inputs)
    {
        inputs = texts.Select(t => Tokenizer.Encode(t, MaxSequenceLength)).ToArray();
        return _encoder.EncodeBatch(inputs);
    }

    /// <summary>Pooled (and, by default per the checkpoint, L2-normalized) embedding per text.</summary>
    public float[][] EmbedTexts(IReadOnlyList<string> texts, PoolingType? pooling = null, bool? normalize = null)
    {
        var hidden = EncodeHidden(texts, out _);
        int h = EmbeddingDimensions;
        var p = pooling ?? DefaultPooling;
        bool norm = normalize ?? NormalizeByDefault;
        var result = new float[hidden.Length][];
        for (int i = 0; i < hidden.Length; i++)
        {
            result[i] = Pool(hidden[i], hidden[i].Length / h, h, p);
            if (norm) L2NormalizeInPlace(result[i]);
        }
        return result;
    }

    public static float[] Pool(float[] hidden, int tokens, int h, PoolingType pooling)
    {
        switch (pooling)
        {
            case PoolingType.Cls:
                return hidden.AsSpan(0, h).ToArray();
            case PoolingType.LastToken:
                return hidden.AsSpan((tokens - 1) * h, h).ToArray();
            case PoolingType.Mean:
                var mean = new float[h];
                for (int t = 0; t < tokens; t++) TensorPrimitives.Add(mean, hidden.AsSpan(t * h, h), mean);
                TensorPrimitives.Divide(mean, tokens, mean);
                return mean;
            default:
                throw new NotSupportedException($"Pooling {pooling} does not produce one vector per input.");
        }
    }

    public static void L2NormalizeInPlace(Span<float> v)
    {
        // sentence-transformers Normalize = F.normalize(p=2, eps=1e-12).
        float norm = MathF.Max(TensorPrimitives.Norm(v), 1e-12f);
        TensorPrimitives.Divide(v, norm, v);
    }

    public EmbeddingResult Embed(EmbeddingRequest request)
    {
        var vectors = EmbedTexts(request.Inputs, request.Pooling, request.Normalize);
        int tokens = request.Inputs.Sum(t => Tokenizer.Encode(t, MaxSequenceLength).Ids.Length);
        var data = new List<EmbeddingData>(vectors.Length);
        for (int i = 0; i < vectors.Length; i++)
        {
            var v = vectors[i];
            if (request.Dimensions is int d && d > 0 && d < v.Length)
            {
                v = v[..d];
                if (request.Normalize) L2NormalizeInPlace(v);
            }
            data.Add(new EmbeddingData { Index = i, Vector = v });
        }
        return new EmbeddingResult(ModelName, data, tokens, tokens);
    }

    public void Dispose() => _encoder.Dispose();
}
