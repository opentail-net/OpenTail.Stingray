using System.Collections.Concurrent;
using System.Text.Json;
using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Resolves a model argument to a real embedding or rerank pipeline:
/// <list type="bullet">
/// <item>an HF encoder checkpoint directory (<c>config.json</c> with an encoder <c>model_type</c> + safetensors +
/// <c>tokenizer.json</c>) → <see cref="HfEncoderEmbeddingPipeline"/>, or for rerank
/// <see cref="HfCrossEncoderPipeline"/> (the checkpoint must have a sequence-classification head);</item>
/// <item>a <c>.gguf</c> file → <see cref="EmbeddingEngine"/> (decoder embedder forward pass; its rerank is a
/// bi-encoder cosine, not a cross-encoder).</item>
/// </list>
/// Anything else throws: there is no synthetic fallback.
/// </summary>
public static class EncoderPipelineFactory
{
    private static readonly HashSet<string> s_encoderModelTypes = new(StringComparer.Ordinal)
    {
        "bert", "electra", "roberta", "xlm-roberta", "camembert", "mpnet", "nomic_bert",
    };

    private static readonly ConcurrentDictionary<string, Lazy<ResidentEmbedder>> s_embedding = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, Lazy<IRerankerPipeline>> s_rerank = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>True when <paramref name="path"/> is a directory holding an HF encoder checkpoint.</summary>
    public static bool IsHfEncoderDirectory(string path)
    {
        string config = Path.Combine(path, "config.json");
        if (!Directory.Exists(path) || !File.Exists(config) || !File.Exists(Path.Combine(path, "tokenizer.json"))) return false;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(config));
            return doc.RootElement.TryGetProperty("model_type", out var mt) && mt.GetString() is { } t && s_encoderModelTypes.Contains(t);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>True when <paramref name="path"/> is a GGUF whose architecture is the <c>bert</c> encoder (not a decoder embedder).</summary>
    public static bool IsGgufBertEncoder(string path)
    {
        if (!path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || !File.Exists(path)) return false;
        using var m = OpenTail.Stingray.Core.GgufModel.Open(path);
        return m.Metadata.TryGetValue("general.architecture", out var a) && a as string == "bert";
    }

    /// <summary>New embedding pipeline for <paramref name="model"/> (caller disposes).</summary>
    public static IEmbeddingPipeline CreateEmbedding(string model)
    {
        if (IsHfEncoderDirectory(model)) return HfEncoderEmbeddingPipeline.Load(model);
        if (IsGgufBertEncoder(model)) return HfEncoderEmbeddingPipeline.LoadGguf(model);
        if (model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && File.Exists(model)) return new EmbeddingEngine(model);
        throw new FileNotFoundException(
            $"Embedding model '{model}' not found. Pass an HF encoder checkpoint directory (config.json, model.safetensors, " +
            "tokenizer.json) or a GGUF file.", model);
    }

    /// <summary>New reranker for <paramref name="model"/> (caller disposes).</summary>
    public static IRerankerPipeline CreateReranker(string model)
    {
        if (IsHfEncoderDirectory(model)) return HfCrossEncoderPipeline.Load(model);
        if (model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && File.Exists(model)) return new EmbeddingEngine(model);
        throw new FileNotFoundException(
            $"Reranker model '{model}' not found. Pass an HF cross-encoder checkpoint directory " +
            "(*ForSequenceClassification) or a GGUF embedding model (bi-encoder cosine).", model);
    }

    /// <summary>Resident embedder per model path (for the server): loaded on first use, kept until unloaded.</summary>
    public static ResidentEmbedder GetResidentEmbedding(string model) =>
        s_embedding.GetOrAdd(Path.GetFullPath(model), p => new Lazy<ResidentEmbedder>(() => new ResidentEmbedder(CreateEmbedding(p)))).Value;

    /// <summary>Shared embedding pipeline per model path. Prefer <see cref="GetResidentEmbedding"/>, which serialises callers.</summary>
    public static IEmbeddingPipeline GetSharedEmbedding(string model) => GetResidentEmbedding(model).Pipeline;

    /// <summary>
    /// Unloads the resident embedder for <paramref name="model"/>, waiting for a call in progress. A later request
    /// loads it again. Returns false when it was not loaded.
    /// </summary>
    public static bool UnloadEmbedding(string model)
    {
        if (!s_embedding.TryRemove(Path.GetFullPath(model), out var lazy)) return false;
        if (lazy.IsValueCreated) lazy.Value.Dispose();
        return true;
    }

    /// <summary>Process-lifetime shared reranker per model path (for the server).</summary>
    public static IRerankerPipeline GetSharedReranker(string model) =>
        s_rerank.GetOrAdd(Path.GetFullPath(model), p => new Lazy<IRerankerPipeline>(() => CreateReranker(p))).Value;
}
