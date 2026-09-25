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

    private static readonly ConcurrentDictionary<string, Lazy<IEmbeddingPipeline>> s_embedding = new(StringComparer.OrdinalIgnoreCase);
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

    /// <summary>New embedding pipeline for <paramref name="model"/> (caller disposes).</summary>
    public static IEmbeddingPipeline CreateEmbedding(string model)
    {
        if (IsHfEncoderDirectory(model)) return HfEncoderEmbeddingPipeline.Load(model);
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

    /// <summary>Process-lifetime shared embedding pipeline per model path (for the server).</summary>
    public static IEmbeddingPipeline GetSharedEmbedding(string model) =>
        s_embedding.GetOrAdd(Path.GetFullPath(model), p => new Lazy<IEmbeddingPipeline>(() => CreateEmbedding(p))).Value;

    /// <summary>Process-lifetime shared reranker per model path (for the server).</summary>
    public static IRerankerPipeline GetSharedReranker(string model) =>
        s_rerank.GetOrAdd(Path.GetFullPath(model), p => new Lazy<IRerankerPipeline>(() => CreateReranker(p))).Value;
}
