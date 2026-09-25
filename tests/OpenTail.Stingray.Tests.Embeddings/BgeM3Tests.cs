using System.Numerics.Tensors;
using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// BAAI/bge-m3 (XLM-R large, 8194 positions). Oracles: the model card's documented example scores (computed by
/// FlagEmbedding with <c>use_fp16=True</c>, so agreement is bounded by fp16 rounding, ~1e-3), and llama.cpp's
/// <c>llama-server --embedding</c> on the gpustack Q8_0 GGUF for the dense vectors themselves.
/// </summary>
public sealed class BgeM3Tests
{
    internal static readonly string[] Queries = ["What is BGE M3?", "Defination of BM25"];

    internal static readonly string[] Passages =
    [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ];

    internal static string? FindDir() => BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/BAAI__bge-m3");

    [Fact]
    public void Dense_MatchesModelCardAndLlamaServer()
    {
        string? dir = FindDir();
        Assert.SkipUnless(dir != null && File.Exists(Path.Combine(dir, "model.safetensors")), "bge-m3 checkpoint not found");

        using var pipeline = HfEncoderEmbeddingPipeline.Load(dir!);
        Assert.Equal(PoolingType.Cls, pipeline.DefaultPooling);
        Assert.True(pipeline.NormalizeByDefault);

        var q = pipeline.EmbedTexts(Queries);
        var p = pipeline.EmbedTexts(Passages);
        // Model card: similarity = embeddings_1 @ embeddings_2.T -> [[0.6265, 0.3477], [0.3499, 0.678]]
        // (compute_score's 'dense' list gives the same four at more digits).
        float[,] card = { { 0.6259765625f, 0.347412109375f }, { 0.349853515625f, 0.67822265625f } };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                float s = TensorPrimitives.Dot(q[i], p[j]);
                Console.WriteLine($"[BgeM3] dense q{i}·p{j} = {s:F5} (card {card[i, j]:F5})");
                Assert.True(Math.Abs(s - card[i, j]) < 3e-3f, $"dense q{i}·p{j}: {s} vs card {card[i, j]}");
            }

        string? models = BertEncoderOnnxParityTests.FindRepoDir("models/_models");
        string? gguf = models is null ? null : Path.Combine(models, "bge-m3-Q8_0.gguf");
        string? exe = LlamaServerOracle.FindServerExe();
        Assert.SkipUnless(gguf != null && File.Exists(gguf) && exe != null, "bge-m3-Q8_0.gguf or llama-server not found (card check passed)");
        using var server = LlamaServerOracle.Start(exe!, gguf!, "--embedding");
        var texts = Queries.Concat(Passages).Concat(BertEncoderOnnxParityTests.Texts).ToArray();
        var reference = server.Embeddings(texts);
        var ours = pipeline.EmbedTexts(texts);
        for (int i = 0; i < texts.Length; i++)
        {
            float cos = TensorPrimitives.CosineSimilarity(ours[i], reference[i]);
            Console.WriteLine($"[BgeM3] text {i}: cos(ours, llama.cpp Q8_0) = {cos:F5}");
            Assert.True(cos >= 0.995f, $"text {i}: cosine {cos}");
        }
    }
}
