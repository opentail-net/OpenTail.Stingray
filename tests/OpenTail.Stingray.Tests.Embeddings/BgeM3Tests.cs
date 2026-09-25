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

    /// <summary>
    /// The model card's <c>compute_score</c> example (pairs q0p0, q0p1, q1p0, q1p1; weights 0.4/0.2/0.4), fp16 on
    /// FlagEmbedding's side. Also the card's lexical weights for "What is BGE M3?" (printed, spot-checked on "GE"/"3").
    /// </summary>
    [Fact]
    public void SparseAndColbert_MatchModelCard()
    {
        string? dir = FindDir();
        Assert.SkipUnless(dir != null && File.Exists(Path.Combine(dir, "model.safetensors")) && BgeM3Pipeline.IsBgeM3Directory(dir),
            "bge-m3 checkpoint (with sparse_linear.pt/colbert_linear.pt) not found");

        using var m3 = BgeM3Pipeline.Load(dir!);
        var q = m3.Encode(Queries);
        var p = m3.Encode(Passages);
        float[] cardColbert = [0.7796499729156494f, 0.4621465802192688f, 0.4523794651031494f, 0.7898575067520142f];
        float[] cardSparse = [0.195556640625f, 0.00879669189453125f, 0.0f, 0.1802978515625f];
        float[] cardDense = [0.6259765625f, 0.347412109375f, 0.349853515625f, 0.67822265625f];
        float[] cardSparseDense = [0.482503205537796f, 0.23454029858112335f, 0.2332356721162796f, 0.5122477412223816f];
        float[] cardAll = [0.6013619303703308f, 0.3255828022956848f, 0.32089319825172424f, 0.6232916116714478f];
        for (int k = 0; k < 4; k++)
        {
            var s = BgeM3Pipeline.Score(q[k / 2], p[k % 2]);
            Console.WriteLine($"[BgeM3] pair {k}: dense {s.Dense:F5} ({cardDense[k]:F5}) sparse {s.Sparse:F5} ({cardSparse[k]:F5}) " +
                $"colbert {s.Colbert:F5} ({cardColbert[k]:F5}) sparse+dense {s.SparseDense:F5} ({cardSparseDense[k]:F5}) all {s.All:F5} ({cardAll[k]:F5})");
            Assert.True(Math.Abs(s.Dense - cardDense[k]) < 3e-3f, $"dense {k}");
            Assert.True(Math.Abs(s.Sparse - cardSparse[k]) < 3e-3f, $"sparse {k}");
            Assert.True(Math.Abs(s.Colbert - cardColbert[k]) < 3e-3f, $"colbert {k}");
            Assert.True(Math.Abs(s.SparseDense - cardSparseDense[k]) < 3e-3f, $"sparse+dense {k}");
            Assert.True(Math.Abs(s.All - cardAll[k]) < 3e-3f, $"colbert+sparse+dense {k}");
        }

        // Card: {'What': 0.08356, 'is': 0.0814, 'B': 0.1296, 'GE': 0.252, 'M': 0.1702, '3': 0.2695, '?': 0.04092}
        var weights = q[0].LexicalWeights.Values.OrderByDescending(w => w).ToArray();
        Console.WriteLine($"[BgeM3] lexical weights q0 (desc): {string.Join(", ", weights.Select(w => w.ToString("F4")))}");
        Assert.Equal(7, weights.Length);
        float[] cardWeights = [0.2695f, 0.252f, 0.1702f, 0.1296f, 0.08356f, 0.0814f, 0.04092f];
        for (int i = 0; i < 7; i++) Assert.True(Math.Abs(weights[i] - cardWeights[i]) < 2e-3f, $"lexical weight {i}: {weights[i]} vs {cardWeights[i]}");
    }

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
