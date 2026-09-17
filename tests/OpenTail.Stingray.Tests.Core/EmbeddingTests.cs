using System.Diagnostics;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core.Embeddings;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class EmbeddingTests(ITestOutputHelper? output = null)
{
    [Fact]
    public void EmbeddingNormalizer_NormalizeL2_ProducesUnitVector()
    {
        float[] v = [3f, 4f, 0f, 0f]; // Norm = 5.0
        EmbeddingNormalizer.NormalizeL2(v);

        Assert.InRange(v[0], 0.599f, 0.601f); // 3 / 5 = 0.6
        Assert.InRange(v[1], 0.799f, 0.801f); // 4 / 5 = 0.8

        float normSq = TensorPrimitives.Dot(v, v);
        Assert.InRange(MathF.Sqrt(normSq), 0.999f, 1.001f);
    }

    [Fact]
    public void EmbeddingNormalizer_TruncateAndNormalize_AppliesMatryoshkaReduction()
    {
        float[] fullVector = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] truncated = EmbeddingNormalizer.TruncateAndNormalize(fullVector, targetDim: 4);

        Assert.Equal(4, truncated.Length);

        float normSq = TensorPrimitives.Dot(truncated, truncated);
        Assert.InRange(MathF.Sqrt(normSq), 0.999f, 1.001f);
    }

    [Fact]
    public void EmbeddingNormalizer_CosineSimilarity_CalculatesCorrectAngles()
    {
        float[] v1 = [1f, 0f, 0f];
        float[] v2 = [1f, 0f, 0f];
        float[] v3 = [0f, 1f, 0f];
        float[] v4 = [-1f, 0f, 0f];

        Assert.InRange(EmbeddingNormalizer.CosineSimilarity(v1, v2), 0.999f, 1.0f);   // Identical -> 1.0
        Assert.InRange(EmbeddingNormalizer.CosineSimilarity(v1, v3), -0.001f, 0.001f); // Orthogonal -> 0.0
        Assert.InRange(EmbeddingNormalizer.CosineSimilarity(v1, v4), -1.0f, -0.999f); // Opposite -> -1.0
    }

    [Fact]
    public void EmbeddingNormalizer_ApplyPooling_CalculatesExpectedReductions()
    {
        // 3 tokens, 2 hidden dims: [[1, 2], [3, 4], [5, 6]]
        float[] hiddenStates = [1f, 2f, 3f, 4f, 5f, 6f];
        int seqLen = 3;
        int dModel = 2;

        // 1. Mean: [(1+3+5)/3, (2+4+6)/3] = [3, 4]
        float[] meanPool = EmbeddingNormalizer.ApplyPooling(hiddenStates, seqLen, dModel, PoolingType.Mean);
        Assert.Equal(2, meanPool.Length);
        Assert.Equal(3f, meanPool[0]);
        Assert.Equal(4f, meanPool[1]);

        // 2. CLS: [1, 2]
        float[] clsPool = EmbeddingNormalizer.ApplyPooling(hiddenStates, seqLen, dModel, PoolingType.Cls);
        Assert.Equal(2, clsPool.Length);
        Assert.Equal(1f, clsPool[0]);
        Assert.Equal(2f, clsPool[1]);

        // 3. LastToken: [5, 6]
        float[] lastPool = EmbeddingNormalizer.ApplyPooling(hiddenStates, seqLen, dModel, PoolingType.LastToken);
        Assert.Equal(2, lastPool.Length);
        Assert.Equal(5f, lastPool[0]);
        Assert.Equal(6f, lastPool[1]);
    }

    [Fact]
    public void EmbeddingEngine_Embed_ProducesNormalizedEmbeddings()
    {
        using var engine = new EmbeddingEngine(
            modelName: "bge-large-en-v1.5",
            embeddingDimensions: 1024,
            defaultPooling: PoolingType.Mean);

        var req = new EmbeddingRequest
        {
            Inputs = ["What is retrieval-augmented generation?", "OpenTail Stingray embedding engine"],
            Normalize = true
        };

        var result = engine.Embed(req);

        Assert.NotNull(result);
        Assert.Equal("bge-large-en-v1.5", result.Model);
        Assert.Equal(2, result.Data.Count);

        for (int i = 0; i < result.Data.Count; i++)
        {
            var item = result.Data[i];
            Assert.Equal(i, item.Index);
            Assert.Equal(1024, item.Vector.Length);

            float normSq = TensorPrimitives.Dot(item.Vector, item.Vector);
            Assert.InRange(MathF.Sqrt(normSq), 0.999f, 1.001f);
        }
    }

    [Fact]
    public void EmbeddingEngine_Rerank_RanksDocumentsByScoreDescending()
    {
        using var engine = new EmbeddingEngine(
            modelName: "bge-reranker-large",
            embeddingDimensions: 768);

        var req = new RerankRequest
        {
            Query = "How to write high-performance C# code",
            Documents =
            [
                "The recipe for chocolate chip cookies includes flour and sugar.",
                "High-performance C# relies on Span, Memory, SIMD, and zero-allocation techniques.",
                "Weather forecast for tomorrow is sunny with scattered clouds."
            ],
            TopN = 2,
            ReturnDocuments = true
        };

        var result = engine.Rerank(req);

        Assert.NotNull(result);
        Assert.Equal(2, result.Results.Count);

        // Verify descending sort by relevance score
        Assert.True(result.Results[0].RelevanceScore >= result.Results[1].RelevanceScore);

        // Top document should have valid score in [0.0, 1.0]
        Assert.InRange(result.Results[0].RelevanceScore, 0.0f, 1.0f);
        Assert.NotNull(result.Results[0].Document);
    }

    [Fact]
    public void EmbeddingEngine_RealGgufModel_GeneratesSemanticEmbeddings()
    {
        var ggufPath = FindModelPath("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        if (ggufPath is null)
        {
            // Skip when fixture is absent
            return;
        }

        using var engine = new EmbeddingEngine(modelName: ggufPath);
        Assert.Equal(1024, engine.EmbeddingDimensions);

        var req = new EmbeddingRequest
        {
            Inputs =
            [
                "What is retrieval-augmented generation?",
                "Vector database indexing and semantic search",
                "How to bake chocolate chip cookies"
            ],
            Normalize = true
        };

        var result = engine.Embed(req);
        Assert.Equal(3, result.Data.Count);

        var v0 = result.Data[0].Vector;
        var v1 = result.Data[1].Vector;
        var v2 = result.Data[2].Vector;

        // Semantic check: RAG query should be much closer to Vector DB query than to Cookie recipe
        float simTech = EmbeddingNormalizer.CosineSimilarity(v0, v1);
        float simCooking = EmbeddingNormalizer.CosineSimilarity(v0, v2);

        Assert.True(simTech > simCooking, $"Expected simTech ({simTech:F4}) > simCooking ({simCooking:F4})");
    }

    [Fact]
    public void EmbeddingEngine_RealGgufModel_SupportsMatryoshkaTruncation()
    {
        var ggufPath = FindModelPath("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        if (ggufPath is null)
        {
            return;
        }

        using var engine = new EmbeddingEngine(modelName: ggufPath);

        var req = new EmbeddingRequest
        {
            Inputs = ["High performance SIMD computing in .NET"],
            Dimensions = 256,
            Normalize = true
        };

        var result = engine.Embed(req);
        Assert.Single(result.Data);
        Assert.Equal(256, result.Data[0].Vector.Length);

        float normSq = TensorPrimitives.Dot(result.Data[0].Vector, result.Data[0].Vector);
        Assert.InRange(MathF.Sqrt(normSq), 0.999f, 1.001f);
    }

    [Fact]
    public void EmbeddingEngine_RealGgufModel_BatchParityWithSingle_ProducesIdenticalEmbeddings()
    {
        var ggufPath = FindModelPath("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        if (ggufPath is null)
        {
            return;
        }

        using var engine = new EmbeddingEngine(modelName: ggufPath);

        string doc0 = "What is retrieval-augmented generation?";
        string doc1 = "Vector database indexing and semantic search";
        string doc2 = "How to bake chocolate chip cookies";

        var res0 = engine.Embed(new EmbeddingRequest { Inputs = [doc0], Normalize = true });
        var res1 = engine.Embed(new EmbeddingRequest { Inputs = [doc1], Normalize = true });
        var res2 = engine.Embed(new EmbeddingRequest { Inputs = [doc2], Normalize = true });

        var resBatch = engine.Embed(new EmbeddingRequest { Inputs = [doc0, doc1, doc2], Normalize = true });

        Assert.Equal(3, resBatch.Data.Count);

        float sim0 = EmbeddingNormalizer.CosineSimilarity(res0.Data[0].Vector, resBatch.Data[0].Vector);
        float sim1 = EmbeddingNormalizer.CosineSimilarity(res1.Data[0].Vector, resBatch.Data[1].Vector);
        float sim2 = EmbeddingNormalizer.CosineSimilarity(res2.Data[0].Vector, resBatch.Data[2].Vector);

        Assert.True(sim0 > 0.999f, $"Doc 0 cosine similarity to single was {sim0:F6}");
        Assert.True(sim1 > 0.999f, $"Doc 1 cosine similarity to single was {sim1:F6}");
        Assert.True(sim2 > 0.999f, $"Doc 2 cosine similarity to single was {sim2:F6}");
    }

    [Fact]
    public void EmbeddingEngine_RealGgufModel_BatchThroughput_ExceedsSequential()
    {
        var ggufPath = FindModelPath("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        if (ggufPath is null)
        {
            return;
        }

        using var engine = new EmbeddingEngine(modelName: ggufPath);

        string[] docs =
        [
            "What is retrieval-augmented generation in modern natural language processing systems?",
            "Vector database indexing and semantic search algorithms for large document corpora.",
            "How to bake chocolate chip cookies with crisp edges and a soft chewy center.",
            "Deep neural network architectures for computer vision and multimodal feature representations.",
            "Quantum computing principles and qubit superposition states in superconducting circuits."
        ];

        // Warm-up
        engine.Embed(new EmbeddingRequest { Inputs = [docs[0]], Normalize = true });
        engine.Embed(new EmbeddingRequest { Inputs = [docs[0], docs[1]], Normalize = true });

        // Measure sequential
        var swSeq = Stopwatch.StartNew();
        for (int i = 0; i < docs.Length; i++)
        {
            engine.Embed(new EmbeddingRequest { Inputs = [docs[i]], Normalize = true });
        }
        swSeq.Stop();
        long seqMs = swSeq.ElapsedMilliseconds;

        // Measure batched
        var swBatch = Stopwatch.StartNew();
        var resBatch = engine.Embed(new EmbeddingRequest { Inputs = docs, Normalize = true });
        swBatch.Stop();
        long batchMs = swBatch.ElapsedMilliseconds;

        output?.WriteLine($"[Embed Benchmark] Sequential ({docs.Length} docs): {seqMs}ms | Batched ({docs.Length} docs): {batchMs}ms | Speedup: {(float)seqMs / Math.Max(1, batchMs):F2}x");
        Console.WriteLine($"[Embed Benchmark] Sequential ({docs.Length} docs): {seqMs}ms | Batched ({docs.Length} docs): {batchMs}ms | Speedup: {(float)seqMs / Math.Max(1, batchMs):F2}x");

        Assert.Equal(docs.Length, resBatch.Data.Count);
        Assert.True(batchMs < seqMs, $"Expected batched ({batchMs}ms) to be faster than sequential ({seqMs}ms)");
    }

    private static string? FindModelPath(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, relativePath);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}


