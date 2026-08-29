using System.Numerics.Tensors;
using OpenTail.Stingray.Core.Embeddings;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

public sealed class EmbeddingTests
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
}
