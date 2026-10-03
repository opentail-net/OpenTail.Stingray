using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core.Embeddings;
using OpenTail.Stingray.Engine.Encoders;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// GGUF <c>bert</c>-architecture encoders (all-MiniLM-L6-v2 Q8_0) loaded through
/// <see cref="HfEncoderEmbeddingPipeline.LoadGguf"/>, against vectors captured from llama.cpp's
/// <c>llama-embedding</c> (build <c>bed0a8566</c>, CPU, <c>--pooling mean</c>, L2-normalised) on the same file.
/// Both sides read the same quantised weights, so agreement is far tighter than the cross-format
/// (safetensors vs Q8_0) test in <see cref="SentenceEmbeddingLlamaCppTests"/>.
/// The model is looked up in <c>models/_models</c> or at <c>STINGRAY_TEST_MINILM_GGUF</c>; without it the tests skip.
/// </summary>
public sealed class GgufBertEncoderTests
{
    // Exactly the inputs given to llama-embedding when the fixture was captured.
    private static readonly string[] Texts =
    [
        "The quick brown fox jumps over the lazy dog.",
        "Embeddings turn text into vectors.",
        "Ünïcödé text, café — naïve résumé!",
    ];

    private static string? FindGguf()
    {
        string? env = Environment.GetEnvironmentVariable("STINGRAY_TEST_MINILM_GGUF");
        if (env is { Length: > 0 } && File.Exists(env)) return env;
        string? models = BertEncoderOnnxParityTests.FindRepoDir("models/_models");
        string? inRepo = models is null ? null : Path.Combine(models, "all-MiniLM-L6-v2.Q8_0.gguf");
        if (inRepo is not null && File.Exists(inRepo)) return inRepo;
        const string scratch = @"E:\_models\embeddings\minilm-q8.gguf";
        return File.Exists(scratch) ? scratch : null;
    }

    private static float[][] LoadReference(string fixture = "minilm-q8-llama-embedding.json")
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", fixture)));
        return doc.RootElement.GetProperty("data").EnumerateArray()
            .OrderBy(d => d.GetProperty("index").GetInt32())
            .Select(d => d.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray())
            .ToArray();
    }

    [Fact]
    public void MiniLmQ8Gguf_MatchesLlamaEmbedding()
    {
        string? gguf = FindGguf();
        Assert.SkipUnless(gguf != null, "all-MiniLM-L6-v2 Q8_0 GGUF not found (set STINGRAY_TEST_MINILM_GGUF)");

        using var pipeline = HfEncoderEmbeddingPipeline.LoadGguf(gguf!);
        Assert.Equal(384, pipeline.EmbeddingDimensions);
        Assert.Equal(PoolingType.Mean, pipeline.DefaultPooling);
        Assert.True(pipeline.NormalizeByDefault);

        var reference = LoadReference();
        var ours = pipeline.EmbedTexts(Texts);
        Assert.Equal(reference.Length, ours.Length);
        var cosines = new float[ours.Length];
        for (int i = 0; i < ours.Length; i++)
        {
            cosines[i] = TensorPrimitives.CosineSimilarity(ours[i], reference[i]);
            Console.WriteLine($"[GgufBert] text {i}: cos(ours, llama.cpp)={cosines[i]:F7}  |ours|={TensorPrimitives.Norm(ours[i]):F4}  |ref|={TensorPrimitives.Norm(reference[i]):F4}");
        }
        for (int i = 0; i < ours.Length; i++)
        {
            // llama.cpp also quantises activations to Q8_0 inside its matmuls; ours accumulate in F32 (measured 0.9998).
            Assert.True(cosines[i] >= 0.9995f, $"text {i}: cosine {cosines[i]}");
            Assert.InRange(TensorPrimitives.Norm(ours[i]), 0.9999f, 1.0001f);
        }
    }

    [Fact]
    public void MiniLmQ8Gguf_BatchEqualsSingle_AndRoutesThroughTheFactory()
    {
        string? gguf = FindGguf();
        Assert.SkipUnless(gguf != null, "all-MiniLM-L6-v2 Q8_0 GGUF not found (set STINGRAY_TEST_MINILM_GGUF)");

        Assert.True(EncoderPipelineFactory.IsGgufBertEncoder(gguf!));
        using var pipeline = (HfEncoderEmbeddingPipeline)EncoderPipelineFactory.CreateEmbedding(gguf!);

        var batch = pipeline.EmbedTexts(Texts);
        for (int i = 0; i < Texts.Length; i++)
        {
            var single = pipeline.EmbedTexts([Texts[i]])[0];
            float cos = TensorPrimitives.CosineSimilarity(batch[i], single);
            Assert.True(cos >= 0.99999f, $"text {i}: batch vs single cosine {cos}");
        }

        // Repeated input is deterministic.
        var again = pipeline.EmbedTexts([Texts[0]])[0];
        Assert.Equal(pipeline.EmbedTexts([Texts[0]])[0], again);
    }

    // XLM-R family (SentencePiece Unigram + precompiled charsmap, CLS pooling, 250k vocab). Multilingual input
    // exercises the Unigram segmentation and the charsmap normaliser, which the WordPiece MiniLM test cannot.
    private static readonly string[] ArcticTexts =
    [
        "The quick brown fox jumps over the lazy dog.",
        "Embeddings turn text into vectors.",
        "Ünïcödé text, café — naïve résumé!",
        "Bonjour le monde. 你好，世界。",
    ];

    private static string? FindArcticGguf()
    {
        string? env = Environment.GetEnvironmentVariable("STINGRAY_TEST_ARCTIC_GGUF");
        if (env is { Length: > 0 } && File.Exists(env)) return env;
        string? models = BertEncoderOnnxParityTests.FindRepoDir("models/_models");
        string? inRepo = models is null ? null : Path.Combine(models, "snowflake-arctic-embed-l-v2.0-q8_0.gguf");
        if (inRepo is not null && File.Exists(inRepo)) return inRepo;
        const string scratch = @"E:\_models\embeddings\arctic-l-v2-q8.gguf";
        return File.Exists(scratch) ? scratch : null;
    }

    [Fact]
    public void ArcticEmbedLV2Q8Gguf_MatchesLlamaEmbedding()
    {
        string? gguf = FindArcticGguf();
        Assert.SkipUnless(gguf != null, "snowflake-arctic-embed-l-v2.0 Q8_0 GGUF not found (set STINGRAY_TEST_ARCTIC_GGUF)");

        using var pipeline = HfEncoderEmbeddingPipeline.LoadGguf(gguf!);
        Assert.Equal(1024, pipeline.EmbeddingDimensions);
        Assert.Equal(PoolingType.Cls, pipeline.DefaultPooling);

        var reference = LoadReference("arctic-l-v2-q8-llama-embedding.json");
        var ours = pipeline.EmbedTexts(ArcticTexts);
        Assert.Equal(reference.Length, ours.Length);
        var cosines = new float[ours.Length];
        for (int i = 0; i < ours.Length; i++)
        {
            cosines[i] = TensorPrimitives.CosineSimilarity(ours[i], reference[i]);
            Console.WriteLine($"[GgufArctic] text {i}: cos(ours, llama.cpp)={cosines[i]:F7}");
        }
        // 24 layers of Q8_0 against llama.cpp's Q8_0-activation matmuls: measured 0.9991-0.9996.
        for (int i = 0; i < ours.Length; i++)
            Assert.True(cosines[i] >= 0.999f, $"text {i}: cosine {cosines[i]}");

        // The same texts through llama.cpp on the F16 GGUF are the near-exact reference. Ours accumulates in F32 over the
        // same Q8 weights, so it must sit closer to F16 than llama.cpp's own Q8 path does (measured 0.9995-0.9997 vs
        // 0.9987-0.9992), and every pairwise text similarity must agree with F16.
        var f16 = LoadReference("arctic-l-v2-f16-llama-embedding.json");
        for (int i = 0; i < ours.Length; i++)
        {
            float oursToF16 = TensorPrimitives.CosineSimilarity(ours[i], f16[i]);
            float llamaQ8ToF16 = TensorPrimitives.CosineSimilarity(reference[i], f16[i]);
            Console.WriteLine($"[GgufArctic] text {i}: ours~f16={oursToF16:F6} llamaQ8~f16={llamaQ8ToF16:F6}");
            Assert.True(oursToF16 >= 0.9993f, $"text {i}: cosine to F16 {oursToF16}");
            Assert.True(oursToF16 >= llamaQ8ToF16, $"text {i}: ours ({oursToF16}) is farther from F16 than llama.cpp Q8 ({llamaQ8ToF16})");
        }
        for (int i = 0; i < ours.Length; i++)
            for (int j = i + 1; j < ours.Length; j++)
                Assert.InRange(TensorPrimitives.CosineSimilarity(ours[i], ours[j]) - TensorPrimitives.CosineSimilarity(f16[i], f16[j]), -0.012f, 0.012f);
    }
}
