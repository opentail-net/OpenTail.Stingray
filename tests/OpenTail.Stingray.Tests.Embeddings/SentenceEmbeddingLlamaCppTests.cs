using System.Diagnostics;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// End-to-end sentence embeddings (tokenizer → encoder → sentence-transformers pooling → normalize) against
/// llama.cpp's <c>llama-server --embedding</c> on a GGUF of the same checkpoint. The GGUF on disk is Q8_0, so
/// agreement is bounded by quantization (cosine ~0.999, not 1.0); what this independently checks is the
/// pooling/normalize choice read from the checkpoint files, which a wrong mode (e.g. mean instead of CLS)
/// fails by a wide margin (printed for comparison).
/// </summary>
public sealed class SentenceEmbeddingLlamaCppTests
{
    [Fact]
    public void BgeSmall_ClsNormalizedEmbeddings_MatchLlamaServer()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/BAAI__bge-small-en-v1.5");
        string? models = BertEncoderOnnxParityTests.FindRepoDir("models/_models");
        string? gguf = models is null ? null : Path.Combine(models, "bge-small-en-v1.5-q8_0.gguf");
        string? exe = LlamaServerOracle.FindServerExe();
        Assert.SkipUnless(dir != null && gguf != null && File.Exists(gguf) && exe != null, "bge-small checkpoint, its GGUF, or llama-server not found");

        using var pipeline = HfEncoderEmbeddingPipeline.Load(dir!);
        Assert.Equal(PoolingType.Cls, pipeline.DefaultPooling);
        Assert.True(pipeline.NormalizeByDefault);

        var sw = Stopwatch.StartNew();
        using var server = LlamaServerOracle.Start(exe!, gguf!, "--embedding");
        var reference = server.Embeddings(BertEncoderOnnxParityTests.Texts);
        Console.WriteLine($"[BgeSmallLlama] llama-server start+embed {sw.ElapsedMilliseconds} ms");

        var ours = pipeline.EmbedTexts(BertEncoderOnnxParityTests.Texts);
        var wrongPooling = pipeline.EmbedTexts(BertEncoderOnnxParityTests.Texts, PoolingType.Mean, normalize: true);
        for (int i = 0; i < ours.Length; i++)
        {
            float cos = TensorPrimitives.CosineSimilarity(ours[i], reference[i]);
            float cosWrong = TensorPrimitives.CosineSimilarity(wrongPooling[i], reference[i]);
            Console.WriteLine($"[BgeSmallLlama] text {i}: cos(ours, llama.cpp Q8_0)={cos:F5}  (mean pooling would give {cosWrong:F5}); |ours|={TensorPrimitives.Norm(ours[i]):F4}");
            Assert.True(cos >= 0.995f, $"text {i}: cosine {cos}");
            Assert.InRange(TensorPrimitives.Norm(ours[i]), 0.9999f, 1.0001f);
        }
    }
}
