using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Real-weight cross-encoder rerankers:
/// <list type="bullet">
/// <item>cross-encoder/ms-marco-MiniLM-L6-v2 (BERT + pooler + classifier): logits vs its own
/// <c>onnx/model.onnx</c> on the same pair ids (token types 0/1), plus relevance ordering.</item>
/// <item>BAAI/bge-reranker-v2-m3 (XLM-R large + RobertaClassificationHead, no ONNX export): the tight oracle is
/// the model card (below). <c>llama-server --rerank</c> is only checked for ranking order: on 2026-09-25 it scored
/// the card's own ["query", "passage"] example -6.37 (FP16) / -6.33 (Q8_0) against the card's -5.652, a systematic
/// llama.cpp-side offset (both quantizations agree), while this port gives -5.650. Ranking vs
/// <c>llama-server --rerank</c> on the gpustack GGUF (FP16 when present, else Q8_0), plus the model
/// card's documented <c>compute_score</c> values (fp16): <c>[query, passage]</c> -5.652, and the panda batch
/// -8.1875 / 5.2617).</item>
/// </list>
/// </summary>
public sealed class CrossEncoderParityTests
{
    private const string Query = "How many people live in Berlin?";

    private static readonly string[] Docs =
    [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "New York City is famous for the Metropolitan Museum of Art.",
        "Berlin is the capital and largest city of Germany, both by area and by population.",
        "The giant panda is a bear species endemic to China.",
    ];

    [Fact]
    public void MsMarcoMiniLm_Logits_MatchOnnx()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/cross-encoder__ms-marco-MiniLM-L6-v2");
        string? onnxPath = dir is null ? null : BertEncoderOnnxParityTests.FindOnnx(dir);
        Assert.SkipUnless(onnxPath != null, "ms-marco-MiniLM-L6-v2 checkpoint/onnx not found");

        var sw = Stopwatch.StartNew();
        using var reranker = HfCrossEncoderPipeline.Load(dir!);
        using var onnx = new OnnxModelSession(onnxPath!);
        var ours = reranker.Score(Query, Docs);
        Console.WriteLine($"[MsMarco] load+score {sw.ElapsedMilliseconds} ms; onnx outputs {string.Join(",", onnx.OutputNames)}");

        float worst = 0f;
        for (int i = 0; i < Docs.Length; i++)
        {
            var input = reranker.Tokenizer.EncodePair(Query, Docs[i], reranker.MaxSequenceLength);
            int n = input.Ids.Length;
            var reference = onnx.Run(
                ("input_ids", input.Ids.Select(x => (long)x).ToArray(), [1, n]),
                ("attention_mask", Enumerable.Repeat(1L, n).ToArray(), [1, n]),
                ("token_type_ids", input.TypeIds.Select(x => (long)x).ToArray(), [1, n]))["logits"][0];
            Console.WriteLine($"[MsMarco] doc {i}: ours {ours[i]:F5}  onnx {reference:F5}");
            worst = MathF.Max(worst, MathF.Abs(ours[i] - reference));
        }
        Assert.True(worst <= 1e-3f, $"max |logit diff| {worst}");
        Assert.True(ours[0] > 5f && ours[1] < -2f, "relevant Berlin passage should score high, the NYC one low");
    }

    [Fact]
    public void BgeRerankerV2M3_Scores_MatchLlamaServerAndModelCard()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/BAAI__bge-reranker-v2-m3");
        string? models = BertEncoderOnnxParityTests.FindRepoDir("models/_models");
        string? gguf = models is null ? null
            : new[] { "bge-reranker-v2-m3-FP16.gguf", "bge-reranker-v2-m3-Q8_0.gguf" }.Select(f => Path.Combine(models, f)).FirstOrDefault(File.Exists);
        string? exe = LlamaServerOracle.FindServerExe();
        Assert.SkipUnless(dir != null && gguf != null && exe != null, "bge-reranker-v2-m3 checkpoint, its GGUF, or llama-server not found");

        var sw = Stopwatch.StartNew();
        using var reranker = HfCrossEncoderPipeline.Load(dir!);
        Console.WriteLine($"[BgeM3Rerank] load {sw.ElapsedMilliseconds} ms");

        // Model card (FlagEmbedding compute_score, fp16): ["query", "passage"] -> -5.65234375; the batch
        // [["what is panda?", "hi"], ["what is panda?", "The giant panda (...) endemic to China."]] -> [-8.1875, 5.26171875].
        sw.Restart();
        var card = reranker.Score("what is panda?",
        [
            "hi",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ]);
        float queryPassage = reranker.Score("query", ["passage"])[0];
        Console.WriteLine($"[BgeM3Rerank] model-card pairs: {queryPassage:F4} (card -5.6523), {card[0]:F4} (card -8.1875), {card[1]:F4} (card 5.2617) in {sw.ElapsedMilliseconds} ms");
        Assert.InRange(queryPassage, -5.70f, -5.60f);
        Assert.InRange(card[0], -8.24f, -8.14f);
        Assert.InRange(card[1], 5.21f, 5.31f);

        sw.Restart();
        var ours = reranker.Score(Query, Docs);
        long oursMs = sw.ElapsedMilliseconds;
        using var server = LlamaServerOracle.Start(exe!, gguf!, "--rerank");
        var reference = server.Rerank(Query, Docs);
        float worst = 0f;
        for (int i = 0; i < Docs.Length; i++)
        {
            Console.WriteLine($"[BgeM3Rerank] doc {i}: ours {ours[i]:F4}  llama.cpp({Path.GetFileName(gguf)}) {reference[i]:F4}");
            worst = MathF.Max(worst, MathF.Abs(ours[i] - reference[i]));
        }
        Console.WriteLine($"[BgeM3Rerank] ours {oursMs} ms for {Docs.Length} pairs; max |diff| {worst:F4}");
        Assert.Equal(Enumerable.Range(0, Docs.Length).OrderByDescending(i => reference[i]), Enumerable.Range(0, Docs.Length).OrderByDescending(i => ours[i]));
    }
}
