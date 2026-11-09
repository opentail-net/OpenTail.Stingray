using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Qwen 3.8 plan, Phase 7 (knowledge collections): a collection is a hot session whose first turn holds the document. Each
/// later question continues that session, so only the question is prefilled again. Real qwen4exp checkpoint, heavy; skipped
/// when the checkpoint is absent. The cold oracle is a fresh session that gets the document and the question in one turn,
/// and the two answers must match (greedy, same token sequence).
/// </summary>
public sealed class QwenKnowledgeCollectionRealModelTests : HeavyTestBase
{
    private const string Document =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type " +
        "in Mainz around 1440. The first book printed in Europe with movable type was the Gutenberg Bible, completed in 1455.";

    private const string Question = "\nQuestion: Where was movable type developed? Answer:";

    // The model reasons first (its output starts with a thinking trace), so the answer needs room after the trace.
    private const int AnswerTokens = 96;

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string Text(IEnumerable<GenerateChunk> chunks) =>
        string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));

    [Fact]
    public async Task Collection_SecondQuestion_PrefillsOnlyItsOwnTokens_AndMatchesAColdRun()
    {
        string? path = FindTrunk();
        Assert.SkipWhen(path is null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var batched = new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // The split must tokenize the same as the joined text, or the cold oracle is not the same token sequence.
        int documentTokens = tokenizer.Encode(Document).Count;
        int questionTokens = tokenizer.Encode(Question).Count;
        Assert.Equal(documentTokens + questionTokens, tokenizer.Encode(Document + Question).Count);

        var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = AnswerTokens };
        var documentOnly = new SamplingParams { Temperature = 0f, MaxNewTokens = 0 };

        // Hot: the document is the first turn (prefill only), the question is the second turn.
        using var hotEngine = new ContinuousBatchingEngine(batched, tokenizer, "qwen-collection", maxBatchSize: 1);
        var hotRuntime = new HotSessionRuntime(hotEngine, tokenizer, new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        using var hot = hotRuntime.Create();
        await hot.RunTurnAsync(Document, documentOnly, SessionRevision.Initial, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue(Document));
        long promptAfterDocument = hot.Metrics.PromptTokens;
        var hotAnswer = await hot.RunTurnAsync(Question, sampling,
            hotRuntime.GetSessionSnapshot(hot.SessionId).CommittedRevision, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue(Question));
        long promptForQuestion = hot.Metrics.PromptTokens - promptAfterDocument;

        // Cold: a fresh session gets the document and the question in one turn.
        using var coldEngine = new ContinuousBatchingEngine(batched, tokenizer, "qwen-collection-cold", maxBatchSize: 1);
        var coldRuntime = new HotSessionRuntime(coldEngine, tokenizer, new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        using var cold = coldRuntime.Create();
        string joined = Document + Question;
        var coldAnswer = await cold.RunTurnAsync(joined, sampling, SessionRevision.Initial, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue(joined));

        string hotText = Text(hotAnswer.Chunks);
        string coldText = Text(coldAnswer.Chunks);
        Console.WriteLine($"KNOWLEDGE document_tokens={documentTokens} question_tokens={questionTokens} " +
                          $"prefilled_for_question={promptForQuestion} hot=\"{hotText}\" cold=\"{coldText}\"");

        Assert.Equal(questionTokens, promptForQuestion);
        Assert.False(string.IsNullOrWhiteSpace(hotText), "the question must produce an answer");
        Assert.Equal(coldText, hotText);
    }
}
