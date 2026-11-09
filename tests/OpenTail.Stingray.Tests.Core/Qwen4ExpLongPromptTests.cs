using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 1 and Phase 6.3, real checkpoint: a prompt longer than the indexer's top-k (2048 tokens) exercises
/// QSA K-pool selection. The prompt is a file (QWEN_LONG_PROMPT_FILE) so the pinned reference (llama-completion) reads the
/// same text. Greedy continuation, per-token forward, the reference path. Skipped unless the file and the trunk are present.
/// </summary>
public sealed class Qwen4ExpLongPromptTests
{
    private const int IndexerTopK = 2048;
    private const int Steps = 16;

    private static string? FindModel() => ModelLocator.FindDeep(Qwen4ExpTestSupport.TrunkFileName);

    [Fact]
    public void LongPrompt_PastTheIndexerTopK_GreedyContinuationIsPrinted()
    {
        string? promptFile = Environment.GetEnvironmentVariable("QWEN_LONG_PROMPT_FILE");
        Assert.SkipUnless(promptFile is not null && File.Exists(promptFile), "Set QWEN_LONG_PROMPT_FILE to a text file.");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        string prompt = File.ReadAllText(promptFile!);
        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] ids = tok.Encode(prompt).ToArray();
        Console.WriteLine($"LONG_PROMPT_TOKENS {ids.Length} indexer_top_k {IndexerTopK}");
        Assert.True(ids.Length > IndexerTopK, $"the prompt is {ids.Length} tokens; it must exceed the indexer top-k ({IndexerTopK})");

        var state = pass.CreateSequenceState();
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in ids) logits = pass.Forward(id, pos++, state);

        var generated = new List<int>();
        for (int s = 0; s < Steps; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            if (tok.EogTokenIds.Contains(best)) break;
            generated.Add(best);
            logits = pass.Forward(best, pos++, state);
        }

        string answer = tok.Decode(generated);
        Console.WriteLine($"LONG_IDS {string.Join(",", generated)}");
        Console.WriteLine($"LONG_TEXT {answer}");
        if (Environment.GetEnvironmentVariable("QWEN_LONG_OUT_FILE") is { } outFile)
            File.WriteAllText(outFile, answer);
        Assert.False(string.IsNullOrWhiteSpace(answer));
    }
}
