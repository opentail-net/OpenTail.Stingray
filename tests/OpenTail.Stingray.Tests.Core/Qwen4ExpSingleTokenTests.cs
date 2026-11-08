using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 6.3, real checkpoint: a one-token prompt, then five greedy decode steps. The prompt must tokenize to
/// exactly one token, so the test checks the smallest prefill the paged path can take. Skipped when the trunk is absent.
/// </summary>
public sealed class Qwen4ExpSingleTokenTests
{
    private const string Prompt = "Hello";
    private const int Steps = 5;

    private static string? FindModel() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void SingleTokenPrompt_FiveDecodeSteps_RunsAndIsPrinted()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] ids = tok.Encode(Prompt).ToArray();
        Console.WriteLine($"SINGLE_PROMPT_IDS {string.Join(",", ids)}");
        Assert.Single(ids);

        var state = pass.CreateSequenceState();
        ReadOnlySpan<float> logits = pass.Forward(ids[0], 0, state);
        var generated = new List<int>();
        int pos = 1;
        for (int s = 0; s < Steps; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            generated.Add(best);
            logits = pass.Forward(best, pos++, state);
        }

        string answer = tok.Decode(generated);
        Console.WriteLine($"SINGLE_IDS {string.Join(",", generated)}");
        Console.WriteLine($"SINGLE_TEXT {answer}");
        Assert.Equal(Steps, generated.Count);
    }
}
