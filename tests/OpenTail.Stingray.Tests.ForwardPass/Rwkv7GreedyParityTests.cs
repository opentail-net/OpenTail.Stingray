using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// RWKV-7 (rwkv7) parity against llama-server (vendored tools/llama.cpp, same GGUF, /completion with
/// temperature 0, top_k 1, repeat_penalty 1, raw prompt, no BOS; captured 2026-10-01) on
/// RWKV7-Goose-World3-1.5B-HF Q8_0. Prompt token ids are llama-server's /tokenize output; the
/// continuations are its greedy token ids.
/// <para>Teacher-forced: llama-server's tokens are fed back, and at every step our argmax must equal
/// llama's choice, except at positions listed as ties, where llama's own top-2 log-probs are within
/// a few hundredths of a nat (there, quantized-activation rounding legitimately decides the order).
/// Measured on the 32-token case: 31/32 argmax agreement, the miss being llama's " far" -0.8717 vs
/// " for" -0.8986; chosen-token |Δlogprob| mean 0.013, max 0.11.</para>
/// </summary>
public sealed class Rwkv7GreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "RWKV7-Goose-World3-1.5B-HF.Q8_0.gguf";

    public static TheoryData<string, int[], int[], int[]> Cases => new()
    {
        {
            // " Paris.\nThe capital of France is Paris.\nThe capital of France is"
            "The capital of France is",
            [6699, 51128, 4706, 44312, 4600],
            [37138, 47, 11, 6699, 51128, 4706, 44312, 4600, 37138, 47, 11, 6699, 51128, 4706, 44312, 4600],
            []
        },
        {
            // " was known far and wide for his wisdom and kindness. He had a young apprentice named
            // Timmy, who was eager to learn everything he could from his master"
            "Once upon a time, in a small village by the sea, there lived an old fisherman who",
            [23977, 32350, 332, 32251, 45, 4596, 332, 39720, 53287, 4450, 22590, 22446, 45, 39934, 38917, 4419, 22221, 45998, 8148, 22762],
            [22748, 38848, 21660, 21265, 32470, 21700, 21823, 47802, 21265, 56520, 47, 3878, 21795, 332, 40240, 50941,
             7759, 39095, 21006, 2058, 45, 22762, 22748, 38326, 4811, 38877, 61960, 4569, 38128, 30917, 21823, 46526],
            [2] // " far" -0.8717 vs " for" -0.8986 in llama-server's own top_logprobs
        },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void TeacherForcedArgmax_MatchesLlamaServer(string prompt, int[] promptTokens, int[] expected, int[] tiePositions)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var model = GgufModel.Open(path!);
        Assert.Equal("rwkv7", Convert.ToString(model.Metadata["general.architecture"]));
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        Assert.Equal(promptTokens, tokenizer.Encode(prompt));

        using var fwd = new Rwkv7ForwardPass(model);
        var logits = fwd.Prefill(promptTokens);
        int pos = promptTokens.Length;
        var mismatches = new List<string>();
        for (int i = 0; i < expected.Length; i++)
        {
            int ours = Sampler.Greedy(logits);
            if (ours != expected[i] && !tiePositions.Contains(i))
                mismatches.Add($"step {i}: ours {ours}, llama {expected[i]}");
            if (i + 1 < expected.Length) logits = fwd.Forward(expected[i], pos++);
        }
        Assert.True(mismatches.Count == 0, string.Join("; ", mismatches));
    }

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindOrReport(ModelFile);
}
