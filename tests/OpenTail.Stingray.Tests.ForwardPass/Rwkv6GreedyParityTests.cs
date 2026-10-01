using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// RWKV-6 (rwkv6) parity against llama-server (vendored tools/llama.cpp, same GGUF, /completion with
/// temperature 0, top_k 1, repeat_penalty 1, raw prompt, no BOS; captured 2026-10-01) on
/// rwkv-6-world-1.6b Q8_0. Prompt token ids are llama-server's /tokenize output; the
/// continuations are its greedy token ids.
/// <para>Teacher-forced: llama-server's tokens are fed back, and at every step our argmax must equal
/// llama's choice, except at positions listed as ties, where llama's own top-2 log-probs are within
/// a few hundredths of a nat (there, quantized-activation rounding legitimately decides the order).
/// Measured: 16/16 and 31/32 argmax agreement, the miss being a 0.007-nat tie in llama's own top-2
/// (token 38621 -2.1126 vs 56837 -2.1196); chosen-token |Δlogprob| mean 0.022 / 0.031, max 0.15.</para>
/// </summary>
public sealed class Rwkv6GreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "rwkv-6-world-1.6b-Q8_0.gguf";

    public static TheoryData<string, int[], int[], int[]> Cases => new()
    {
        {
            // " Paris.\nThe capital of the United States is Washington, D.C."
            "The capital of France is",
            [6699, 51128, 4706, 44312, 4600],
            [37138, 47, 11, 6699, 51128, 4706, 22590, 45010, 44910, 4600, 61509, 45, 303, 47, 68, 47],
            []
        },
        {
            // " had a granddaughter named Lily. Lily was a curious girl who loved to explore the world
            // around her. One day, she asked her grandpa, \""
            "Once upon a time, in a small village by the sea, there lived an old fisherman who",
            [23977, 32350, 332, 32251, 45, 4596, 332, 39720, 53287, 4450, 22590, 22446, 45, 39934, 38917, 4419, 22221, 45998, 8148, 22762],
            [21795, 332, 38621, 53893, 39095, 29281, 47, 29281, 22748, 332, 51371, 30971, 22762, 38937, 4811, 51691,
             22590, 40213, 45202, 21811, 47, 20556, 21509, 45, 22464, 37756, 21811, 38621, 2111, 45, 269, 33139],
            [2] // 38621 -2.1126 vs 56837 -2.1196 in llama-server's own top_logprobs
        },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void TeacherForcedArgmax_MatchesLlamaServer(string prompt, int[] promptTokens, int[] expected, int[] tiePositions)
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var model = GgufModel.Open(path!);
        Assert.Equal("rwkv6", Convert.ToString(model.Metadata["general.architecture"]));
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        Assert.Equal(promptTokens, tokenizer.Encode(prompt));

        using var fwd = new Rwkv6ForwardPass(model);
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

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir, sub, ModelFile);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        var external = Path.Combine(@"E:\_models\rwkv6-world-1b6", ModelFile);
        return File.Exists(external) ? external : null;
    }
}
