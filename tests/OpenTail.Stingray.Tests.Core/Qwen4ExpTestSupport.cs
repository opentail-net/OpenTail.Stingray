using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Helpers shared by the Qwen 3.8 (qwen4exp) real-checkpoint tests: the checkpoint file name, greedy argmax, and the top-5
/// membership test of the D1 token criterion. One copy, so each test checks the same thing the same way.
/// </summary>
internal static class Qwen4ExpTestSupport
{
    /// <summary>The first shard of the UD-IQ1_S checkpoint; the other shards are found next to it.</summary>
    public const string TrunkFileName = "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";

    public static string? TrunkPath() => ModelLocator.FindDeep(TrunkFileName);

    public static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    /// <summary>True when <paramref name="id"/> is one of the five largest logits.</summary>
    public static bool Top5Contains(ReadOnlySpan<float> logits, int id)
    {
        var top = new int[5];
        for (int k = 0; k < 5; k++)
        {
            int best = -1;
            for (int i = 0; i < logits.Length; i++)
            {
                if (Array.IndexOf(top, i, 0, k) >= 0) continue;
                if (best < 0 || logits[i] > logits[best]) best = i;
            }
            top[k] = best;
        }
        return Array.IndexOf(top, id) >= 0;
    }
}
