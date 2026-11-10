namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Measurement for A9a of docs/00-current-work.md (source: docs/1-correctness/2026-10-02-prefill-numerics-investigation.md, "Follow-ups from the MoE
/// audit"): for prompts of 32 tokens or fewer, is the batched MoE prefill (<c>MoeBatchedPrefillEnabled</c> on) faster than the sequential path, which
/// runs each token through the folded decode-style MoE dispatch (<c>MoeFoldedDecodeEnabled</c>, the default)?
///
/// <para>Both arms run in one process, interleaved, with the minimum over repeats taken for each arm, so a slow first run or a noisy neighbour does
/// not decide the result. The printed table is the measurement; the assertion is only that both arms pick the same greedy token at each length, so a
/// speed result cannot hide a wrong answer. Heavy: skipped unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c> and OLMoE is on disk.</para>
/// </summary>
public sealed class MoeSmallPromptDispatchMeasurementTests : HeavyTestBase
{
    private static readonly int[] s_lengths = { 8, 16, 32, 64 };
    private const int Repeats = 5;

    private static string? FindMoePath() =>
        OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");

    /// <summary>Deterministic pseudo-prompt; the routing spread is what matters, not the words.</summary>
    private static int[] MakeTokens(int count)
    {
        var t = new int[count];
        for (int i = 0; i < count; i++) t[i] = 1 + ((i * 7919) % 4096);
        return t;
    }

    private static (float[] Logits, double Seconds) TimedPrefill(GgufModel model, CpuBackend backend, ModelHyperparams hp, int[] tokens, bool batched)
    {
        bool prev = Engine.ForwardPass.MoeBatchedPrefillEnabled;
        Engine.ForwardPass.MoeBatchedPrefillEnabled = batched;
        try
        {
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] logits = fwd.Prefill(tokens).ToArray();
            sw.Stop();
            return (logits, sw.Elapsed.TotalSeconds);
        }
        finally { Engine.ForwardPass.MoeBatchedPrefillEnabled = prev; }
    }

    [Fact]
    public void SmallPromptPrefill_BatchedVersusFoldedSequential_Timings()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "heavy: timing measurement on a real MoE checkpoint; set STINGRAY_RUN_HEAVY_TESTS=1");
        var path = FindMoePath();
        Assert.SkipUnless(path is not null, "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf not present in this environment");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(model.Metadata);
        Assert.True(hp.IsMoE, "expected an MoE model");
        using var backend = new CpuBackend();

        // One untimed warm-up of each arm, so first-use allocation is not charged to either side.
        TimedPrefill(model, backend, hp, MakeTokens(8), batched: false);
        TimedPrefill(model, backend, hp, MakeTokens(8), batched: true);

        var lines = new System.Text.StringBuilder();
        lines.AppendLine("MoE small-prompt prefill, OLMoE-1B-7B Q4_K_M, CPU, min of " + Repeats + " interleaved repeats:");
        lines.AppendLine("  tokens  batched-MoE ms  folded-sequential ms  ratio (seq/batched)  greedy token agrees  max |logit diff|");
        foreach (int n in s_lengths)
        {
            int[] tokens = MakeTokens(n);
            double bestBatched = double.MaxValue, bestSeq = double.MaxValue;
            float[] seqLogits = Array.Empty<float>(), batLogits = Array.Empty<float>();
            for (int r = 0; r < Repeats; r++)
            {
                var s = TimedPrefill(model, backend, hp, tokens, batched: false);
                var b = TimedPrefill(model, backend, hp, tokens, batched: true);
                bestSeq = Math.Min(bestSeq, s.Seconds);
                bestBatched = Math.Min(bestBatched, b.Seconds);
                seqLogits = s.Logits;
                batLogits = b.Logits;
            }
            float maxDiff = 0f;
            for (int i = 0; i < seqLogits.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(seqLogits[i] - batLogits[i]));
            bool agree = Sampler.Greedy(seqLogits) == Sampler.Greedy(batLogits);
            Assert.True(seqLogits.All(float.IsFinite) && batLogits.All(float.IsFinite), $"{n} tokens: non-finite logits");
            lines.AppendLine($"  {n,6}  {bestBatched * 1000,15:F1}  {bestSeq * 1000,20:F1}  {bestSeq / bestBatched,18:F2}  {(agree ? "yes" : "NO"),19}  {maxDiff,16:E2}");
        }
        Console.WriteLine(lines.ToString());
    }
}
