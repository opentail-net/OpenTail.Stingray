using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 2 admission (Qwen 3.8 plan): under a KV budget that fits one request at a time, a second request queues rather
/// than being admitted over budget, and every queued request still finishes with the same output. Synthetic.
/// </summary>
public sealed class ContinuousBatchingBackpressureTests
{
    private const long BudgetTokens = 12;     // one request needs about 8 (prompt 2 + 6 new), so two cannot fit

    [Fact]
    public async Task QueuedRequests_NeverExceedTheKvBudget_AndAllFinishUnchanged()
    {
        var tok = new ContinuousBatchingCancellationMemoryTests.ByteTokenizer();
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(60));

        // Baseline: the same request alone, with the same budget.
        using (var solo = new ContinuousBatchingEngine(
            new ContinuousBatchingCancellationMemoryTests.CountingForwardPass(), tok, "solo",
            maxBatchSize: 2, kvBudgetBytes: BudgetTokens))
        {
            string baseline = await CollectAsync(solo, "QR", sp, cts.Token);
            Assert.Equal(6, baseline.Length);

            using var engine = new ContinuousBatchingEngine(
                new ContinuousBatchingCancellationMemoryTests.CountingForwardPass(), tok, "queued",
                maxBatchSize: 2, kvBudgetBytes: BudgetTokens);

            long maxCommitted = 0;
            using var stop = new CancellationTokenSource();
            var poll = Task.Run(async () =>
            {
                while (!stop.IsCancellationRequested)
                {
                    maxCommitted = Math.Max(maxCommitted, engine.CommittedKvTokens);
                    await Task.Delay(1);
                }
            });

            var results = await Task.WhenAll(
                CollectAsync(engine, "QR", sp, cts.Token),
                CollectAsync(engine, "QR", sp, cts.Token),
                CollectAsync(engine, "QR", sp, cts.Token));

            stop.Cancel();
            await poll;

            Assert.All(results, r => Assert.Equal(baseline, r));
            Assert.True(maxCommitted <= BudgetTokens,
                $"committed KV reached {maxCommitted} tokens, above the budget of {BudgetTokens}.");
            Assert.True(maxCommitted > 0, "The poll never saw a reservation, so the budget check tests nothing.");
            Assert.Equal(0, engine.CommittedKvTokens);
        }
    }

    private static async Task<string> CollectAsync(
        ContinuousBatchingEngine engine, string prompt, SamplingParams sp, CancellationToken ct)
    {
        var sb = new StringBuilder();
        await foreach (var chunk in engine.GenerateAsync(prompt, sp, ct)) sb.Append(chunk);
        return sb.ToString();
    }
}
