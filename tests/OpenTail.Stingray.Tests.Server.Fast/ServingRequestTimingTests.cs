using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// <see cref="ServingRequestTiming"/> in isolation. The endpoint-level tests
/// (<see cref="ResponseTimingExtensionTests"/>, <see cref="StatusDocumentTests"/>) only see values
/// well under a millisecond from an in-memory fake engine, so they cannot distinguish "generation
/// duration excludes delivery time" from "generation duration happens to be small" — these tests
/// use real, deliberate sleeps to make that distinction unambiguous.
/// </summary>
public sealed class ServingRequestTimingTests
{
    /// <summary>
    /// Regression test: <c>Complete()</c>/<c>Dispose()</c> fires from a <c>using</c> block at
    /// endpoint-method exit, which on the non-streaming path is after the full response has been
    /// serialized and written to the client. Recording generation duration at that point — as the
    /// code did before <see cref="ServingRequestTiming.MarkGenerationComplete"/> existed — means a
    /// slow client or a large response inflates the reported decode/inter-token rate with delivery
    /// time that has nothing to do with the engine. This pins that calling
    /// <see cref="ServingRequestTiming.MarkGenerationComplete"/> right after the token loop finishes
    /// keeps that delivery time out of the sample.
    /// </summary>
    [Fact]
    public void MarkGenerationComplete_ExcludesTimeSpentAfterItFromGenerationDuration()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.MarkFirstToken();
        Thread.Sleep(30); // stands in for real decode time between tokens
        timing.MarkGenerationComplete();

        // Stands in for slow response serialization/delivery to the client — must NOT be counted
        // as generation time. Comfortably larger than the 30ms above and any scheduling jitter.
        Thread.Sleep(200);
        timing.Dispose();

        var generation = metrics.GenerationDurationSummary;
        Assert.Equal(1, generation.Count);
        // A hard ceiling well below the 200ms "delivery" sleep: if delivery time leaked in, this
        // would be >= 200ms. Generous above the 30ms "decode" sleep to absorb CI scheduling jitter.
        Assert.True(generation.TotalSeconds < 0.15,
            $"generation duration was {generation.TotalSeconds * 1000:F0} ms; " +
            "post-MarkGenerationComplete delivery time leaked into the sample.");

        // Request duration is the OTHER histogram and is documented to include delivery — it must
        // still see the full ~230ms, or this test would not be distinguishing the two at all.
        var request = metrics.RequestDurationSummary;
        Assert.True(request.TotalSeconds >= 0.2,
            $"request duration was {request.TotalSeconds * 1000:F0} ms; expected it to include the delivery sleep.");
    }

    /// <summary>
    /// Calling <see cref="ServingRequestTiming.MarkGenerationComplete"/> a second time (e.g. a
    /// call site that both finishes its loop AND later re-enters an error path) must not record a
    /// second, larger sample — the first call wins.
    /// </summary>
    [Fact]
    public void MarkGenerationComplete_SecondCallIsANoOp()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.MarkFirstToken();
        timing.MarkGenerationComplete();
        Thread.Sleep(50);
        timing.MarkGenerationComplete(); // must be ignored
        timing.Dispose();

        Assert.Equal(1, metrics.GenerationDurationSummary.Count);
    }

    /// <summary>
    /// A request that is cancelled or throws before the generation loop finishes never calls
    /// <see cref="ServingRequestTiming.MarkGenerationComplete"/>. <c>Complete()</c>'s fallback must
    /// still record a sample (using dispose time, same as the pre-fix behavior) so those requests
    /// don't silently vanish from the histogram.
    /// </summary>
    [Fact]
    public void Complete_FallsBackToDisposeTimeWhenGenerationWasNeverMarkedComplete()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.MarkFirstToken();
        timing.Dispose(); // no MarkGenerationComplete() call — simulates an error/cancellation path

        Assert.Equal(1, metrics.GenerationDurationSummary.Count);
    }

    /// <summary>A request with no tokens at all (error before the first token) records neither histogram entry.</summary>
    [Fact]
    public void NoFirstToken_RecordsNoGenerationDurationSample()
    {
        var metrics = new ServerMetrics();
        var timing = metrics.BeginServingRequest();

        timing.Dispose();

        Assert.Equal(0, metrics.GenerationDurationSummary.Count);
        Assert.Equal(1, metrics.RequestDurationSummary.Count);
    }
}
