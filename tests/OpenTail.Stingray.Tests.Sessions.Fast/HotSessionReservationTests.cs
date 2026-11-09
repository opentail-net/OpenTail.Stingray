using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Job 3 reservation projection, Qwen 3.8 plan: a turn that starts after earlier positions reserves the bytes of the prior state
/// when the engine keeps a turn-start snapshot (<see cref="HotSession"/> <c>TurnBytes</c>). Isolated here by giving the same
/// session budget to two engines that differ only in <c>KeepsTurnSnapshot</c>: the limit fits the turn without the snapshot term
/// and not the turn with it. Synthetic; no model.
/// </summary>
public sealed class HotSessionReservationTests
{
    /// <summary>Residency after turn 1 and after turn 2, with no limit and no snapshot.</summary>
    private static async Task<(long AfterTurn1, long AfterTurn2)> ResidencyAsync()
    {
        using var engine = new ContinuousBatchingEngine(
            new HotSessionSnapshotAccountingTests.SnapshotForwardPass(0, keepsTurnSnapshot: false),
            new HotSessionSnapshotAccountingTests.Tokenizer(), "res", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new HotSessionSnapshotAccountingTests.Tokenizer(),
            new HotSessionRuntimeOptions(maxResidentBytes: long.MaxValue));
        using var session = runtime.Create();
        var sampling = HotSessionSnapshotAccountingTests.Sampling;
        await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(),
            HotSessionSnapshotAccountingTests.Digest("hello"));
        long after1 = runtime.ResidentBytes;
        await session.RunTurnAsync("world", sampling, runtime.GetSessionSnapshot(session.SessionId).CommittedRevision,
            SessionOperationId.New(), HotSessionSnapshotAccountingTests.Digest("world"));
        return (after1, runtime.ResidentBytes);
    }

    /// <summary>Runs two turns under a per-session limit. Returns the exception the second turn raised, or null.</summary>
    private static async Task<Exception?> SecondTurnUnderLimitAsync(bool keepsTurnSnapshot, long limit)
    {
        using var engine = new ContinuousBatchingEngine(
            new HotSessionSnapshotAccountingTests.SnapshotForwardPass(500, keepsTurnSnapshot),
            new HotSessionSnapshotAccountingTests.Tokenizer(), "lim", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, new HotSessionSnapshotAccountingTests.Tokenizer(),
            new HotSessionRuntimeOptions(maxResidentBytes: limit, maxSessionBytes: limit));
        using var session = runtime.Create();
        var sampling = HotSessionSnapshotAccountingTests.Sampling;
        await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(),
            HotSessionSnapshotAccountingTests.Digest("hello"));
        try
        {
            await session.RunTurnAsync("world", sampling, runtime.GetSessionSnapshot(session.SessionId).CommittedRevision,
                SessionOperationId.New(), HotSessionSnapshotAccountingTests.Digest("world"));
            return null;
        }
        catch (Exception ex)
        {
            return ex;
        }
    }

    [Fact]
    public async Task TurnFromASnapshot_ReservesThePriorStatesBytes()
    {
        var (after1, after2) = await ResidencyAsync();
        // The limit covers the second turn's own positions but not the prior state on top of them.
        long limit = after2 + after1 / 2;

        Assert.Null(await SecondTurnUnderLimitAsync(keepsTurnSnapshot: false, limit));
        var refused = await SecondTurnUnderLimitAsync(keepsTurnSnapshot: true, limit);
        Assert.IsType<SessionResourceBudgetExceededException>(refused);
    }
}
