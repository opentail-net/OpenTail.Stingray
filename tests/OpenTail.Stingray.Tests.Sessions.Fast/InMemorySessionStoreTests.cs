using OpenTail.Stingray.Sessions;
using System.Collections.Immutable;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public sealed class InMemorySessionStoreTests
{
    [Fact]
    public void Complete_AdvancesRevisionExactlyOnce()
    {
        var store = new InMemorySessionStore();
        var session = store.Create();
        var lease = store.AcquireLease(session.SessionId);
        var operation = store.Begin(lease, SessionRevision.Initial, SessionOperationId.New(), Digest("turn-1"));

        store.Transition(lease, operation.OperationId, SessionOperationState.Accepted, SessionOperationState.Prefilling);
        store.Transition(lease, operation.OperationId, SessionOperationState.Prefilling, SessionOperationState.Generating);
        store.Transition(lease, operation.OperationId, SessionOperationState.Generating, SessionOperationState.CommitPrepared);
        var completed = store.Complete(lease, operation.OperationId);

        Assert.Equal(SessionOperationState.Completed, completed.State);
        Assert.Equal(new SessionRevision(1), completed.CommittedRevision);
        Assert.Equal(new SessionRevision(1), store.Open(session.SessionId).CommittedRevision);
    }

    [Fact]
    public void DuplicateOperation_WithSameDigestIsIdempotent_ButDifferentDigestIsRejected()
    {
        var store = new InMemorySessionStore();
        var session = store.Create();
        var lease = store.AcquireLease(session.SessionId);
        var id = SessionOperationId.New();

        var first = store.Begin(lease, SessionRevision.Initial, id, Digest("turn-1"));
        var replay = store.Begin(lease, SessionRevision.Initial, id, Digest("turn-1"));

        Assert.Same(first, replay);
        Assert.Throws<SessionOperationConflictException>(() =>
            store.Begin(lease, SessionRevision.Initial, id, Digest("different-turn")));
    }

    [Fact]
    public void StaleLease_CannotTransitionOrCommit()
    {
        var store = new InMemorySessionStore();
        var session = store.Create();
        var firstLease = store.AcquireLease(session.SessionId);
        var operation = store.Begin(firstLease, SessionRevision.Initial, SessionOperationId.New(), Digest("turn-1"));
        var currentLease = store.AcquireLease(session.SessionId);

        Assert.Throws<SessionFencedException>(() =>
            store.Transition(firstLease, operation.OperationId, SessionOperationState.Accepted, SessionOperationState.Prefilling));

        var transitioned = store.Transition(currentLease, operation.OperationId,
            SessionOperationState.Accepted, SessionOperationState.Prefilling);
        Assert.Equal(SessionOperationState.Prefilling, transitioned.State);
    }

    [Fact]
    public void CancelAndFailure_DoNotAdvanceRevision()
    {
        var store = new InMemorySessionStore();
        var session = store.Create();
        var lease = store.AcquireLease(session.SessionId);

        var cancelled = store.Begin(lease, SessionRevision.Initial, SessionOperationId.New(), Digest("cancel"));
        store.Cancel(lease, cancelled.OperationId);
        var failed = store.Begin(lease, SessionRevision.Initial, SessionOperationId.New(), Digest("failure"));
        store.Fail(lease, failed.OperationId, "test failure");

        var snapshot = store.Open(session.SessionId);
        Assert.Equal(SessionRevision.Initial, snapshot.CommittedRevision);
        Assert.Null(snapshot.DurableRevision);
        Assert.Equal(SessionOperationState.Cancelled, Assert.Single(snapshot.Operations, x => x.OperationId == cancelled.OperationId).State);
        Assert.Equal(SessionOperationState.Failed, Assert.Single(snapshot.Operations, x => x.OperationId == failed.OperationId).State);
    }

    [Fact]
    public void CompletedOperations_AreBoundedWhileTheRevisionRemainsMonotonic()
    {
        var store = new InMemorySessionStore(maxCompletedOperationRecords: 2);
        var session = store.Create();
        var lease = store.AcquireLease(session.SessionId);

        for (int i = 0; i < 3; i++)
        {
            var operation = store.Begin(lease, new SessionRevision(i), SessionOperationId.New(), Digest($"turn-{i}"));
            store.Transition(lease, operation.OperationId, SessionOperationState.Accepted, SessionOperationState.Prefilling);
            store.Transition(lease, operation.OperationId, SessionOperationState.Prefilling, SessionOperationState.Generating);
            store.Transition(lease, operation.OperationId, SessionOperationState.Generating, SessionOperationState.CommitPrepared);
            store.Complete(lease, operation.OperationId);
        }

        var snapshot = store.Open(session.SessionId);
        Assert.Equal(new SessionRevision(3), snapshot.CommittedRevision);
        Assert.Equal(2, snapshot.Operations.Count);
        Assert.All(snapshot.Operations, operation => Assert.Equal(SessionOperationState.Completed, operation.State));
    }

    [Fact]
    public void Open_PrunesExpiredCompletedOperationsForIdleSessions()
    {
        var clock = new MutableTimeProvider(DateTimeOffset.UtcNow);
        var store = new InMemorySessionStore(completedOperationRetention: TimeSpan.FromMinutes(1), clock: clock);
        var session = store.Create();
        var lease = store.AcquireLease(session.SessionId);
        var operation = store.Begin(lease, SessionRevision.Initial, SessionOperationId.New(), Digest("expired"));
        store.Transition(lease, operation.OperationId, SessionOperationState.Accepted, SessionOperationState.Prefilling);
        store.Transition(lease, operation.OperationId, SessionOperationState.Prefilling, SessionOperationState.Generating);
        store.Transition(lease, operation.OperationId, SessionOperationState.Generating, SessionOperationState.CommitPrepared);
        store.Complete(lease, operation.OperationId);

        clock.Advance(TimeSpan.FromMinutes(2));

        Assert.Empty(store.Open(session.SessionId).Operations);
    }

    private sealed class MutableTimeProvider(DateTimeOffset now) : TimeProvider
    {
        private DateTimeOffset _now = now;
        public override DateTimeOffset GetUtcNow() => _now;
        public void Advance(TimeSpan duration) => _now += duration;
    }

    private static SessionRequestDigest Digest(string value) => SessionRequestDigest.FromCanonicalValue(value);
}

public sealed class ExecutionHistoryTests
{
    [Fact]
    public void CursorCodec_RoundTripsTokenAndAtomicExecutionSegments()
    {
        ImmutableArray<ExecutionSegment> log =
        [
            new TokenSegment([1, 2, 3]),
            new AtomicExecutionSegment("image", 4, ContentDigest.FromCanonicalText("image-1")),
            new TokenSegment([4]),
        ];
        var cursor = new SessionCursor(log, 8, 8, 8, 8, StateCoverage.Full);

        var restored = SessionCursorCodec.Decode(SessionCursorCodec.Encode(cursor));

        Assert.Equal(cursor.AcceptedPositionCount, restored.AcceptedPositionCount);
        Assert.Equal(cursor.MaterializedPositionCount, restored.MaterializedPositionCount);
        Assert.Equal(cursor.NextLogicalPosition, restored.NextLogicalPosition);
        Assert.Equal(cursor.PhysicalSlotCount, restored.PhysicalSlotCount);
        Assert.Equal(cursor.Coverage, restored.Coverage);
        Assert.Equal(3, restored.ExecutionLog.Length);
        Assert.Equal([1, 2, 3], Assert.IsType<TokenSegment>(restored.ExecutionLog[0]).TokenIds);
        var atomic = Assert.IsType<AtomicExecutionSegment>(restored.ExecutionLog[1]);
        Assert.Equal("image", atomic.Kind);
        Assert.Equal(4, atomic.Positions);
        Assert.Equal(ContentDigest.FromCanonicalText("image-1"), atomic.CanonicalInputDigest);
        Assert.Equal([4], Assert.IsType<TokenSegment>(restored.ExecutionLog[2]).TokenIds);
        Assert.Equal(cursor.InputIdentity, restored.InputIdentity);
    }

    [Fact]
    public void CursorCodec_RejectsInvalidMagicAndHostileSegmentCountBeforeAllocation()
    {
        var cursor = new SessionCursor([], 0, 0, 0, 0, StateCoverage.Full);
        var invalidMagic = SessionCursorCodec.Encode(cursor);
        invalidMagic[0] ^= 0xFF;
        Assert.Throws<SessionCursorFormatException>(() => SessionCursorCodec.Decode(invalidMagic));

        var hostileCount = SessionCursorCodec.Encode(cursor);
        // The required cursor section begins at byte 19; its segment count is after four counts and coverage.
        BitConverter.GetBytes(int.MaxValue).CopyTo(hostileCount, 19 + 21 - sizeof(int));
        Assert.Throws<SessionCursorFormatException>(() => SessionCursorCodec.Decode(hostileCount));
    }

    [Fact]
    public void CursorCodec_SkipsUnknownOptionalSectionsAndRefusesUnknownRequiredSections()
    {
        var cursor = new SessionCursor([new TokenSegment([1])], 1, 1, 1, 1, StateCoverage.Full);
        var optional = AddUnknownSection(SessionCursorCodec.Encode(cursor), required: false);

        var decoded = SessionCursorCodec.DecodeEnvelope(optional);
        Assert.Equal(cursor.InputIdentity, decoded.Cursor.InputIdentity);
        Assert.Single(decoded.OptionalSections);
        Assert.Equal((ushort)99, decoded.OptionalSections[0].Id);
        Assert.Equal([0xAA, 0xBB, 0xCC], decoded.OptionalSections[0].Payload);
        var rewritten = SessionCursorCodec.DecodeEnvelope(SessionCursorCodec.Encode(decoded));
        Assert.Single(rewritten.OptionalSections);
        Assert.Equal(decoded.OptionalSections[0].Id, rewritten.OptionalSections[0].Id);
        Assert.Equal(decoded.OptionalSections[0].Payload, rewritten.OptionalSections[0].Payload);

        var required = AddUnknownSection(SessionCursorCodec.Encode(cursor), required: true);
        Assert.Throws<SessionCursorFormatException>(() => SessionCursorCodec.Decode(required));
    }

    [Fact]
    public void CursorCodec_DecodesV1AndRejectsUnreferencedEnvelopeBytes()
    {
        var cursor = new SessionCursor([new TokenSegment([1, 2])], 2, 2, 2, 2, StateCoverage.Full);
        var legacy = ToV1(SessionCursorCodec.Encode(cursor));
        Assert.Equal(cursor.InputIdentity, SessionCursorCodec.Decode(legacy).InputIdentity);

        var unreferenced = SessionCursorCodec.Encode(cursor).Append((byte)0xFF).ToArray();
        Assert.Throws<SessionCursorFormatException>(() => SessionCursorCodec.Decode(unreferenced));
    }

    private static byte[] AddUnknownSection(byte[] payload, bool required)
    {
        const int headerBytes = 8, entryBytes = 11, oldCursorOffset = headerBytes + entryBytes;
        int cursorLength = payload.Length - oldCursorOffset;
        var expanded = new byte[payload.Length + entryBytes + 3];
        Array.Copy(payload, 0, expanded, 0, headerBytes);
        BitConverter.GetBytes((ushort)2).CopyTo(expanded, 6);
        BitConverter.GetBytes((ushort)1).CopyTo(expanded, 8);
        expanded[10] = 1;
        BitConverter.GetBytes(headerBytes + 2 * entryBytes).CopyTo(expanded, 11);
        BitConverter.GetBytes(cursorLength).CopyTo(expanded, 15);
        BitConverter.GetBytes((ushort)99).CopyTo(expanded, 19);
        expanded[21] = required ? (byte)1 : (byte)0;
        BitConverter.GetBytes(headerBytes + 2 * entryBytes + cursorLength).CopyTo(expanded, 22);
        BitConverter.GetBytes(3).CopyTo(expanded, 26);
        Array.Copy(payload, oldCursorOffset, expanded, headerBytes + 2 * entryBytes, cursorLength);
        expanded[^3] = 0xAA;
        expanded[^2] = 0xBB;
        expanded[^1] = 0xCC;
        return expanded;
    }

    private static byte[] ToV1(byte[] v2)
    {
        int cursorOffset = BitConverter.ToInt32(v2, 11);
        int cursorLength = BitConverter.ToInt32(v2, 15);
        var legacy = new byte[6 + cursorLength];
        Array.Copy(v2, 0, legacy, 0, 4);
        BitConverter.GetBytes((ushort)1).CopyTo(legacy, 4);
        Array.Copy(v2, cursorOffset, legacy, 6, cursorLength);
        return legacy;
    }

    [Fact]
    public void CursorCodec_EnforcesSharedEncodeLimitsAndRejectsMalformedUtf8()
    {
        var oversized = new SessionCursor([new TokenSegment([1, 2])], 2, 2, 2, 2, StateCoverage.Full);
        Assert.Throws<SessionCursorFormatException>(() => SessionCursorCodec.Encode(oversized,
            new SessionCursorCodecLimits(MaxTokensPerSegment: 1)));

        var atomic = new SessionCursor(
            [new AtomicExecutionSegment("a", 1, ContentDigest.FromCanonicalText("input"))],
            1, 1, 1, 1, StateCoverage.Full);
        var malformedUtf8 = SessionCursorCodec.Encode(atomic);
        malformedUtf8[45] = 0xFF; // first byte of the first atomic segment's UTF-8 kind
        Assert.Throws<SessionCursorFormatException>(() => SessionCursorCodec.Decode(malformedUtf8));
        Assert.Throws<ArgumentException>(() => new ContentDigest(new string('A', 64)));
        Assert.Throws<ArgumentException>(() => new AtomicExecutionSegment("image", 1, default));
    }

    [Fact]
    public void Reconcile_MatchesTokenPrefixAndReportsExactDivergence()
    {
        ImmutableArray<ExecutionSegment> current = [new TokenSegment([1, 2, 3])];
        ImmutableArray<ExecutionSegment> target = [new TokenSegment([1, 2, 4])];

        var result = ExecutionReconciler.Compare(current, target);

        Assert.Equal(2, result.MatchedPositions);
        Assert.Equal(0, result.DivergenceSegmentIndex);
        Assert.Equal(2, result.DivergencePositionInSegment);
        Assert.Equal(SessionReuseReason.PrefixDivergence, result.ReuseReason);
    }

    [Fact]
    public void Diagnose_SeparatesExactAppendFromReplayRequiredDivergence()
    {
        ImmutableArray<ExecutionSegment> currentLog = [new TokenSegment([1, 2])];
        var cursor = new SessionCursor(currentLog, 2, 2, 2, 2, StateCoverage.Full);

        var append = ExecutionReconciler.Diagnose(cursor, [new TokenSegment([1, 2, 3])]);
        var mismatch = ExecutionReconciler.Diagnose(cursor, [new TokenSegment([1, 4])]);

        Assert.True(append.CanAppendWithoutReplay);
        Assert.Equal(ContinuationGrade.ExactLossless, append.Grade);
        Assert.Equal(SessionReuseReason.None, append.ReuseReason);
        Assert.False(mismatch.CanAppendWithoutReplay);
        Assert.Equal(ContinuationGrade.ReplayedFromExecutionLog, mismatch.Grade);
        Assert.Equal(SessionReuseReason.PrefixDivergence, mismatch.ReuseReason);
    }

    [Fact]
    public void Diagnose_TreatsAdjacentTokenSegmentsAsOneTokenStream()
    {
        var cursor = new SessionCursor([new TokenSegment([1, 2])], 2, 2, 2, 2, StateCoverage.Full);

        var diagnostic = ExecutionReconciler.Diagnose(cursor,
            [new TokenSegment([1]), new TokenSegment([2, 3])]);

        Assert.True(diagnostic.CanAppendWithoutReplay);
        Assert.Equal(2, diagnostic.MatchedPositions);
    }

    [Fact]
    public void InputIdentity_DistinguishesExecutionInputs_AndPayloadHashUsesCanonicalBytes()
    {
        ImmutableArray<ExecutionSegment> first = [new TokenSegment([1, 2]), new TokenSegment([3])];
        ImmutableArray<ExecutionSegment> second = [new TokenSegment([1, 2]), new TokenSegment([4])];

        Assert.NotEqual(InputIdentityHash.Compute(first), InputIdentityHash.Compute(second));
        Assert.Equal(StatePayloadHash.Compute([1, 2, 3]), StatePayloadHash.Compute([1, 2, 3]));
        Assert.NotEqual(StatePayloadHash.Compute([1, 2, 3]), StatePayloadHash.Compute([1, 2, 4]));
    }

    /// <summary>
    /// Milestone 1 required invariant: canonical payload hashing ignores inactive allocation tails.
    ///
    /// <para>State buffers are pooled and reused, so the bytes past the active length are whatever
    /// a previous turn left there. If the integrity hash covered them, two sessions holding
    /// identical state would hash differently purely because of allocator history — and
    /// <see cref="StatePayloadHash"/> is what answers "are these restored bytes intact", so that
    /// would make every restore look corrupt.</para>
    ///
    /// <para><b>On the strength of this test.</b> The property is currently guaranteed by the API
    /// shape — <c>Compute</c> takes a <see cref="ReadOnlySpan{T}"/>, so the caller slices and the
    /// tail is not reachable. That makes this a shape-locking regression guard rather than a deep
    /// behavioural check: it fails if the parameter ever becomes a <c>byte[]</c>, or if length
    /// padding is introduced. Said plainly rather than dressed up as more than it is.</para>
    /// </summary>
    [Fact]
    public void PayloadHash_IgnoresInactiveAllocationTail()
    {
        const int active = 4;
        // Same active prefix, deliberately different garbage in the reused tail.
        byte[] pooledA = [1, 2, 3, 4, 0xAA, 0xBB, 0xCC];
        byte[] pooledB = [1, 2, 3, 4, 0x11, 0x22, 0x33];

        Assert.Equal(
            StatePayloadHash.Compute(pooledA.AsSpan(0, active)),
            StatePayloadHash.Compute(pooledB.AsSpan(0, active)));

        // NON-VACUITY: the tails must actually differ, or the equality above proves nothing.
        Assert.NotEqual(
            StatePayloadHash.Compute(pooledA),
            StatePayloadHash.Compute(pooledB));

        // And the hash must still be sensitive to the ACTIVE bytes it does cover.
        byte[] changed = [1, 2, 3, 5, 0xAA, 0xBB, 0xCC];
        Assert.NotEqual(
            StatePayloadHash.Compute(pooledA.AsSpan(0, active)),
            StatePayloadHash.Compute(changed.AsSpan(0, active)));
    }

    [Fact]
    public void Cursor_RejectsImpossibleAcceptedAndMaterializedCounts()
    {
        ImmutableArray<ExecutionSegment> log = [new TokenSegment([1, 2])];

        Assert.Throws<ArgumentException>(() => new SessionCursor(log, 2, 3, 2, 2, StateCoverage.Full));
        Assert.Throws<ArgumentException>(() => new SessionCursor(log, 2, 2, 3, 2, StateCoverage.Full));
        Assert.Throws<ArgumentException>(() => new SessionCursor(log, 1, 1, 1, 1, StateCoverage.Full));
    }
}
