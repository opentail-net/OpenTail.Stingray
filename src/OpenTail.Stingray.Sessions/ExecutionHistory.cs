
namespace OpenTail.Stingray.Sessions;

/// <summary>How much of the logical sequence is represented by retained state.</summary>
public enum StateCoverage
{
    Full,
    PartialWindow,
    Recurrent,
}

/// <summary>Quality of a resumed inference state, surfaced rather than implied.</summary>
public enum ContinuationGrade
{
    ExactLossless,
    DeterministicEquivalent,
    NumericallyEquivalent,
    CodecBoundedLossy,
    ReplayedFromExecutionLog,
    PartialWindow,
    ColdStart,
}

/// <summary>Why a requested execution history could not reuse existing decoder state exactly.</summary>
public enum SessionReuseReason
{
    None,
    PrefixDivergence,
    RewindUnsupported,
    CoverageInsufficient,
    BackendIncompatible,
    StateUnavailable,
}

/// <summary>A SHA-256 value represented as lowercase hexadecimal.</summary>
public readonly record struct ContentDigest
{
    public string Value { get; }

    public ContentDigest(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (value.Length != 64 || value.Any(character => !((character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f'))))
            throw new ArgumentException("A content digest must be 64 lowercase hexadecimal SHA-256 characters.", nameof(value));
        Value = value;
    }

    public static ContentDigest FromCanonicalBytes(ReadOnlySpan<byte> bytes) =>
        new(Convert.ToHexStringLower(SHA256.HashData(bytes)));

    public static ContentDigest FromCanonicalText(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return FromCanonicalBytes(Encoding.UTF8.GetBytes(value));
    }
}

/// <summary>Identity of the requested model execution input.</summary>
public readonly record struct InputIdentityHash(ContentDigest Value)
{
    public static InputIdentityHash Compute(IEnumerable<ExecutionSegment> segments) =>
        new(ContentDigest.FromCanonicalBytes(ExecutionSegmentCodec.Canonicalize(segments)));
}

/// <summary>Integrity identity of a canonical active-state payload.</summary>
public readonly record struct StatePayloadHash(ContentDigest Value)
{
    public static StatePayloadHash Compute(ReadOnlySpan<byte> canonicalState) =>
        new(ContentDigest.FromCanonicalBytes(canonicalState));
}

/// <summary>One authoritative unit of model execution, distinct from user-facing transcript text.</summary>
public abstract record ExecutionSegment
{
    /// <summary>Logical decoder positions consumed by this segment.</summary>
    public abstract int PositionCount { get; }

    internal abstract void WriteCanonical(BinaryWriter writer);
}

/// <summary>Token IDs executed by the decoder.</summary>
public sealed record TokenSegment : ExecutionSegment
{
    public ImmutableArray<int> TokenIds { get; }
    public override int PositionCount => TokenIds.Length;

    public TokenSegment(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        TokenIds = tokenIds.ToImmutableArray();
        if (TokenIds.IsDefaultOrEmpty)
            throw new ArgumentException("A token segment must contain at least one token.", nameof(tokenIds));
    }

    internal override void WriteCanonical(BinaryWriter writer)
    {
        writer.Write((byte)1);
        writer.Write(TokenIds.Length);
        foreach (var token in TokenIds) writer.Write(token);
    }
}

/// <summary>
/// Atomic non-token execution input, such as an embedded-media position range. Its canonical
/// digest is part of input identity; it cannot be partially matched like a token segment.
/// </summary>
public sealed record AtomicExecutionSegment : ExecutionSegment
{
    public string Kind { get; }
    public int Positions { get; }
    public ContentDigest CanonicalInputDigest { get; }
    public override int PositionCount => Positions;

    public AtomicExecutionSegment(string kind, int positions, ContentDigest canonicalInputDigest)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kind);
        if (positions < 0) throw new ArgumentOutOfRangeException(nameof(positions));
        _ = new ContentDigest(canonicalInputDigest.Value
            ?? throw new ArgumentException("An atomic execution segment requires a content digest.", nameof(canonicalInputDigest)));
        Kind = kind;
        Positions = positions;
        CanonicalInputDigest = canonicalInputDigest;
    }

    internal override void WriteCanonical(BinaryWriter writer)
    {
        writer.Write((byte)2);
        writer.Write(Kind);
        writer.Write(Positions);
        writer.Write(CanonicalInputDigest.Value);
    }
}

/// <summary>Authoritative execution cursor for hot and persisted session state.</summary>
public sealed record SessionCursor
{
    public ImmutableArray<ExecutionSegment> ExecutionLog { get; }
    public int AcceptedPositionCount { get; }
    public int MaterializedPositionCount { get; }
    public int NextLogicalPosition { get; }
    public int PhysicalSlotCount { get; }
    public StateCoverage Coverage { get; }

    public SessionCursor(
        ImmutableArray<ExecutionSegment> executionLog,
        int acceptedPositionCount,
        int materializedPositionCount,
        int nextLogicalPosition,
        int physicalSlotCount,
        StateCoverage coverage)
    {
        if (executionLog.IsDefault) throw new ArgumentException("Execution log must not be default.", nameof(executionLog));
        if (acceptedPositionCount < 0 || materializedPositionCount < 0 || nextLogicalPosition < 0 || physicalSlotCount < 0)
            throw new ArgumentOutOfRangeException(nameof(acceptedPositionCount), "Cursor counts must be non-negative.");
        if (materializedPositionCount > acceptedPositionCount)
            throw new ArgumentException("Materialized positions cannot exceed accepted positions.", nameof(materializedPositionCount));
        if (nextLogicalPosition != acceptedPositionCount)
            throw new ArgumentException("Next logical position must equal accepted position count.", nameof(nextLogicalPosition));
        if (executionLog.Sum(x => x.PositionCount) != acceptedPositionCount)
            throw new ArgumentException("Execution log positions must equal accepted position count.", nameof(executionLog));
        ExecutionLog = executionLog;
        AcceptedPositionCount = acceptedPositionCount;
        MaterializedPositionCount = materializedPositionCount;
        NextLogicalPosition = nextLogicalPosition;
        PhysicalSlotCount = physicalSlotCount;
        Coverage = coverage;
    }

    public InputIdentityHash InputIdentity => InputIdentityHash.Compute(ExecutionLog);
}

/// <summary>Deterministic reconciliation result before a backend-specific rewind or append runs.</summary>
public sealed record ExecutionReconciliation(
    int MatchedPositions,
    int DivergenceSegmentIndex,
    int DivergencePositionInSegment,
    SessionReuseReason ReuseReason)
{
    public bool IsExactInputMatch => ReuseReason == SessionReuseReason.None;
}

/// <summary>
/// Caller-visible continuation decision derived from the authoritative execution cursor. This is
/// diagnostic only in the append-only hot lane: callers must not reinterpret a replay-required
/// result as permission to mutate retained state.
/// </summary>
public sealed record SessionContinuationDiagnostic(
    ContinuationGrade Grade,
    SessionReuseReason ReuseReason,
    int MatchedPositions,
    int DivergenceSegmentIndex,
    int DivergencePositionInSegment,
    int CurrentAcceptedPositions,
    int TargetAcceptedPositions)
{
    public bool CanAppendWithoutReplay => Grade == ContinuationGrade.ExactLossless;
}

/// <summary>Computes execution-log common prefixes without retokenising transcript text.</summary>
public static class ExecutionReconciler
{
    public static ExecutionReconciliation Compare(
        ImmutableArray<ExecutionSegment> current,
        ImmutableArray<ExecutionSegment> target)
    {
        int matched = 0;
        int leftIndex = 0, rightIndex = 0, leftPosition = 0, rightPosition = 0;
        while (leftIndex < current.Length && rightIndex < target.Length)
        {
            var left = current[leftIndex];
            var right = target[rightIndex];
            if (left is TokenSegment leftTokens && right is TokenSegment rightTokens)
            {
                while (leftPosition < leftTokens.TokenIds.Length && rightPosition < rightTokens.TokenIds.Length)
                {
                    if (leftTokens.TokenIds[leftPosition] != rightTokens.TokenIds[rightPosition])
                        return new ExecutionReconciliation(matched, rightIndex, rightPosition, SessionReuseReason.PrefixDivergence);
                    matched++;
                    leftPosition++;
                    rightPosition++;
                }
                if (leftPosition == leftTokens.TokenIds.Length) { leftIndex++; leftPosition = 0; }
                if (rightPosition == rightTokens.TokenIds.Length) { rightIndex++; rightPosition = 0; }
                continue;
            }

            if (left.GetType() != right.GetType() || !Equals(left, right))
                return new ExecutionReconciliation(matched, rightIndex, rightPosition, SessionReuseReason.PrefixDivergence);
            matched += left.PositionCount;
            leftIndex++;
            rightIndex++;
        }

        if (leftIndex == current.Length && rightIndex == target.Length)
            return new ExecutionReconciliation(matched, rightIndex, rightPosition, SessionReuseReason.None);
        return new ExecutionReconciliation(matched, rightIndex, rightPosition, SessionReuseReason.PrefixDivergence);
    }

    /// <summary>Classifies whether a target history can append to the current hot state exactly.</summary>
    public static SessionContinuationDiagnostic Diagnose(
        SessionCursor current,
        ImmutableArray<ExecutionSegment> target)
    {
        ArgumentNullException.ThrowIfNull(current);
        var reconciliation = Compare(current.ExecutionLog, target);
        int targetPositions = target.Sum(x => x.PositionCount);
        bool currentIsExactPrefix = reconciliation.MatchedPositions == current.AcceptedPositionCount
            && targetPositions >= current.AcceptedPositionCount;

        if (currentIsExactPrefix && current.Coverage == StateCoverage.Full)
        {
            return new SessionContinuationDiagnostic(
                ContinuationGrade.ExactLossless,
                SessionReuseReason.None,
                reconciliation.MatchedPositions,
                reconciliation.DivergenceSegmentIndex,
                reconciliation.DivergencePositionInSegment,
                current.AcceptedPositionCount,
                targetPositions);
        }

        return new SessionContinuationDiagnostic(
            current.Coverage == StateCoverage.Full
                ? ContinuationGrade.ReplayedFromExecutionLog
                : ContinuationGrade.PartialWindow,
            current.Coverage == StateCoverage.Full
                ? reconciliation.ReuseReason
                : SessionReuseReason.CoverageInsufficient,
            reconciliation.MatchedPositions,
            reconciliation.DivergenceSegmentIndex,
            reconciliation.DivergencePositionInSegment,
            current.AcceptedPositionCount,
            targetPositions);
    }
}

internal static class ExecutionSegmentCodec
{
    public static byte[] Canonicalize(IEnumerable<ExecutionSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write("OpenTail.Stingray.ExecutionLog.v1");
        foreach (var segment in segments)
        {
            ArgumentNullException.ThrowIfNull(segment);
            segment.WriteCanonical(writer);
        }
        writer.Flush();
        return stream.ToArray();
    }
}
