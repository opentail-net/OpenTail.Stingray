namespace OpenTail.Stingray.Sessions;

/// <summary>Opaque identifier for one tenant-scoped logical inference session.</summary>
public readonly record struct SessionId(Guid Value)
{
    public static SessionId New() => new(Guid.NewGuid());
    public override string ToString() => Value.ToString("N");
}

/// <summary>Multi-model routing address for a session by tenant, role, thread and model fingerprint.</summary>
public readonly record struct SessionAddress(string TenantId, string Role, string ThreadId, string ModelFingerprint)
{
    /// <summary>
    /// Derives the session id deterministically from the address, so the same
    /// (tenant, role, thread, model) always resolves to the same session — the property the whole
    /// hot-routing story rests on.
    ///
    /// <para><b>Length-prefixed canonical form</b> (<c>len:value</c> per field). Plain
    /// concatenation would let distinct addresses collide: <c>("a:b", "c")</c> and
    /// <c>("a", "b:c")</c> produce identical text. Prefixing each field with its length makes the
    /// encoding unambiguous whatever the field contents.</para>
    ///
    /// <para><b>Null is encoded distinctly from empty.</b> A length of <c>-1</c> marks null, so
    /// <c>default(SessionAddress)</c> does not collide with an address of four empty strings.</para>
    ///
    /// <para><b>SHA-256, truncated to 16 bytes</b> — not MD5. There is no security claim here; the
    /// reason is that MD5 trips analyzers such as CA5351, and with
    /// <c>TreatWarningsAsErrors</c> enabled repo-wide that turns a future analyzer rollout into a
    /// build break. The static <c>HashData</c> form also avoids allocating a hasher per call, which
    /// matters because routing resolves an address on every turn.</para>
    /// </summary>
    public SessionId ToSessionId()
    {
        static int Len(string? v) => v is null ? -1 : v.Length;
        var canonical =
            $"{Len(TenantId)}:{TenantId}:{Len(Role)}:{Role}:"
            + $"{Len(ThreadId)}:{ThreadId}:{Len(ModelFingerprint)}:{ModelFingerprint}";
        Span<byte> hash = stackalloc byte[32];
        System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(canonical), hash);
        return new SessionId(new Guid(hash[..16]));
    }
}

/// <summary>Monotonically increasing committed session revision.</summary>
public readonly record struct SessionRevision(long Value)
{
    public static readonly SessionRevision Initial = new(0);
    public SessionRevision Next() => checked(new SessionRevision(Value + 1));
    public override string ToString() => Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
}

/// <summary>Tenant-scoped idempotency identifier supplied by a session caller.</summary>
public readonly record struct SessionOperationId(Guid Value)
{
    public static SessionOperationId New() => new(Guid.NewGuid());
    public override string ToString() => Value.ToString("N");
}

/// <summary>Monotonic single-writer fence for a session generation worker.</summary>
public readonly record struct SessionLease(SessionId SessionId, long FencingEpoch);

/// <summary>Operation lifecycle used by the hot in-memory ledger.</summary>
public enum SessionOperationState
{
    Accepted,
    Prefilling,
    Generating,
    CommitPrepared,
    Completed,
    Cancelled,
    Failed,
}

/// <summary>A stable caller digest used to detect unsafe reuse of an operation identifier.</summary>
public readonly record struct SessionRequestDigest(string Value)
{
    public static SessionRequestDigest FromCanonicalValue(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        return new SessionRequestDigest(value);
    }
}

/// <summary>Current authoritative operation state and, once complete, its committed revision.</summary>
public sealed record SessionOperationSnapshot(
    SessionId SessionId,
    SessionOperationId OperationId,
    SessionRequestDigest RequestDigest,
    SessionOperationState State,
    long FencingEpoch,
    SessionRevision? CommittedRevision,
    DateTimeOffset CreatedAt,
    DateTimeOffset? CompletedAt,
    string? FailureReason,
    IReadOnlyList<OpenTail.Stingray.Engine.GenerateChunk>? ResultChunks = null);

/// <summary>Immutable session state visible to callers of the in-memory transaction core.</summary>
public sealed record SessionSnapshot(
    SessionId SessionId,
    SessionRevision CommittedRevision,
    SessionRevision? DurableRevision,
    long CurrentFencingEpoch,
    IReadOnlyCollection<SessionOperationSnapshot> Operations);

public sealed class SessionNotFoundException(SessionId sessionId)
    : InvalidOperationException($"Session '{sessionId}' does not exist.");

/// <summary>
/// A turn was submitted against a revision that is no longer current.
/// <para>Both revisions are exposed, matching <see cref="SessionResourceBudgetExceededException"/>:
/// a caller that loses the optimistic-concurrency race needs the ACTUAL revision to rebase and
/// retry, and parsing it back out of the message is not an API.</para>
/// </summary>
public sealed class SessionRevisionConflictException(SessionRevision expected, SessionRevision actual)
    : InvalidOperationException($"Expected session revision {expected}, but the committed revision is {actual}.")
{
    /// <summary>The revision the caller believed was current.</summary>
    public SessionRevision ExpectedRevision { get; } = expected;

    /// <summary>The revision actually committed; the value to rebase onto.</summary>
    public SessionRevision ActualRevision { get; } = actual;
}

public sealed class SessionFencedException(long requestedEpoch, long currentEpoch)
    : InvalidOperationException($"Lease epoch {requestedEpoch} is fenced by current epoch {currentEpoch}.");

public sealed class SessionOperationConflictException(SessionOperationId operationId)
    : InvalidOperationException($"Operation '{operationId}' was reused with a different request digest.");

public sealed class SessionOperationStateException(SessionOperationState expected, SessionOperationState actual)
    : InvalidOperationException($"Expected operation state {expected}, but its state is {actual}.");
