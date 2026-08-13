using System;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Immutable incremental session delta capturing committed state changes (appended tokens and metadata updates)
/// committed since a base <see cref="ResponseContinuationToken"/>.
/// </summary>
public sealed record SessionDelta
{
    public required SessionId SessionId { get; init; }
    public required ResponseContinuationToken BaseToken { get; init; }
    public required ResponseContinuationToken ResultToken { get; init; }
    public IReadOnlyList<int> AppendedTokens { get; init; } = Array.Empty<int>();
    public IReadOnlyDictionary<string, string?> MetadataChanges { get; init; } = ImmutableDictionary<string, string?>.Empty;
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
}
