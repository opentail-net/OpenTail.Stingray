using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Represents an individual immutable branch vote outcome in a parallel consensus operation.
/// </summary>
public sealed record BranchVote(
    SessionId BranchId,
    string Text,
    string NormalizedAnswer,
    bool IsWinner);

/// <summary>
/// Represents the overall outcome of a <see cref="InferenceSessionConsensusExtensions.ForkAndVoteAsync"/> consensus operation.
/// </summary>
public sealed record BranchVoteResult(
    IInferenceSession WinningBranch,
    string WinningText,
    IReadOnlyList<BranchVote> Votes);
