using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Immutable structured final result representing the committed outcome of a generation operation.
/// </summary>
public sealed record GenerationResult(
    FinishReason FinishReason,
    int GeneratedTokenCount,
    IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall> ToolCalls,
    ResponseContinuationToken? ContinuationToken,
    SessionMetricsSnapshot Metrics);
