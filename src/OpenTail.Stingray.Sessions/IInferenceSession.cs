using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Lifecycle state for an <see cref="IInferenceSession"/>.
/// </summary>
public enum SessionState
{
    Created,
    Ready,
    Generating,
    Suspended,
    Cold,
    Faulted,
    Disposed
}

/// <summary>
/// Rich inference checkpoint capturing committed token position, logical KV state, and deterministic sampling RNG state.
/// </summary>
public readonly record struct SessionCheckpoint(
    SessionId SessionId,
    long TokenPosition,
    int RngSeed,
    int RngStep,
    IReadOnlyList<int> CommittedTokens,
    DateTimeOffset CreatedAt);

/// <summary>
/// Serializable session snapshot DTO capturing token history and metadata for session recovery.
/// Note: Physical KV cache pages are reconstructed via re-prefill upon session restore.
/// </summary>
public sealed record InferenceSessionSnapshot
{
    public int Version { get; init; } = 1;
    public SessionId Id { get; init; }
    public string ModelId { get; init; } = "";
    public IReadOnlyList<int> Tokens { get; init; } = Array.Empty<int>();
    public long Position { get; init; }
    public long Generation { get; init; }
    public DateTimeOffset SavedAt { get; init; } = DateTimeOffset.UtcNow;
}

/// <summary>
/// First-class native inference session owning token history, current position,
/// logical KV cache sequence, generation state, and zero-copy page forking.
/// </summary>
public interface IInferenceSession : IAsyncDisposable
{
    SessionId Id { get; }
    SessionId? ParentSessionId { get; }
    SessionState State { get; }
    DateTimeOffset LastActivityUtc { get; }
    long TokenCount { get; }
    IReadOnlyList<int> TokenHistory { get; }
    IKvSequence KvSequence { get; }
    ISessionMetadata Metadata { get; }

    /// <summary>Read-only metrics surface exposing per-session inference and physical KV page usage statistics.</summary>
    ISessionMetrics Metrics { get; }

    /// <summary>Read-only topology surface exposing session lineage, parent-child branch relationships, and aggregated tree metrics.</summary>
    ISessionTree Tree { get; }

    /// <summary>Read-only model capability descriptor exposing loaded model characteristics.</summary>
    OpenTail.Stingray.Core.Capabilities.ModelCapabilities ModelCapabilities { get; }

    /// <summary>Configured maximum sequence context limit in tokens. Returns null if session is unlimited.</summary>
    int? MaxContextTokens { get; }

    /// <summary>Indicates whether the committed token count has reached or exceeded MaxContextTokens.</summary>
    bool IsContextLimitReached { get; }

    /// <summary>Remaining token capacity before MaxContextTokens is reached. Returns int.MaxValue if session is unlimited.</summary>
    int RemainingContextTokens { get; }

    OpenTail.Stingray.Core.Tools.IToolProvider? ToolProvider { get; set; }
    OpenTail.Stingray.Core.Tools.InferenceToolContext? ToolContext { get; set; }
    OpenTail.Stingray.Core.ITokenizer? Tokenizer { get; set; }
    Func<IReadOnlyList<int>, IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall>?>? ToolCallParser { get; set; }

    /// <summary>Attaches a native declarative skill package to the inference session.</summary>
    void AttachSkill(OpenTail.Stingray.Core.ISkill skill);

    /// <summary>Detaches a skill from the session by name.</summary>
    bool DetachSkill(string skillName);

    /// <summary>Read-only list of skills currently attached to the session.</summary>
    IReadOnlyList<OpenTail.Stingray.Core.ISkill> AttachedSkills { get; }

    /// <summary>Fires synchronously when a model token is successfully committed to session state.</summary>
    event Action<int, string>? OnTokenGenerated;

    ValueTask AppendAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken = default);
    ValueTask AppendToolResultAsync(OpenTail.Stingray.Core.Tools.ToolResult result, CancellationToken cancellationToken = default);
    ValueTask AppendToolResultAsync(OpenTail.Stingray.Core.Tools.ToolResult result, ResponseContinuationToken? token, CancellationToken cancellationToken = default);
    IAsyncEnumerable<GenerateChunk> GenerateAsync(SamplingParams sampling, CancellationToken cancellationToken = default);
    GenerationStream GenerateWithResultAsync(SamplingParams sampling, CancellationToken cancellationToken = default);

    ResponseContinuationToken GetContinuationToken();
    void ValidateContinuationToken(ResponseContinuationToken token);
    ValueTask ContinueAsync(ResponseContinuationToken token, CancellationToken cancellationToken = default);

    bool ValidateToolCall(OpenTail.Stingray.Core.Tools.ToolCall call);

    SessionCheckpoint CreateCheckpoint();
    void Rollback(SessionCheckpoint checkpoint);
    SessionDelta CreateDelta(ResponseContinuationToken baseToken);
    ValueTask ApplyDeltaAsync(SessionDelta delta, CancellationToken cancellationToken = default);
    InferenceSessionSnapshot ToSnapshot(string modelId);
    void RestoreFromSnapshot(InferenceSessionSnapshot snapshot);
    ValueTask SuspendAsync(CancellationToken cancellationToken = default);
    ValueTask ResumeAsync(CancellationToken cancellationToken = default);
    ValueTask EvictToColdAsync(ISessionStore store, CancellationToken cancellationToken = default);
    ValueTask EnsureActiveAsync(ISessionStore? store = null, CancellationToken cancellationToken = default);
    IInferenceSession Fork();
    IReadOnlyList<IInferenceSession> ForkMany(int count);
}
