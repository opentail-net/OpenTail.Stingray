using System;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Exception thrown when a <see cref="ResponseContinuationToken"/> specifies an outdated checkpoint generation or invalid token position.
/// </summary>
public sealed class StaleContinuationException : InvalidOperationException
{
    public StaleContinuationException(SessionId sessionId, long tokenPosition, long tokenGeneration, long currentPosition, long currentGeneration)
        : base($"Continuation token is stale or invalid for session '{sessionId}': Token specified (Position: {tokenPosition}, Generation: {tokenGeneration}), but current session state is (Position: {currentPosition}, Generation: {currentGeneration}). Continuation rejected to prevent rewinding or overwriting newer state.")
    {
        SessionId = sessionId;
        TokenPosition = tokenPosition;
        TokenGeneration = tokenGeneration;
        CurrentPosition = currentPosition;
        CurrentGeneration = currentGeneration;
    }

    public SessionId SessionId { get; }
    public long TokenPosition { get; }
    public long TokenGeneration { get; }
    public long CurrentPosition { get; }
    public long CurrentGeneration { get; }
}
