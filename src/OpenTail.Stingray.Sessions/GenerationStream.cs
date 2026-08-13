using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Result-bearing streaming handle implementing <see cref="IAsyncEnumerable{GenerateChunk}"/>.
/// Allows streaming chunks via <c>await foreach</c> while accumulating a structured <see cref="GenerationResult"/>
/// accessible via <see cref="Result"/> after enumeration finishes or completes.
/// </summary>
public sealed class GenerationStream : IAsyncEnumerable<GenerateChunk>
{
    private readonly IInferenceSession _session;
    private readonly SamplingParams _sampling;
    private readonly CancellationToken _cancellationToken;
    private GenerationResult? _result;

    public GenerationStream(IInferenceSession session, SamplingParams sampling, CancellationToken cancellationToken = default)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _sampling = sampling ?? throw new ArgumentNullException(nameof(sampling));
        _cancellationToken = cancellationToken;
    }

    /// <summary>
    /// Gets the final committed <see cref="GenerationResult"/> of this generation operation.
    /// Throws <see cref="InvalidOperationException"/> if accessed before the stream has been enumerated or completed.
    /// </summary>
    public GenerationResult Result =>
        _result ?? throw new InvalidOperationException("GenerationResult is only available after the stream has been enumerated or completed.");

    public IAsyncEnumerator<GenerateChunk> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        CancellationTokenSource? linkedCts = null;
        CancellationToken token;

        if (!_cancellationToken.CanBeCanceled)
        {
            token = cancellationToken;
        }
        else if (!cancellationToken.CanBeCanceled || _cancellationToken == cancellationToken)
        {
            token = _cancellationToken;
        }
        else
        {
            linkedCts = CancellationTokenSource.CreateLinkedTokenSource(_cancellationToken, cancellationToken);
            token = linkedCts.Token;
        }

        return new Enumerator(_session, _sampling, token, linkedCts, this);
    }

    private sealed class Enumerator : IAsyncEnumerator<GenerateChunk>
    {
        private readonly IInferenceSession _session;
        private readonly IAsyncEnumerator<GenerateChunk> _inner;
        private readonly GenerationStream _owner;
        private readonly CancellationTokenSource? _linkedCts;
        private readonly List<OpenTail.Stingray.Core.Tools.ToolCall> _toolCalls = new();
        private int _generatedTokenCount;
        private FinishReason? _finishReason;
        private bool _completed;

        public Enumerator(
            IInferenceSession session,
            SamplingParams sampling,
            CancellationToken token,
            CancellationTokenSource? linkedCts,
            GenerationStream owner)
        {
            _session = session;
            _linkedCts = linkedCts;
            _owner = owner;
            _inner = session.GenerateAsync(sampling, token).GetAsyncEnumerator(token);
        }

        public GenerateChunk Current => _inner.Current;

        public async ValueTask<bool> MoveNextAsync()
        {
            try
            {
                bool hasNext = await _inner.MoveNextAsync().ConfigureAwait(false);
                if (hasNext)
                {
                    var chunk = _inner.Current;
                    if (!string.IsNullOrEmpty(chunk.Text))
                    {
                        _generatedTokenCount++;
                    }

                    if (chunk.Kind == GenerateChunkKind.ToolCall)
                    {
                        _finishReason = FinishReason.ToolCall;
                        if (chunk.ToolCalls is not null)
                        {
                            _toolCalls.AddRange(chunk.ToolCalls);
                        }
                    }
                    else if (chunk.Kind == GenerateChunkKind.Stop)
                    {
                        if (chunk.TruncatedByResourceBudget || _session.IsContextLimitReached)
                        {
                            _finishReason = FinishReason.ContextLimit;
                        }
                        else if (chunk.TruncatedByMaxTokens)
                        {
                            _finishReason = FinishReason.MaxTokens;
                        }
                        else
                        {
                            _finishReason ??= FinishReason.Completed;
                        }
                    }

                    if (chunk.ToolCalls is not null && chunk.Kind != GenerateChunkKind.ToolCall)
                    {
                        _toolCalls.AddRange(chunk.ToolCalls);
                    }

                    return true;
                }
                else
                {
                    FinalizeResult(cancelled: false);
                    return false;
                }
            }
            catch (OperationCanceledException)
            {
                FinalizeResult(cancelled: true);
                throw;
            }
            catch
            {
                FinalizeResult(cancelled: false);
                throw;
            }
        }

        public async ValueTask DisposeAsync()
        {
            try
            {
                await _inner.DisposeAsync().ConfigureAwait(false);
            }
            finally
            {
                FinalizeResult(cancelled: false);
                _linkedCts?.Dispose();
            }
        }

        private void FinalizeResult(bool cancelled)
        {
            if (_completed) return;
            _completed = true;

            FinishReason reason;
            if (cancelled)
            {
                reason = FinishReason.Cancelled;
            }
            else if (_finishReason.HasValue)
            {
                reason = _finishReason.Value;
            }
            else if (_session.IsContextLimitReached)
            {
                reason = FinishReason.ContextLimit;
            }
            else
            {
                reason = FinishReason.Completed;
            }

            ResponseContinuationToken? continuation = null;
            try
            {
                continuation = _session.GetContinuationToken();
            }
            catch
            {
                // Disposed or invalid state
            }

            var metricsSnapshot = SessionMetricsSnapshot.Capture(_session.Metrics);

            _owner._result = new GenerationResult(
                FinishReason: reason,
                GeneratedTokenCount: _generatedTokenCount,
                ToolCalls: _toolCalls.AsReadOnly(),
                ContinuationToken: continuation,
                Metrics: metricsSnapshot);
        }
    }
}
