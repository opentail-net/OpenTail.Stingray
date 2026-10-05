#nullable enable

using System.Runtime.CompilerServices;
using System.Text;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Executors;

/// <summary>
/// Stateful executor maintaining context state across sequential turns.
/// Equivalent to LLamaSharp's <c>InteractiveExecutor</c>.
/// Leverages Stingray's native KV cache prefix reuse (<c>canonicalHistoryPrefix</c>) to avoid re-evaluating prompt history.
/// </summary>
public sealed class InteractiveExecutor : IExecutor
{
    private readonly ModelContext _context;
    private readonly StringBuilder _history = new();
    private string? _lastPrefix;
    private readonly object _lock = new();

    /// <inheritdoc/>
    public IModelContext Context => _context;

    /// <summary>Concrete model context driving this executor.</summary>
    public ModelContext ModelContext => _context;

    /// <summary>
    /// Creates an interactive executor over the provided <see cref="ModelContext"/>.
    /// </summary>
    public InteractiveExecutor(ModelContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        _context = context;
    }

    /// <summary>
    /// Creates an interactive executor over the provided <see cref="IModelContext"/>.
    /// </summary>
    public InteractiveExecutor(IModelContext context)
        : this(context as ModelContext ?? throw new ArgumentException("Context must be a concrete ModelContext instance.", nameof(context)))
    {
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> InferAsync(
        string prompt,
        IInferenceParams? inferenceParams = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var chunk in InferChunksAsync(prompt, inferenceParams, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (chunk.Kind == GenerateChunkKind.Text)
            {
                yield return chunk.Text;
            }
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<GenerateChunk> InferChunksAsync(
        string prompt,
        IInferenceParams? inferenceParams = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);

        var sp = (inferenceParams as InferenceParams)?.ToSamplingParams() ?? new SamplingParams();
        string? prefixToUse;
        lock (_lock)
        {
            prefixToUse = _lastPrefix;
        }

        var outputBuilder = new StringBuilder();

        await foreach (var chunk in _context.Engine.GenerateChunksAsync(prompt, sp, cancellationToken, canonicalHistoryPrefix: prefixToUse).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (chunk.Kind is GenerateChunkKind.Text or GenerateChunkKind.Thinking)
            {
                outputBuilder.Append(chunk.Text);
            }
            yield return chunk;
        }

        lock (_lock)
        {
            _history.Append(prompt);
            _history.Append(outputBuilder.ToString());
            _lastPrefix = prompt;
        }
    }
}
