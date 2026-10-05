#nullable enable

using System.Runtime.CompilerServices;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Executors;

/// <summary>
/// High-throughput concurrent multi-sequence executor leveraging continuous batching.
/// Equivalent to LLamaSharp's <c>BatchedExecutor</c>.
/// Maps directly to Stingray's <see cref="ContinuousBatchingEngine"/>.
/// </summary>
public sealed class BatchedExecutor : IExecutor
{
    private readonly ModelContext _context;

    /// <inheritdoc/>
    public IModelContext Context => _context;

    /// <summary>Concrete model context driving this executor.</summary>
    public ModelContext ModelContext => _context;

    /// <summary>
    /// Creates a batched executor over the provided <see cref="ModelContext"/>.
    /// </summary>
    public BatchedExecutor(ModelContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        _context = context;
    }

    /// <summary>
    /// Creates a batched executor over the provided <see cref="IModelContext"/>.
    /// </summary>
    public BatchedExecutor(IModelContext context)
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

        await foreach (var chunk in _context.Engine.GenerateChunksAsync(prompt, sp, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            yield return chunk;
        }
    }
}
