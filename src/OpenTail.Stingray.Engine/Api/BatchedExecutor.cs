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
    private readonly IModelContext _context;

    /// <inheritdoc/>
    public IModelContext Context => _context;

    /// <summary>Concrete model context driving this executor, or null if using a custom <see cref="IModelContext"/>.</summary>
    public ModelContext? ModelContext => _context as ModelContext;

    /// <summary>
    /// Creates a batched executor over the provided <see cref="ModelContext"/>.
    /// </summary>
    public BatchedExecutor(ModelContext context)
        : this((IModelContext)context)
    {
    }

    /// <summary>
    /// Creates a batched executor over the provided <see cref="IModelContext"/>.
    /// </summary>
    public BatchedExecutor(IModelContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        _context = context;
        if (_context is ModelContext mc && !mc.HasExplicitEngine)
        {
            mc.EnsureContinuousBatchingEngine();
        }
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

        var sp = inferenceParams.ToSamplingParams();

        await foreach (var chunk in _context.Engine.GenerateChunksAsync(prompt, sp, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            yield return chunk;
        }
    }
}
