#nullable enable

using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Executors;

/// <summary>
/// Common execution contract for driving text generation pipelines over a model context.
/// Inspired by LLamaSharp's <c>ILLamaExecutor</c>, with dual-streaming support for simple string output
/// and typed chunk output (<see cref="GenerateChunk"/>) preserving thinking and stop reasons.
/// </summary>
public interface IExecutor
{
    /// <summary>The execution context driving this executor.</summary>
    IModelContext Context { get; }

    /// <summary>
    /// Executes autoregressive token generation for the given prompt, yielding generated text chunks.
    /// Filters internal reasoning/thinking tags to provide a clean user-facing text stream.
    /// </summary>
    /// <param name="prompt">Input prompt text.</param>
    /// <param name="inferenceParams">Optional per-inference generation parameters.</param>
    /// <param name="cancellationToken">Cancellation token to halt generation.</param>
    /// <returns>Asynchronous stream of generated text pieces.</returns>
    IAsyncEnumerable<string> InferAsync(
        string prompt,
        IInferenceParams? inferenceParams = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes autoregressive token generation for the given prompt, yielding typed <see cref="GenerateChunk"/> objects.
    /// Preserves thinking tokens (<see cref="GenerateChunkKind.Thinking"/>), final text, and usage metrics.
    /// </summary>
    /// <param name="prompt">Input prompt text.</param>
    /// <param name="inferenceParams">Optional per-inference generation parameters.</param>
    /// <param name="cancellationToken">Cancellation token to halt generation.</param>
    /// <returns>Asynchronous stream of typed generation chunks.</returns>
    IAsyncEnumerable<GenerateChunk> InferChunksAsync(
        string prompt,
        IInferenceParams? inferenceParams = null,
        CancellationToken cancellationToken = default);
}
