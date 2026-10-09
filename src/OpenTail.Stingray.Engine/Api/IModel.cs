#nullable enable

using OpenTail.Stingray.Core;

namespace OpenTail.Stingray;

/// <summary>
/// Represents loaded model weights and architecture metadata in memory.
/// Equivalent to LLamaSharp's <c>LLamaWeights</c>, named neutrally for Stingray's multi-architecture runtime.
/// Can be reused across multiple execution contexts.
/// </summary>
public interface IModel : IDisposable
{
    /// <summary>Path to the source model file on disk.</summary>
    string ModelPath { get; }

    /// <summary>Model architecture identifier (e.g. "llama", "qwen2", "gemma4", "deepseek2").</summary>
    string Architecture { get; }

    /// <summary>Maximum native context length trained into the model.</summary>
    int ContextLength { get; }

    /// <summary>Hidden embedding dimension size.</summary>
    int EmbeddingLength { get; }

    /// <summary>
    /// Allocates an independent execution context and KV cache over this model's weights.
    /// </summary>
    /// <param name="contextParams">Optional context configuration parameters.</param>
    /// <returns>An allocated <see cref="IModelContext"/> ready for execution.</returns>
    IModelContext CreateContext(IContextParams? contextParams = null);
}

/// <summary>
/// Represents an allocated execution context over a loaded model, owning KV cache state, memory arenas, and tokenizer view.
/// Equivalent to LLamaSharp's <c>LLamaContext</c>.
/// </summary>
public interface IModelContext : IDisposable
{
    /// <summary>Reference to the underlying model weights.</summary>
    IModel Model { get; }

    /// <summary>Model tokenizer for encoding text prompts and decoding output tokens.</summary>
    ITokenizer Tokenizer { get; }

    /// <summary>Effective maximum context length in tokens allocated for this context.</summary>
    int ContextSize { get; }

    /// <summary>Underlying inference engine powering this context.</summary>
    OpenTail.Stingray.Engine.IInferenceEngine Engine { get; }

    /// <summary>
    /// Clears active KV cache tokens and resets sequence state to clean initial status.
    /// </summary>
    void Reset();
}
