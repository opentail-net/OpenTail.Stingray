namespace OpenTail.Stingray.Core;

/// <summary>
/// Supplies a model's weight tensors to the engine, independently of the file format they came from.
/// </summary>
/// <remarks>
/// <para><b>Why the seam is here and not inside the forward pass.</b> The plan originally called for
/// extracting the dense path of <c>ForwardPass</c> behind a tensor abstraction. That is a logic
/// refactor of the engine's hottest file, and this project treats GGUF regressions as release
/// blockers — it would need performance evidence to be safe. Measured instead: the engine touches
/// exactly five members of <c>GgufModel</c> (<see cref="FindTensor"/>, <see cref="GetTensorData"/>,
/// <see cref="GetTensorDataPtr"/>, <see cref="Tensors"/>, <see cref="Metadata"/>). Lifting those into
/// an interface is a type swap that cannot alter GGUF behaviour, and it lets another format feed the
/// <b>existing, unmodified</b> transformer loop.</para>
///
/// <para><b>On <see cref="GgufTensorInfo"/> in the signatures.</b> It is the descriptor this codebase
/// already speaks — name, dimensions, dtype, offset — and it is not inherently GGUF-specific apart
/// from its name. A non-GGUF source synthesises descriptors for its own tensors and serves the bytes
/// from its own storage. Introducing a parallel descriptor would mean translating at every call site
/// for no gain.</para>
///
/// <para><b>Contract for implementers.</b> <see cref="GetTensorDataPtr"/> must return a pointer that
/// stays valid for the lifetime of the source, because the engine holds it across calls and reads it
/// on other threads. A source that cannot promise that must not implement this interface — returning
/// a pointer into a buffer it may reallocate would corrupt inference in a way no test would localise.</para>
/// </remarks>
public unsafe interface IModelTensorSource
{
    /// <summary>All tensors this source exposes.</summary>
    IReadOnlyList<GgufTensorInfo> Tensors { get; }

    /// <summary>Model-level metadata (hyperparameters, tokenizer, architecture).</summary>
    IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>Finds a tensor by canonical name, or null when the source has no such tensor.</summary>
    GgufTensorInfo? FindTensor(string name);

    /// <summary>Returns the tensor's bytes in its stored dtype.</summary>
    ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor);

    /// <summary>
    /// Returns a stable pointer to the tensor's bytes. See the lifetime contract in the type remarks.
    /// </summary>
    byte* GetTensorDataPtr(GgufTensorInfo tensor);
}
