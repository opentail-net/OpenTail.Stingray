namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

/// <summary>
/// Read-mostly view of a model's tensors and metadata used for architecture detection,
/// admission validation, and shape-dependent dispatch.
/// </summary>
public sealed class ArchitectureProbe
{
    public string? Path { get; init; }
    public required string? Architecture { get; init; }
    public required IModelTensorSource TensorSource { get; init; }
    public required ModelHyperparams Hyperparams { get; init; }
    public bool IsGguf { get; init; }
    public GgufModel? Gguf { get; init; }

    public bool HasTensor(string name) => TensorSource.FindTensor(name) is not null;
    public GgufTensorInfo? FindTensor(string name) => TensorSource.FindTensor(name);
    public string? GetMetadataString(string key) =>
        TensorSource.Metadata.TryGetValue(key, out var v) ? v as string ?? v.ToString() : null;
    public long? GetMetadataInt64(string key) =>
        TensorSource.Metadata.TryGetValue(key, out var v) ? Convert.ToInt64(v) : null;
    public int? GetMetadataInt32(string key) =>
        TensorSource.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : null;
    public bool? GetMetadataBool(string key) =>
        TensorSource.Metadata.TryGetValue(key, out var v) ? Convert.ToBoolean(v) : null;
}
