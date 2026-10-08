namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

/// <summary>A tensor-less <see cref="IModelTensorSource"/> over a metadata dictionary (synthetic headers, tests, tooling).</summary>
public sealed unsafe class MetadataOnlyTensorSource(IReadOnlyDictionary<string, object> metadata) : IModelTensorSource
{
    public IReadOnlyList<GgufTensorInfo> Tensors => [];
    public IReadOnlyDictionary<string, object> Metadata { get; } = metadata;
    public GgufTensorInfo? FindTensor(string name) => null;
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => [];
    public byte* GetTensorDataPtr(GgufTensorInfo tensor) => null;
    public void Dispose() { }
}
