using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio;

/// <summary>
/// Tensor source for <c>audio.cpp</c>-packed GGUF checkpoints
/// (<c>general.architecture=audiocpp</c>, <c>audiocpp.tensor_name_format=native</c>):
/// maps real tensor names from <c>audiocpp.tensor_names</c> metadata to underlying tensors,
/// avoiding eager F32 materialization when quantized representations (e.g. Q8_0) are used directly.
/// </summary>
public class AudioCppPackedTensorSource
{
    private readonly GgufModel _model;
    private readonly Dictionary<string, GgufTensorInfo> _byRealName;

    public AudioCppPackedTensorSource(GgufModel model)
    {
        _model = model;
        if (!model.Metadata.TryGetValue("audiocpp.tensor_names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidDataException("Expected audio.cpp-packed GGUF with an 'audiocpp.tensor_names' metadata array.");
        if (names.Length != model.Tensors.Count)
            throw new InvalidDataException($"audiocpp.tensor_names length ({names.Length}) does not match tensor count ({model.Tensors.Count}).");

        _byRealName = new Dictionary<string, GgufTensorInfo>(names.Length, StringComparer.Ordinal);
        for (int i = 0; i < names.Length; i++)
            _byRealName[(string)names[i]] = model.Tensors[i];
    }

    public float[] GetTensor(string realName)
    {
        if (!_byRealName.TryGetValue(realName, out var info))
            throw new InvalidDataException($"audio.cpp packed GGUF missing required tensor '{realName}'.");
        var bytes = _model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Cpu.Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public bool HasTensor(string realName) => _byRealName.ContainsKey(realName);

    public GgufTensorInfo GetRawInfo(string realName)
    {
        if (!_byRealName.TryGetValue(realName, out var info))
            throw new InvalidDataException($"audio.cpp packed GGUF missing required tensor '{realName}'.");
        return info;
    }

    public unsafe byte* GetRawDataPtr(string realName)
    {
        if (!_byRealName.TryGetValue(realName, out var info))
            throw new InvalidDataException($"audio.cpp packed GGUF missing required tensor '{realName}'.");
        return _model.GetTensorDataPtr(info);
    }

    /// <summary>Copies the raw, un-dequantized on-disk tensor bytes into an owned byte array,
    /// validating that its on-disk DType matches <paramref name="expectedDType"/> if specified.</summary>
    public byte[] GetRawBytes(string realName, DType? expectedDType = null)
    {
        if (!_byRealName.TryGetValue(realName, out var info))
            throw new InvalidDataException($"audio.cpp packed GGUF missing required tensor '{realName}'.");
        if (expectedDType.HasValue && info.DType != expectedDType.Value)
            throw new InvalidDataException($"audio.cpp packed GGUF tensor '{realName}' has dtype {info.DType}, expected {expectedDType.Value}.");
        return _model.GetTensorData(info).ToArray();
    }
}
