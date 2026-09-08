
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real name resolution for `rvc-f16.gguf` and similar `audio.cpp`-packed checkpoints
/// (`general.architecture=audiocpp`, `audiocpp.tensor_name_format=native`): the GGUF's own
/// tensors carry opaque `_audiocpp.NNNN` names, not the real, meaningful ones -- the real names
/// only exist in the `audiocpp.tensor_names` metadata array, paired by index with
/// `GgufModel.Tensors` (confirmed directly against the real checkpoint, not assumed -- see
/// `RvcTensorNameDumpDebugTest`). This class builds that name -&gt; tensor lookup once, so
/// `RvcHubertWeights`/`RvcRmvpeWeights`/etc. can read tensors by their real, familiar names
/// exactly like any other GGUF loader in this codebase.
/// </summary>
public sealed class RvcPackedTensorSource
{
    private readonly GgufModel _model;
    private readonly Dictionary<string, GgufTensorInfo> _byRealName;

    public RvcPackedTensorSource(GgufModel model)
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
            throw new InvalidDataException($"RVC packed GGUF missing required tensor '{realName}'.");
        var bytes = _model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Cpu.Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public bool HasTensor(string realName) => _byRealName.ContainsKey(realName);

    /// <summary>Real tensor info (element count, real on-disk `DType` -- Q8_0/Q4_K/etc., NOT
    /// forced to Float32) for a packed tensor, without dequantizing. Real, non-obvious purpose:
    /// large LLM weight matrices don't need to be FP32-dequantized at all -- `ForwardPass`
    /// natively computes on quantized weights the same way every other GGUF-loaded model in this
    /// codebase does, so a bridging `IModelTensorSource` can pass the real DType straight through
    /// (see `VibeVoiceLlmTensorSource`'s real-DType path) instead of eagerly materializing a much
    /// larger FP32 copy of every weight, which is what was causing a real OOM on 7B-class
    /// checkpoints (see `docs/audio-review-progress.md`'s VibeVoice ASR entries).</summary>
    public GgufTensorInfo GetRawInfo(string realName)
    {
        if (!_byRealName.TryGetValue(realName, out var info))
            throw new InvalidDataException($"RVC packed GGUF missing required tensor '{realName}'.");
        return info;
    }

    /// <summary>Zero-copy pointer to the tensor's real on-disk bytes (still in `GetRawInfo`'s
    /// `DType`, NOT dequantized) -- backed by <see cref="GgufModel.GetTensorDataPtr"/>'s own
    /// mmap'd pointer, so this allocates nothing.</summary>
    public unsafe byte* GetRawDataPtr(string realName)
    {
        if (!_byRealName.TryGetValue(realName, out var info))
            throw new InvalidDataException($"RVC packed GGUF missing required tensor '{realName}'.");
        return _model.GetTensorDataPtr(info);
    }
}
