
namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Same technique as <see cref="FunAsrNanoLlmTensorSource"/> (presents `model.language_model.*` as
/// a standard `qwen3` model to the existing engine `ForwardPass`), but sourced from the real
/// audio.cpp-packed GGUF (`models/paraformer-q8.gguf`, via <see cref="Rvc.RvcPackedTensorSource"/>)
/// instead of a separately-downloaded raw `.safetensors` file -- this checkpoint already contains
/// every real tensor needed (confirmed: `general.architecture=audiocpp`,
/// `audiocpp.tensor_name_format=native`, real tensor names include both
/// `model.audio_tower.*` and `model.language_model.*`), so no second download is needed.
/// </summary>
public sealed unsafe class FunAsrNanoLlmGgufTensorSource : IModelTensorSource, IDisposable, QwenASR.IQwenAsrAudioConditionableSource
{
    private readonly Rvc.RvcPackedTensorSource _source;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonical = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private bool _disposed;

    public FunAsrNanoLlmGgufTensorSource(GgufModel model, int numLayers, int hiddenDim, int numHeads, int numKvHeads, int headDim, int ffDim, int vocabSize, float ropeTheta, float rmsNormEps)
    {
        _source = new Rvc.RvcPackedTensorSource(model);

        MapIfPresent(model, "model.language_model.embed_tokens.weight", "token_embd.weight");
        MapIfPresent(model, "model.language_model.norm.weight", "output_norm.weight");
        // No lm_head mapping -- tied embeddings (text_config.tie_word_embeddings=true).

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.language_model.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent(model, p + "input_layernorm.weight", b + "attn_norm.weight");
            MapIfPresent(model, p + "self_attn.q_proj.weight", b + "attn_q.weight");
            MapIfPresent(model, p + "self_attn.k_proj.weight", b + "attn_k.weight");
            MapIfPresent(model, p + "self_attn.v_proj.weight", b + "attn_v.weight");
            MapIfPresent(model, p + "self_attn.o_proj.weight", b + "attn_output.weight");
            MapIfPresent(model, p + "self_attn.q_norm.weight", b + "attn_q_norm.weight");
            MapIfPresent(model, p + "self_attn.k_norm.weight", b + "attn_k_norm.weight");
            MapIfPresent(model, p + "post_attention_layernorm.weight", b + "ffn_norm.weight");
            MapIfPresent(model, p + "mlp.gate_proj.weight", b + "ffn_gate.weight");
            MapIfPresent(model, p + "mlp.up_proj.weight", b + "ffn_up.weight");
            MapIfPresent(model, p + "mlp.down_proj.weight", b + "ffn_down.weight");
        }

        _tensors = [.. _byName.Values];

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen3",
            ["qwen3.embedding_length"] = hiddenDim,
            ["qwen3.block_count"] = numLayers,
            ["qwen3.attention.head_count"] = numHeads,
            ["qwen3.attention.head_count_kv"] = numKvHeads,
            ["qwen3.attention.key_length"] = headDim,
            ["qwen3.attention.value_length"] = headDim,
            ["qwen3.feed_forward_length"] = ffDim,
            ["qwen3.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["qwen3.rope.freq_base"] = ropeTheta,
            ["qwen3.vocab_size"] = vocabSize,
            ["qwen3.context_length"] = 40960,
        };
    }

    private void MapIfPresent(GgufModel model, string realName, string canonicalName)
    {
        if (!_source.HasTensor(realName)) return;
        var info = FindByRealName(model, realName) ?? throw new InvalidDataException($"'{realName}' reported present but not found in tensor list.");
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, info.Dimensions.Length, info.Dimensions, DType.Float32, DataOffset: 0);
        _sourceNameByCanonical[canonicalName] = realName;
    }

    private static GgufTensorInfo? FindByRealName(GgufModel model, string realName)
    {
        if (!model.Metadata.TryGetValue("audiocpp.tensor_names", out var namesObj) || namesObj is not object[] names) return null;
        for (int i = 0; i < names.Length; i++)
            if ((string)names[i] == realName) return model.Tensors[i];
        return null;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) =>
        _byName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        byte* pointer = GetTensorDataPtr(tensor);
        return new ReadOnlySpan<byte>(pointer, checked((int)(tensor.ElementCount * sizeof(float))));
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint syntheticPtr))
            return (byte*)syntheticPtr;
        if (_resolvedPointers.TryGetValue(tensor.Name, out nint cached))
            return (byte*)cached;

        string sourceName = _sourceNameByCanonical[tensor.Name];
        var data = _source.GetTensor(sourceName);
        float* buffer = (float*)NativeMemory.Alloc((nuint)(data.Length * sizeof(float)));
        fixed (float* src = data)
            Buffer.MemoryCopy(src, buffer, data.Length * sizeof(float), data.Length * sizeof(float));
        _ownedPointers.Add((nint)buffer);
        _resolvedPointers[tensor.Name] = (nint)buffer;
        return (byte*)buffer;
    }

    public int AudioTokenIdOffset { get; private set; } = -1;

    public void EnableAudioConditioning(ReadOnlySpan<float> audioEmbeddings, int numAudioTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (numAudioTokens <= 0) throw new ArgumentOutOfRangeException(nameof(numAudioTokens));

        var textEmbedInfo = _byName["token_embd.weight"];
        int hiddenDim = checked((int)textEmbedInfo.Dimensions[0]);
        int textVocab = checked((int)textEmbedInfo.Dimensions[1]);
        if (audioEmbeddings.Length != (long)numAudioTokens * hiddenDim)
            throw new ArgumentException($"audioEmbeddings length {audioEmbeddings.Length} != numAudioTokens*hiddenDim ({numAudioTokens}*{hiddenDim}).", nameof(audioEmbeddings));

        byte* textEmbedPtr = GetTensorDataPtr(textEmbedInfo);

        int combinedVocab = textVocab + numAudioTokens;
        long combinedElementCount = (long)combinedVocab * hiddenDim;
        float* combined = (float*)NativeMemory.Alloc((nuint)(combinedElementCount * sizeof(float)));
        Buffer.MemoryCopy(textEmbedPtr, combined, combinedElementCount * sizeof(float), (long)textVocab * hiddenDim * sizeof(float));
        fixed (float* audioPtr = audioEmbeddings)
        {
            long audioElementCount = (long)numAudioTokens * hiddenDim;
            Buffer.MemoryCopy(audioPtr, combined + (long)textVocab * hiddenDim,
                audioElementCount * sizeof(float), audioElementCount * sizeof(float));
        }

        _ownedPointers.Add((nint)combined);
        _syntheticBuffers["token_embd.weight"] = (nint)combined;
        _byName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [hiddenDim, combinedVocab], DType.Float32, DataOffset: 0);
        _tensors.Clear();
        _tensors.AddRange(_byName.Values);

        AudioTokenIdOffset = textVocab;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _ownedPointers) NativeMemory.Free((void*)p);
        _ownedPointers.Clear();
    }
}
