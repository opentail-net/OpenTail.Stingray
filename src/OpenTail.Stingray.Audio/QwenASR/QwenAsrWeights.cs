using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Container for Alibaba Qwen3-ASR (AuT audio encoder + Qwen3 LLM text decoder) GGUF weights.
/// </summary>
public sealed class QwenAsrWeights : IDisposable
{
    public GgufModel Model { get; }

    public int AudioLayers { get; } = 18;
    public int AudioDim { get; } = 896;
    public int AudioHeads { get; } = 14;
    public int LlmLayers { get; } = 28;
    public int LlmDim { get; } = 1024;
    public int LlmHeads { get; } = 16;
    public int LlmKvHeads { get; } = 8;
    public int LlmVocabSize { get; } = 151936;

    public QwenAsrWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Qwen3-ASR model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        if (Model.Metadata.TryGetValue("qwen3asr.audio.n_layers", out var anl) && anl is int anli)
            AudioLayers = anli;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.d_model", out var adm) && adm is int admi)
            AudioDim = admi;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.n_heads", out var anh) && anh is int anhi)
            AudioHeads = anhi;

        if (Model.Metadata.TryGetValue("qwen3asr.llm.n_layers", out var lnl) && lnl is int lnli)
            LlmLayers = lnli;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.d_model", out var ldm) && ldm is int ldmi)
            LlmDim = ldmi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.n_heads", out var lnh) && lnh is int lnhi)
            LlmHeads = lnhi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.n_kv_heads", out var lnkv) && lnkv is int lnkvi)
            LlmKvHeads = lnkvi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.vocab_size", out var lvs) && lvs is int lvsi)
            LlmVocabSize = lvsi;
    }

    public void Dispose()
    {
        Model.Dispose();
    }
}
