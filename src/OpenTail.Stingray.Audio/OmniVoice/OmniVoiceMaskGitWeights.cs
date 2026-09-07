namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>Real per-layer weights for OmniVoice's own MaskGIT transformer, ported from
/// `generator.cpp`'s `decoder_layer`/`load_layer_weights` (not guessed): a standard Qwen3-style
/// GQA decoder layer (real per-head `q_norm`/`k_norm` RMSNorm, RoPE-NEOX, bias-free SwiGLU MLP) --
/// same real per-layer tensor names/shapes as `OmniVoiceLlmTensorSource` already uses (`llm.
/// layers.{i}.*`), confirmed identical hyperparameters (28 layers, hidden=1024, 16 heads, 8 kv
/// heads, head_dim=128, ff=3072) from the checkpoint's own real `config.json`.</summary>
public sealed class OmniVoiceMaskGitLayerWeights
{
    public required float[] InputNorm { get; init; }
    public required float[] QProj { get; init; } // [numHeads*headDim, hidden]
    public required float[] KProj { get; init; } // [numKvHeads*headDim, hidden]
    public required float[] VProj { get; init; } // [numKvHeads*headDim, hidden]
    public required float[] OProj { get; init; } // [hidden, numHeads*headDim]
    public required float[] QNorm { get; init; } // [headDim]
    public required float[] KNorm { get; init; } // [headDim]
    public required float[] PostNorm { get; init; }
    public required float[] GateProj { get; init; } // [ffDim, hidden]
    public required float[] UpProj { get; init; } // [ffDim, hidden]
    public required float[] DownProj { get; init; } // [hidden, ffDim]
}

/// <summary>
/// Real weights for OmniVoice's own MaskGIT/SoundStorm-style non-causal audio generation
/// transformer, ported from `generator.cpp`'s `WeightsRuntime`/`load_layer_weights` (not guessed).
/// Genuinely self-contained (NOT routed through the shared `IForwardPass`/`ModelGraph` causal
/// engine, unlike every OTHER LLM bridge this session ported) -- real, confirmed non-causal full
/// self-attention over the whole real prompt+target sequence, real classifier-free-guidance dual
/// batching, real per-frame-summed 8-codebook embedding fusion. See
/// docs/audio-review-progress.md's "MaskGIT" entries for the full real algorithm derivation.
/// </summary>
public sealed class OmniVoiceMaskGitWeights
{
    public const int NumLayers = 28;
    public const int HiddenDim = 1024;
    public const int NumHeads = 16;
    public const int NumKvHeads = 8;
    public const int HeadDim = 128;
    public const int FfDim = 3072;
    public const int TextVocabSize = 151676;
    public const float RopeTheta = 1_000_000f;
    public const float RmsNormEps = 1e-6f;

    // Real config.json values, confirmed directly (not guessed).
    public const int NumCodebooks = 8;
    public const int AudioVocabSize = 1025; // 1024 real codes + 1 mask token
    public const int AudioMaskId = 1024;

    public required float[] TextEmbedding { get; init; } // [TextVocabSize, HiddenDim]
    public required float[] AudioEmbedding { get; init; } // [NumCodebooks*AudioVocabSize, HiddenDim]
    public required float[] AudioHead { get; init; } // [NumCodebooks*AudioVocabSize, HiddenDim], bias-free
    public required float[] FinalNorm { get; init; }
    public required OmniVoiceMaskGitLayerWeights[] Layers { get; init; }

    public static OmniVoiceMaskGitWeights Load(Func<string, float[]> get)
    {
        var layers = new OmniVoiceMaskGitLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"llm.layers.{i}.";
            layers[i] = new OmniVoiceMaskGitLayerWeights
            {
                InputNorm = get(p + "input_layernorm.weight"),
                QProj = get(p + "self_attn.q_proj.weight"),
                KProj = get(p + "self_attn.k_proj.weight"),
                VProj = get(p + "self_attn.v_proj.weight"),
                OProj = get(p + "self_attn.o_proj.weight"),
                QNorm = get(p + "self_attn.q_norm.weight"),
                KNorm = get(p + "self_attn.k_norm.weight"),
                PostNorm = get(p + "post_attention_layernorm.weight"),
                GateProj = get(p + "mlp.gate_proj.weight"),
                UpProj = get(p + "mlp.up_proj.weight"),
                DownProj = get(p + "mlp.down_proj.weight"),
            };
        }

        return new OmniVoiceMaskGitWeights
        {
            TextEmbedding = get("llm.embed_tokens.weight"),
            AudioEmbedding = get("audio_embeddings.weight"),
            AudioHead = get("audio_heads.weight"),
            FinalNorm = get("llm.norm.weight"),
            Layers = layers,
        };
    }
}
