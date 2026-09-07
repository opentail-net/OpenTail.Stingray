using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// One GPT2-style transformer layer's weights for MOSS-TTS-Nano's global transformer, ported from
/// `examples/audio.cpp/src/models/moss/moss_tts_nano/global_transformer.cpp`'s
/// `TransformerLayerWeights`/loader (real `moss-tts-nano-100m-q8_0.gguf` tensor names, not
/// guessed -- confirmed via `MossTtsTensorNameDumpDebugTest`). All weight tensors are read as
/// native-GGUF-order `[in,out]`, which dequantizes to exactly the row-major `[outDim,inDim]`
/// layout `SimdKernels.MatVecF32` expects -- no transpose needed (same convention already used by
/// `ChatterboxAcousticLm`/`ChatterboxWeights`).
/// </summary>
public sealed class MossTtsGlobalTransformerLayerWeights
{
    public required float[] Ln1Weight { get; init; }
    public required float[] Ln1Bias { get; init; }
    public required float[] CAttnWeight { get; init; } // [3*HiddenDim, HiddenDim]
    public required float[] CAttnBias { get; init; }
    public required float[] CProjWeight { get; init; } // [HiddenDim, HiddenDim]
    public required float[] CProjBias { get; init; }
    public required float[] Ln2Weight { get; init; }
    public required float[] Ln2Bias { get; init; }
    public required float[] FcInWeight { get; init; }  // [IntermediateDim, HiddenDim]
    public required float[] FcInBias { get; init; }
    public required float[] FcOutWeight { get; init; } // [HiddenDim, IntermediateDim]
    public required float[] FcOutBias { get; init; }
}

/// <summary>
/// Real weight loader for MOSS-TTS-Nano's "global transformer" -- a standard GPT2-architecture
/// causal decoder (real config: `gpt2_config` in the checkpoint's embedded `config.json`, extracted
/// via `MossTtsMetaDumpDebugTest`: 12 layers, 12 heads, hidden=768, intermediate=3072,
/// vocab=16384, `position_embedding_type=rope`, `rope_base=10000`, `activation_function=gelu_new`)
/// that consumes an interleaved sequence of one text-token embedding plus (for audio frames) 16
/// summed RVQ-codebook audio-token embeddings per row, and produces one hidden state per row for
/// the local transformer / text head to consume. Real prefix: `model_weights/transformer.*` (see
/// `global_transformer.cpp`'s constructor); the 16 real `model_weights/audio_embeddings.{q}.weight`
/// tables live alongside it, indexed by RVQ codebook.
/// </summary>
public sealed class MossTtsGlobalTransformerWeights
{
    public const int HiddenDim = 768;
    public const int NumLayers = 12;
    public const int NumHeads = 12;
    public const int HeadDim = HiddenDim / NumHeads; // 64
    public const int IntermediateDim = 3072;
    public const int VocabSize = 16384;
    public const int NumCodebooks = 16;
    public const int AudioCodebookSize = 1024;
    public const int AudioPadTokenId = 1024; // real audio_pad_token_id from config.json
    public const int PadTokenId = 3;         // real pad_token_id from config.json
    public const int ImStartTokenId = 4;
    public const int ImEndTokenId = 5;
    public const int AudioStartTokenId = 6;
    public const int AudioEndTokenId = 7;
    public const int AudioUserSlotTokenId = 8;
    public const int AudioAssistantSlotTokenId = 9;
    public const float LayerNormEps = 1e-5f;
    public const float RopeBase = 10000f;

    public float[] TextEmbedding { get; }        // [VocabSize, HiddenDim]
    public float[][] AudioEmbeddings { get; }     // [NumCodebooks][AudioCodebookSize, HiddenDim]
    public MossTtsGlobalTransformerLayerWeights[] Layers { get; } = new MossTtsGlobalTransformerLayerWeights[NumLayers];
    public float[] FinalNormWeight { get; }
    public float[] FinalNormBias { get; }
    public float[] TextLmHeadWeight { get; }      // [VocabSize, HiddenDim], no bias (tied embedding checkpoint)

    public MossTtsGlobalTransformerWeights(RvcPackedTensorSource source) : this(source.GetTensor)
    {
    }

    /// <summary>Test-friendly overload taking a raw name-&gt;tensor getter (avoids needing a real GGUF
    /// for synthetic-weight structural tests).</summary>
    public MossTtsGlobalTransformerWeights(Func<string, float[]> get)
    {
        TextEmbedding = get("model_weights/transformer.wte.weight");
        AudioEmbeddings = new float[NumCodebooks][];
        for (int q = 0; q < NumCodebooks; q++)
            AudioEmbeddings[q] = get($"model_weights/audio_embeddings.{q}.weight");

        for (int l = 0; l < NumLayers; l++)
        {
            string p = $"model_weights/transformer.h.{l}";
            Layers[l] = new MossTtsGlobalTransformerLayerWeights
            {
                Ln1Weight = get($"{p}.ln_1.weight"),
                Ln1Bias = get($"{p}.ln_1.bias"),
                CAttnWeight = get($"{p}.attn.c_attn.weight"),
                CAttnBias = get($"{p}.attn.c_attn.bias"),
                CProjWeight = get($"{p}.attn.c_proj.weight"),
                CProjBias = get($"{p}.attn.c_proj.bias"),
                Ln2Weight = get($"{p}.ln_2.weight"),
                Ln2Bias = get($"{p}.ln_2.bias"),
                FcInWeight = get($"{p}.mlp.fc_in.weight"),
                FcInBias = get($"{p}.mlp.fc_in.bias"),
                FcOutWeight = get($"{p}.mlp.fc_out.weight"),
                FcOutBias = get($"{p}.mlp.fc_out.bias"),
            };
        }

        FinalNormWeight = get("model_weights/transformer.ln_f.weight");
        FinalNormBias = get("model_weights/transformer.ln_f.bias");
        TextLmHeadWeight = get("model_weights/text_lm_head.weight");
    }
}
