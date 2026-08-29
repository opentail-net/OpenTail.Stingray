
namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Container for Chatterbox-Turbo T3 (Acoustic LM, a GPT2-medium-architecture transformer -- see
/// examples/chatterbox-tts-py/chatterbox/models/t3/t3.py and llama_configs.py's GPT2_MEDIUM_CONFIG)
/// and S3Gen (Vocoder/Flow Decoder, not yet real -- see ChatterboxDecoder.cs) GGUF weights.
/// All tensors are eagerly dequantized to float32 in file storage order, which for a GGUF-converted
/// nn.Linear is [outFeatures, inFeatures] row-major (matches the "wBase = outIdx*inFeatures" access
/// pattern used throughout this file and ChatterboxAcousticLm.cs) -- the same ggml-dims-reversed-vs-
/// torch-shape convention established and verified for Kokoro (see docs/audio-review-progress.md).
/// </summary>
public sealed class ChatterboxWeights : IDisposable
{
    public GgufModel T3Model { get; }
    public GgufModel? S3GenModel { get; }
    public GgufTokenizer Tokenizer { get; }

    public int NumLayers { get; }
    public int HiddenDim { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int IntermediateSize { get; }
    public int TextVocabSize { get; }
    public int SpeechVocabSize { get; }
    public int MaxPositions { get; }
    public int SpeakerEmbedSize { get; }

    public int StartTextToken { get; }
    public int StopTextToken { get; }
    public int StartSpeechToken { get; }
    public int StopSpeechToken { get; }
    public int SpeechCondPromptLen { get; }

    // --- Conditioning (built-in default voice, baked into the GGUF) ---
    public float[]? SpeakerEmbedding { get; }        // conds.t3.speaker_emb [256]
    public int[]? SpeechPromptTokens { get; }         // conds.t3.speech_prompt_tokens [375]

    // --- S3Gen conditioning (ChatterboxTurboTTS.Conditionals.gen dict, baked-in default voice) ---
    public float[]? GenEmbedding { get; }             // conds.gen.embedding [192] -- CAMPPlus x-vector
    public float[]? GenPromptFeat { get; }            // conds.gen.prompt_feat [80, 500] -- reference mel
    public int[]? GenPromptToken { get; }             // conds.gen.prompt_token [250] -- reference S3 tokens

    // --- Top-level ---
    public float[] TextEmbWeight { get; }             // [TextVocabSize, HiddenDim]
    public float[] SpeechEmbWeight { get; }            // [SpeechVocabSize, HiddenDim]
    public float[] WpeWeight { get; }                  // [MaxPositions, HiddenDim]
    public float[] OutputNormWeight { get; }
    public float[] OutputNormBias { get; }
    public float[] SpeechHeadWeight { get; }           // [SpeechVocabSize, HiddenDim]
    public float[] SpeechHeadBias { get; }             // [SpeechVocabSize]
    public float[] SpkrEncWeight { get; }              // [HiddenDim, SpeakerEmbedSize]
    public float[] SpkrEncBias { get; }                // [HiddenDim]

    public ChatterboxT3Layer[] Layers { get; }

    public ChatterboxWeights(string t3Path, string? s3GenPath = null)
    {
        if (!File.Exists(t3Path))
            throw new FileNotFoundException($"Chatterbox T3 model file not found: {t3Path}");

        T3Model = GgufModel.Open(t3Path);
        Tokenizer = GgufTokenizer.FromGgufModel(T3Model);

        NumLayers = GetInt("chatterbox.t3.n_layers", 24);
        HiddenDim = GetInt("chatterbox.t3.hidden_size", 1024);
        NumHeads = GetInt("chatterbox.t3.n_heads", 16);
        HeadDim = GetInt("chatterbox.t3.head_dim", HiddenDim / NumHeads);
        IntermediateSize = GetInt("chatterbox.t3.intermediate_size", 4096);
        TextVocabSize = GetInt("chatterbox.t3.text_vocab_size", 50276);
        SpeechVocabSize = GetInt("chatterbox.t3.speech_vocab_size", 6563);
        MaxPositions = GetInt("chatterbox.t3.wpe_max_positions", 8196);
        SpeakerEmbedSize = GetInt("chatterbox.t3.speaker_embed_size", 256);

        StartTextToken = GetInt("chatterbox.t3.start_text_token", 255);
        StopTextToken = GetInt("chatterbox.t3.stop_text_token", 0);
        StartSpeechToken = GetInt("chatterbox.t3.start_speech_token", 6561);
        StopSpeechToken = GetInt("chatterbox.t3.stop_speech_token", 6562);
        SpeechCondPromptLen = GetInt("chatterbox.t3.speech_cond_prompt_len", 375);

        SpeakerEmbedding = TryGetTensor("conds.t3.speaker_emb");
        SpeechPromptTokens = TryGetIntTensor("conds.t3.speech_prompt_tokens");

        GenEmbedding = TryGetTensor("conds.gen.embedding");
        GenPromptFeat = TryGetTensor("conds.gen.prompt_feat");
        GenPromptToken = TryGetIntTensor("conds.gen.prompt_token");

        TextEmbWeight = GetTensor("t3.text_emb.weight");
        SpeechEmbWeight = GetTensor("t3.speech_emb.weight");
        WpeWeight = GetTensor("t3.wpe.weight");
        OutputNormWeight = GetTensor("t3.output_norm.weight");
        OutputNormBias = GetTensor("t3.output_norm.bias");
        SpeechHeadWeight = GetTensor("t3.speech_head.weight");
        SpeechHeadBias = GetTensor("t3.speech_head.bias");
        SpkrEncWeight = GetTensor("t3.cond.spkr_enc.weight");
        SpkrEncBias = GetTensor("t3.cond.spkr_enc.bias");

        Layers = new ChatterboxT3Layer[NumLayers];
        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new ChatterboxT3Layer(this, $"t3.blk.{i}");

        if (s3GenPath != null && File.Exists(s3GenPath))
        {
            S3GenModel = GgufModel.Open(s3GenPath);
        }
    }

    private int GetInt(string key, int fallback) =>
        T3Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    public float[] GetTensor(string name)
    {
        var info = T3Model.FindTensor(name) ?? throw new InvalidDataException($"Chatterbox T3 GGUF missing required tensor '{name}'.");
        return DequantTensor(T3Model, info);
    }

    public float[]? TryGetTensor(string name)
    {
        var info = T3Model.FindTensor(name);
        return info is null ? null : DequantTensor(T3Model, info.Value);
    }

    /// <summary>Reads an Int32 GGUF tensor (e.g. token-id conditioning tensors) without dequantization.</summary>
    private int[]? TryGetIntTensor(string name)
    {
        var info = T3Model.FindTensor(name);
        if (info is null) return null;
        var bytes = T3Model.GetTensorData(info.Value);
        int count = (int)info.Value.ElementCount;
        var result = new int[count];
        for (int i = 0; i < count && (i * 4 + 4) <= bytes.Length; i++)
            result[i] = BitConverter.ToInt32(bytes.Slice(i * 4, 4));
        return result;
    }

    private static float[] DequantTensor(GgufModel model, GgufTensorInfo info)
    {
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose()
    {
        S3GenModel?.Dispose();
        T3Model.Dispose();
    }
}

/// <summary>One GPT2-medium transformer block's weights (pre-LN attn + pre-LN GELU MLP).</summary>
public sealed class ChatterboxT3Layer
{
    public float[] AttnNormWeight { get; }
    public float[] AttnNormBias { get; }
    public float[] AttnQkvWeight { get; }   // [3*HiddenDim, HiddenDim]
    public float[] AttnQkvBias { get; }     // [3*HiddenDim]
    public float[] AttnOutputWeight { get; } // [HiddenDim, HiddenDim]
    public float[] AttnOutputBias { get; }
    public float[] FfnNormWeight { get; }
    public float[] FfnNormBias { get; }
    public float[] FfnFcWeight { get; }     // [IntermediateSize, HiddenDim]
    public float[] FfnFcBias { get; }
    public float[] FfnProjWeight { get; }   // [HiddenDim, IntermediateSize]
    public float[] FfnProjBias { get; }

    public ChatterboxT3Layer(ChatterboxWeights w, string prefix)
    {
        AttnNormWeight = w.GetTensor($"{prefix}.attn_norm.weight");
        AttnNormBias = w.GetTensor($"{prefix}.attn_norm.bias");
        AttnQkvWeight = w.GetTensor($"{prefix}.attn_qkv.weight");
        AttnQkvBias = w.GetTensor($"{prefix}.attn_qkv.bias");
        AttnOutputWeight = w.GetTensor($"{prefix}.attn_output.weight");
        AttnOutputBias = w.GetTensor($"{prefix}.attn_output.bias");
        FfnNormWeight = w.GetTensor($"{prefix}.ffn_norm.weight");
        FfnNormBias = w.GetTensor($"{prefix}.ffn_norm.bias");
        FfnFcWeight = w.GetTensor($"{prefix}.ffn_fc.weight");
        FfnFcBias = w.GetTensor($"{prefix}.ffn_fc.bias");
        FfnProjWeight = w.GetTensor($"{prefix}.ffn_proj.weight");
        FfnProjBias = w.GetTensor($"{prefix}.ffn_proj.bias");
    }
}
