
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real weight loader for the Qwen3-TTS 12Hz codec decoder's own 8-layer transformer
/// (`tok_dec.pre_tfm.*` in `models/qwen-tokenizer-12hz-Q8_0.gguf`).
///
/// <para><b>Genuinely NOT a <see cref="Engine.ForwardPass"/>-reuse case</b> (unlike the Talker/
/// Code Predictor, both confirmed reusable this fire): this layer variant has real per-layer
/// LayerScale (`attn_scale`/`ffn_scale`, matching the real `layer_scale_initial_scale=0.01`
/// config) applied to the attention/FFN sublayer output before the residual add, and has NO
/// QK-RMSNorm at all -- the opposite combination from the Talker/Code Predictor. Confirmed
/// directly against the real, local `examples/qwentts.cpp/src/tokenizer-transformer.h`
/// (`tok_trans_layer_forward`), transcribed math-for-math, not guessed.</para>
///
/// <para>Real config, confirmed via `list-metadata`: `hidden_size=512`, `latent_dim=1024`,
/// `num_hidden_layers=8`, `num_attention_heads=16`, `num_key_value_heads=16` (full MHA, NOT GQA),
/// `head_dim=64`, `intermediate_size=1024`, `sliding_window=72`, `rope_theta=10000` (NOT the
/// Talker's 1e6 -- a real, per-component difference), `rms_norm_eps=1e-5`. RoPE convention
/// confirmed NEOX (`ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX, ...)`) via the same real source.</para>
/// </summary>
public sealed class QwenTtsCodecTransformerWeights
{
    public int HiddenSize { get; }
    public int LatentDim { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int NumKvHeads { get; }
    public int HeadDim { get; }
    public int IntermediateSize { get; }
    public int SlidingWindow { get; }
    public float RopeTheta { get; }
    public float RmsNormEps { get; }

    public float[] InputProjWeight { get; }
    public float[] InputProjBias { get; }
    public QwenTtsCodecTransformerLayerWeights[] Layers { get; }
    public float[] NormWeight { get; }
    public float[] OutputProjWeight { get; }
    public float[] OutputProjBias { get; }

    public QwenTtsCodecTransformerWeights(GgufModel model)
    {
        HiddenSize = GetU32(model, "qwen3-tts-tokenizer.decoder.hidden_size");
        LatentDim = GetU32(model, "qwen3-tts-tokenizer.decoder.latent_dim");
        NumLayers = GetU32(model, "qwen3-tts-tokenizer.decoder.num_hidden_layers");
        NumHeads = GetU32(model, "qwen3-tts-tokenizer.decoder.num_attention_heads");
        NumKvHeads = GetU32(model, "qwen3-tts-tokenizer.decoder.num_key_value_heads");
        HeadDim = GetU32(model, "qwen3-tts-tokenizer.decoder.head_dim");
        IntermediateSize = GetU32(model, "qwen3-tts-tokenizer.decoder.intermediate_size");
        SlidingWindow = GetU32(model, "qwen3-tts-tokenizer.decoder.sliding_window");
        RopeTheta = Convert.ToSingle(model.Metadata["qwen3-tts-tokenizer.decoder.rope_theta"]);
        RmsNormEps = Convert.ToSingle(model.Metadata["qwen3-tts-tokenizer.decoder.rms_norm_eps"]);

        InputProjWeight = GetF32(model, "tok_dec.pre_tfm.input_proj.weight");
        InputProjBias = GetF32(model, "tok_dec.pre_tfm.input_proj.bias");
        NormWeight = GetF32(model, "tok_dec.pre_tfm.norm.weight");
        OutputProjWeight = GetF32(model, "tok_dec.pre_tfm.output_proj.weight");
        OutputProjBias = GetF32(model, "tok_dec.pre_tfm.output_proj.bias");

        Layers = new QwenTtsCodecTransformerLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"tok_dec.pre_tfm.blk.{i}";
            Layers[i] = new QwenTtsCodecTransformerLayerWeights
            {
                AttnNormWeight = GetF32(model, $"{p}.attn_norm.weight"),
                QWeight = GetF32(model, $"{p}.attn_q.weight"),
                KWeight = GetF32(model, $"{p}.attn_k.weight"),
                VWeight = GetF32(model, $"{p}.attn_v.weight"),
                OWeight = GetF32(model, $"{p}.attn_output.weight"),
                AttnScale = GetF32(model, $"{p}.attn_scale"),
                FfnNormWeight = GetF32(model, $"{p}.ffn_norm.weight"),
                GateWeight = GetF32(model, $"{p}.ffn_gate.weight"),
                UpWeight = GetF32(model, $"{p}.ffn_up.weight"),
                DownWeight = GetF32(model, $"{p}.ffn_down.weight"),
                FfnScale = GetF32(model, $"{p}.ffn_scale"),
            };
        }
    }

    private static int GetU32(GgufModel model, string key) => Convert.ToInt32(model.Metadata[key]);

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"QwenTTS codec transformer GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}

public sealed class QwenTtsCodecTransformerLayerWeights
{
    public required float[] AttnNormWeight { get; init; }
    public required float[] QWeight { get; init; }
    public required float[] KWeight { get; init; }
    public required float[] VWeight { get; init; }
    public required float[] OWeight { get; init; }
    public required float[] AttnScale { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required float[] GateWeight { get; init; }
    public required float[] UpWeight { get; init; }
    public required float[] DownWeight { get; init; }
    public required float[] FfnScale { get; init; }
}
