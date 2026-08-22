using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real Parler-TTS decoder weight loader (MusicGen-style causal decoder), loaded from the real
/// `parler-tts-mini-v1` checkpoint's `model.safetensors` (`models/parler-tts-mini-v1.safetensors`,
/// already downloaded this session for the T5 encoder -- same file, no new download).
///
/// <para>Real config, confirmed from the checkpoint's own `config.json`'s `decoder` block (not
/// guessed): `hidden_size=1024`, `num_hidden_layers=24`, `num_attention_heads=16`,
/// `num_key_value_heads=16` (full MHA, NOT GQA), `num_cross_attention_key_value_heads=16` (cross-
/// attn also full MHA), `ffn_dim=4096`, `activation_function=gelu` (plain, NOT gated),
/// `rope_embeddings=false` (real sinusoidal positional embeddings, precomputed and stored
/// directly as a real tensor -- `embed_positions.weights` -- not a formula to reimplement),
/// `num_codebooks=9`, `vocab_size=1088` (note: the INPUT embedding tables are sized
/// `vocab_size+1=1089` -- a real, intentional discrepancy confirmed from the real source's own
/// "TODO... +1 for pad token id... too late to change now" comment, not a bug), `bias=False` on
/// every Linear (self-attn/cross-attn/fc1/fc2).</para>
///
/// <para>Real tensor names (`decoder.model.decoder.*` prefix for the trunk,
/// `decoder.lm_heads.{0..8}.weight` for the 9 separate output heads,
/// `decoder.model.decoder.embed_tokens.{0..8}.weight` for the 9 separate input embedding
/// tables) -- confirmed via the real safetensors header, not assumed from the earlier session's
/// GGUF-derived names (which used a DIFFERENT prefix convention, `decoder.layers.*` without the
/// doubled `model.decoder`).</para>
/// </summary>
public sealed class ParlerDecoderWeights
{
    public const int HiddenDim = 1024;
    public const int NumLayers = 24;
    public const int NumHeads = 16;
    public const int FfnDim = 4096;
    public const int NumCodebooks = 9;
    public const int InputVocabSize = 1089; // embed_tokens: vocab_size + 1
    public const int OutputVocabSize = 1088; // lm_heads: vocab_size
    public const int MaxPositions = 4096;
    public const float LayerNormEps = 1e-5f; // real nn.LayerNorm default

    /// <summary>[NumCodebooks][InputVocabSize, HiddenDim] -- 9 separate real embedding tables.</summary>
    public float[][] EmbedTokens { get; } = new float[NumCodebooks][];
    /// <summary>[MaxPositions, HiddenDim] -- real precomputed sinusoidal positional embedding buffer, loaded directly (not recomputed).</summary>
    public float[] EmbedPositions { get; }
    public ParlerDecoderLayerWeights[] Layers { get; } = new ParlerDecoderLayerWeights[NumLayers];
    public float[] FinalLayerNormWeight { get; }
    public float[] FinalLayerNormBias { get; }
    /// <summary>[NumCodebooks][OutputVocabSize, HiddenDim] -- 9 separate real output heads, NOT tied to EmbedTokens.</summary>
    public float[][] LmHeads { get; } = new float[NumCodebooks][];

    public ParlerDecoderWeights(SafetensorsLoader loader)
    {
        for (int cb = 0; cb < NumCodebooks; cb++)
        {
            EmbedTokens[cb] = loader.ReadF32($"decoder.model.decoder.embed_tokens.{cb}.weight");
            LmHeads[cb] = loader.ReadF32($"decoder.lm_heads.{cb}.weight");
        }
        EmbedPositions = loader.ReadF32("decoder.model.decoder.embed_positions.weights");
        FinalLayerNormWeight = loader.ReadF32("decoder.model.decoder.layer_norm.weight");
        FinalLayerNormBias = loader.ReadF32("decoder.model.decoder.layer_norm.bias");

        // The 8 big matrices per layer are Q8_0-quantized at load time (once) -- same real
        // technique and rationale as Fish Speech's fast-AR (see Q8_0WeightQuantizer's doc
        // comment): this decoder's autoregressive per-step decode re-reads its full weight set
        // every generated token, so the same memory-bandwidth argument applies. HiddenDim=1024
        // and FfnDim=4096 are both cleanly divisible by 32 -- no partial-block handling needed.
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"decoder.model.decoder.layers.{i}";
            Layers[i] = new ParlerDecoderLayerWeights
            {
                SelfAttnQWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.self_attn.q_proj.weight"), HiddenDim, HiddenDim),
                SelfAttnKWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.self_attn.k_proj.weight"), HiddenDim, HiddenDim),
                SelfAttnVWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.self_attn.v_proj.weight"), HiddenDim, HiddenDim),
                SelfAttnOutWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.self_attn.out_proj.weight"), HiddenDim, HiddenDim),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.self_attn_layer_norm.weight"),
                SelfAttnLayerNormBias = loader.ReadF32($"{p}.self_attn_layer_norm.bias"),
                CrossAttnQWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.encoder_attn.q_proj.weight"), HiddenDim, HiddenDim),
                CrossAttnKWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.encoder_attn.k_proj.weight"), HiddenDim, HiddenDim),
                CrossAttnVWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.encoder_attn.v_proj.weight"), HiddenDim, HiddenDim),
                CrossAttnOutWeight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.encoder_attn.out_proj.weight"), HiddenDim, HiddenDim),
                CrossAttnLayerNormWeight = loader.ReadF32($"{p}.encoder_attn_layer_norm.weight"),
                CrossAttnLayerNormBias = loader.ReadF32($"{p}.encoder_attn_layer_norm.bias"),
                Fc1Weight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.fc1.weight"), FfnDim, HiddenDim),
                Fc2Weight = Q8_0WeightQuantizer.QuantizeRef(loader.ReadF32($"{p}.fc2.weight"), HiddenDim, FfnDim),
                FinalLayerNormWeight = loader.ReadF32($"{p}.final_layer_norm.weight"),
                FinalLayerNormBias = loader.ReadF32($"{p}.final_layer_norm.bias"),
            };
        }
    }

    /// <summary>
    /// Real GGUF loader, for the community `ecyht2/parler-tts-mini-v1-GGUF` conversion of the
    /// SAME real `parler-tts/parler-tts-mini-v1` checkpoint the Safetensors constructor loads --
    /// confirmed via this GGUF's own real metadata (`hidden_size=1024`, `attn_heads=16`,
    /// `num_hidden_layers=24`, `out_vocab_size=1088`, `parler-tts.decoder_start_token_id=1025`,
    /// `output_heads=9` -- exact matches to this class's own constants) and real tensor names
    /// (`decoder.layers.{i}.{self_attn,encoder_attn}.{q,k,v,out}_proj.weight`,
    /// `decoder.layers.{i}.fc{1,2}.weight`, `decoder.embed_tokens.{cb}.weight`,
    /// `decoder.lm_heads.{cb}.weight.head` -- note the real GGUF's simpler `decoder.*` prefix vs.
    /// the Safetensors checkpoint's doubled `decoder.model.decoder.*`, and the real trailing
    /// `.head` on lm_head names, confirmed via `list-tensors`, not assumed).
    ///
    /// <para><b>Real, confirmed limitation of this specific community conversion</b>: it contains
    /// ONLY `decoder.*` and `audio_encoder.*` (DAC codec) tensors -- NO `text_encoder.*` (T5)
    /// tensors at all, confirmed via an exhaustive tensor-name-prefix scan. This loader is
    /// therefore only ever used for the decoder half of a pipeline; the T5 encoder must still come
    /// from the Safetensors checkpoint (see docs/audio-review-progress.md's GGUF-matrix
    /// entries).</para>
    ///
    /// <para>Unlike the Safetensors path (plain float32 on disk, re-quantized to Q8_0 once here),
    /// this GGUF conversion's big matrices are ALREADY real on-disk Q8_0 (confirmed per-tensor via
    /// `list-tensors` -- self_attn Q/K/V/O and fc1/fc2 are `Q8_0`; oddly, `encoder_attn.k_proj`/
    /// `v_proj` specifically are stored as plain `Float32` while `q_proj`/`out_proj` are `Q8_0` --
    /// a real, observed converter quirk, not something this loader "fixes": every tensor's REAL
    /// dtype is read directly via <see cref="NativeGgufWeightRef"/>, which dispatches correctly
    /// regardless of which dtype a given tensor actually has.</para>
    /// </summary>
    public ParlerDecoderWeights(GgufModel model)
    {
        for (int cb = 0; cb < NumCodebooks; cb++)
        {
            EmbedTokens[cb] = GetF32(model, $"decoder.embed_tokens.{cb}.weight");
            LmHeads[cb] = GetF32(model, $"decoder.lm_heads.{cb}.weight.head");
        }
        EmbedPositions = GetF32(model, "decoder.embed_positions.weights");
        FinalLayerNormWeight = GetF32(model, "decoder.layer_norm.weight");
        FinalLayerNormBias = GetF32(model, "decoder.layer_norm.bias");

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"decoder.layers.{i}";
            Layers[i] = new ParlerDecoderLayerWeights
            {
                SelfAttnQWeight = NativeGgufWeightRef.Get(model, $"{p}.self_attn.q_proj.weight"),
                SelfAttnKWeight = NativeGgufWeightRef.Get(model, $"{p}.self_attn.k_proj.weight"),
                SelfAttnVWeight = NativeGgufWeightRef.Get(model, $"{p}.self_attn.v_proj.weight"),
                SelfAttnOutWeight = NativeGgufWeightRef.Get(model, $"{p}.self_attn.out_proj.weight"),
                SelfAttnLayerNormWeight = GetF32(model, $"{p}.self_attn_layer_norm.weight"),
                SelfAttnLayerNormBias = GetF32(model, $"{p}.self_attn_layer_norm.bias"),
                CrossAttnQWeight = NativeGgufWeightRef.Get(model, $"{p}.encoder_attn.q_proj.weight"),
                CrossAttnKWeight = NativeGgufWeightRef.Get(model, $"{p}.encoder_attn.k_proj.weight"),
                CrossAttnVWeight = NativeGgufWeightRef.Get(model, $"{p}.encoder_attn.v_proj.weight"),
                CrossAttnOutWeight = NativeGgufWeightRef.Get(model, $"{p}.encoder_attn.out_proj.weight"),
                CrossAttnLayerNormWeight = GetF32(model, $"{p}.encoder_attn_layer_norm.weight"),
                CrossAttnLayerNormBias = GetF32(model, $"{p}.encoder_attn_layer_norm.bias"),
                Fc1Weight = NativeGgufWeightRef.Get(model, $"{p}.fc1.weight"),
                Fc2Weight = NativeGgufWeightRef.Get(model, $"{p}.fc2.weight"),
                FinalLayerNormWeight = GetF32(model, $"{p}.final_layer_norm.weight"),
                FinalLayerNormBias = GetF32(model, $"{p}.final_layer_norm.bias"),
            };
        }
    }

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new System.IO.InvalidDataException($"Parler GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}

public sealed class ParlerDecoderLayerWeights
{
    public required IQuantWeightRef SelfAttnQWeight { get; init; }
    public required IQuantWeightRef SelfAttnKWeight { get; init; }
    public required IQuantWeightRef SelfAttnVWeight { get; init; }
    public required IQuantWeightRef SelfAttnOutWeight { get; init; }
    public required float[] SelfAttnLayerNormWeight { get; init; }
    public required float[] SelfAttnLayerNormBias { get; init; }
    public required IQuantWeightRef CrossAttnQWeight { get; init; }
    public required IQuantWeightRef CrossAttnKWeight { get; init; }
    public required IQuantWeightRef CrossAttnVWeight { get; init; }
    public required IQuantWeightRef CrossAttnOutWeight { get; init; }
    public required float[] CrossAttnLayerNormWeight { get; init; }
    public required float[] CrossAttnLayerNormBias { get; init; }
    public required IQuantWeightRef Fc1Weight { get; init; }
    public required IQuantWeightRef Fc2Weight { get; init; }
    public required float[] FinalLayerNormWeight { get; init; }
    public required float[] FinalLayerNormBias { get; init; }
}
