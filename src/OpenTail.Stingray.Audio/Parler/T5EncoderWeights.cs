using System;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real T5 text encoder weight loader for Parler-TTS's `text_encoder` (google/flan-t5-large
/// shape: d_model=1024, d_ff=2816, d_kv=64, num_layers=24, num_heads=16,
/// relative_attention_num_buckets=32, relative_attention_max_distance=128,
/// layer_norm_epsilon=1e-6, feed_forward_proj=gated-gelu -- confirmed from the real
/// `parler-tts/parler-tts-mini-v1` HF `config.json`, see docs/audio-review-progress.md's
/// Parler-TTS section).
///
/// <para><b>Loads from the REAL `parler-tts-mini-v1` checkpoint's own `model.safetensors`, NOT a
/// stock `google/flan-t5-large` checkpoint</b> -- confirmed via the real `parler-tts` Python
/// package's training code (`training/arguments.py`: `freeze_text_encoder` defaults to `False`)
/// that the text encoder is fine-tuned jointly with the rest of the model, not frozen. A stock
/// flan-t5-large checkpoint would very likely be numerically wrong for this specific model.</para>
///
/// <para>Real tensor names (`text_encoder.*` prefix in the safetensors file, stripped here):
/// `shared.weight` (token embedding, tied with nothing else -- T5 encoder-only has no output
/// head), `encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight` (no bias, real T5
/// convention), `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`
/// (`[32, 16]` = `[num_buckets, num_heads]` -- present ONLY on block 0; the computed bias is
/// shared/reused across all 24 layers, not recomputed per layer, confirmed from the real
/// `transformers` T5 source), `encoder.block.{i}.layer.0.layer_norm.weight` /
/// `encoder.block.{i}.layer.1.{DenseReluDense.{wi_0,wi_1,wo},layer_norm}.weight`,
/// `encoder.final_layer_norm.weight`.</para>
/// </summary>
public sealed class T5EncoderWeights
{
    public const int DModel = 1024;
    public const int DFf = 2816;
    public const int DKv = 64;
    public const int NumLayers = 24;
    public const int NumHeads = 16;
    public const int RelativeAttentionNumBuckets = 32;
    public const int RelativeAttentionMaxDistance = 128;
    public const float LayerNormEps = 1e-6f;

    public float[] SharedEmbedding { get; } // [vocab, DModel]
    public T5LayerWeights[] Layers { get; } = new T5LayerWeights[NumLayers];
    public float[] FinalLayerNormWeight { get; }
    public float[] RelativeAttentionBias { get; } // [RelativeAttentionNumBuckets, NumHeads], block 0 only

    public T5EncoderWeights(SafetensorsLoader loader)
    {
        SharedEmbedding = loader.ReadF32("text_encoder.shared.weight");
        FinalLayerNormWeight = loader.ReadF32("text_encoder.encoder.final_layer_norm.weight");
        RelativeAttentionBias = loader.ReadF32("text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"text_encoder.encoder.block.{i}";
            Layers[i] = new T5LayerWeights
            {
                SelfAttnQWeight = loader.ReadF32($"{p}.layer.0.SelfAttention.q.weight"),
                SelfAttnKWeight = loader.ReadF32($"{p}.layer.0.SelfAttention.k.weight"),
                SelfAttnVWeight = loader.ReadF32($"{p}.layer.0.SelfAttention.v.weight"),
                SelfAttnOWeight = loader.ReadF32($"{p}.layer.0.SelfAttention.o.weight"),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.layer.0.layer_norm.weight"),
                FfnWi0Weight = loader.ReadF32($"{p}.layer.1.DenseReluDense.wi_0.weight"),
                FfnWi1Weight = loader.ReadF32($"{p}.layer.1.DenseReluDense.wi_1.weight"),
                FfnWoWeight = loader.ReadF32($"{p}.layer.1.DenseReluDense.wo.weight"),
                FfnLayerNormWeight = loader.ReadF32($"{p}.layer.1.layer_norm.weight"),
            };
        }
    }
}

public sealed class T5LayerWeights
{
    public required float[] SelfAttnQWeight { get; init; }
    public required float[] SelfAttnKWeight { get; init; }
    public required float[] SelfAttnVWeight { get; init; }
    public required float[] SelfAttnOWeight { get; init; }
    public required float[] SelfAttnLayerNormWeight { get; init; }
    public required float[] FfnWi0Weight { get; init; }
    public required float[] FfnWi1Weight { get; init; }
    public required float[] FfnWoWeight { get; init; }
    public required float[] FfnLayerNormWeight { get; init; }
}
