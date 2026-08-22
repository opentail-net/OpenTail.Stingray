using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Real GGUF weight/config loader for the parts of Fish Speech S2 Pro needed by the slow-AR
/// semantic-token generation loop (talker only -- the fast-AR codebook expander and the codec
/// are separate, not-yet-ported components, see docs/audio-review-progress.md's Fish Speech
/// section). Confirmed real metadata/tensor names via `list-metadata`/`list-tensors` on
/// `models/s2-pro-q4_k_m.gguf`, cross-checked against `examples/s2.cpp/src/s2_model.cpp`'s real
/// embedding-composition code (not guessed).
/// </summary>
public sealed class FishSpeechWeights : IDisposable
{
    public GgufModel Model { get; }

    public int EmbeddingDim { get; }
    public int VocabSize { get; }
    public int NumCodebooks { get; }
    public int CodebookSize { get; }
    public int SemanticBeginId { get; }
    public int SemanticEndId { get; }
    public bool ScaleCodebookEmbeddings { get; }

    /// <summary>[VocabSize, EmbeddingDim] flat row-major -- the real `embeddings.weight`, tied with the LM output head.</summary>
    public float[] Embeddings { get; }

    /// <summary>[NumCodebooks * CodebookSize, EmbeddingDim] flat row-major -- the real shared `codebook_embeddings.weight`, indexed by `value + codebookIndex * CodebookSize` (confirmed from real source, see FishSpeechPipeline's doc comment).</summary>
    public float[] CodebookEmbeddings { get; }

    // ── Fast-AR (per-codebook expansion transformer) -- real spec confirmed from
    // examples/s2.cpp/src/s2_model.cpp's real fast_decode forward pass, see FishSpeechFastAr's
    // doc comment for the full derivation.
    public int FastEmbeddingDim { get; }
    public int FastHeadCount { get; }
    public int FastHeadCountKv { get; }
    public int FastHeadDim { get; }
    public int FastBlockCount { get; }
    public float FastRopeFreqBase { get; }
    public int FastContextLength { get; }
    public float FastRmsNormEps { get; }
    public bool FastAttentionQkNorm { get; }

    /// <summary>[CodebookSize, FastEmbeddingDim] -- single shared table, plain (non-offset) lookup by raw codebook value.</summary>
    public float[] FastEmbeddings { get; }
    public FishSpeechFastLayerWeights[] FastLayers { get; }
    public float[] FastNormWeight { get; }
    /// <summary>[CodebookSize, FastEmbeddingDim] -- separate, NOT tied to FastEmbeddings (fast_tie_word_embeddings=false).</summary>
    public float[] FastOutputWeight { get; }

    public FishSpeechWeights(string ggufPath)
    {
        Model = GgufModel.Open(ggufPath);

        EmbeddingDim = GetU32("fish-speech.embedding_length", 2560);
        VocabSize = GetU32("fish-speech.vocab_size", 155776);
        NumCodebooks = GetU32("fish_speech.num_codebooks", 10);
        CodebookSize = GetU32("fish_speech.codebook_size", 4096);
        SemanticBeginId = GetU32("fish_speech.semantic_begin_id", 0);
        SemanticEndId = GetU32("fish_speech.semantic_end_id", 0);
        ScaleCodebookEmbeddings = Model.Metadata.TryGetValue("fish_speech.scale_codebook_embeddings", out var s) && s is bool b && b;

        Embeddings = GetTensor("embeddings.weight");
        CodebookEmbeddings = GetTensor("codebook_embeddings.weight");

        FastEmbeddingDim = GetU32("fish_speech.fast_embedding_length", 2560);
        FastHeadCount = GetU32("fish_speech.fast_head_count", 32);
        FastHeadCountKv = GetU32("fish_speech.fast_head_count_kv", 8);
        FastHeadDim = GetU32("fish_speech.fast_head_dim", 128);
        FastBlockCount = GetU32("fish_speech.fast_block_count", 4);
        FastRopeFreqBase = Model.Metadata.TryGetValue("fish_speech.fast_rope_freq_base", out var frb) ? Convert.ToSingle(frb) : 1_000_000f;
        FastContextLength = GetU32("fish_speech.fast_context_length", 11);
        FastRmsNormEps = Model.Metadata.TryGetValue("fish_speech.fast_layer_norm_rms_eps", out var eps) ? Convert.ToSingle(eps) : 1e-6f;
        FastAttentionQkNorm = Model.Metadata.TryGetValue("fish_speech.fast_attention_qk_norm", out var qkn) && qkn is bool qknb && qknb;

        FastEmbeddings = GetTensor("fast_embeddings.weight");
        FastNormWeight = GetTensor("fast_norm.weight");
        FastOutputWeight = GetTensor("fast_output.weight");

        FastLayers = new FishSpeechFastLayerWeights[FastBlockCount];
        for (int i = 0; i < FastBlockCount; i++)
        {
            string p = $"fast_layers.{i}";
            FastLayers[i] = new FishSpeechFastLayerWeights
            {
                AttentionNormWeight = GetTensor($"{p}.attention_norm.weight"),
                WqkvWeight = GetTensor($"{p}.attention.wqkv.weight"),
                WoWeight = GetTensor($"{p}.attention.wo.weight"),
                FfnNormWeight = GetTensor($"{p}.ffn_norm.weight"),
                W1Weight = GetTensor($"{p}.feed_forward.w1.weight"),
                W2Weight = GetTensor($"{p}.feed_forward.w2.weight"),
                W3Weight = GetTensor($"{p}.feed_forward.w3.weight"),
            };
        }
    }

    private int GetU32(string key, int fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Fish Speech GGUF missing required tensor '{name}'.");
        var bytes = Model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose() => Model.Dispose();
}

public sealed class FishSpeechFastLayerWeights
{
    public required float[] AttentionNormWeight { get; init; }
    public required float[] WqkvWeight { get; init; }
    public required float[] WoWeight { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required float[] W1Weight { get; init; } // gate
    public required float[] W2Weight { get; init; } // down
    public required float[] W3Weight { get; init; } // up
}
