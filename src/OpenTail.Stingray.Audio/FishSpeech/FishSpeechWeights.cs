
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
    public float RmsNormEps { get; }

    /// <summary>[VocabSize, EmbeddingDim] flat row-major -- the real `embeddings.weight`, tied with the LM output head.</summary>
    public float[] Embeddings { get; }

    /// <summary>[NumCodebooks * CodebookSize, EmbeddingDim] flat row-major -- the real shared `codebook_embeddings.weight`, indexed by `value + codebookIndex * CodebookSize` (confirmed from real source, see FishSpeechPipeline's doc comment).</summary>
    public float[] CodebookEmbeddings { get; }

    /// <summary>[EmbeddingDim] -- the slow-AR's final RMSNorm weight `norm.weight`, applied before feeding hidden state to fast-AR.</summary>
    public float[] NormWeight { get; }

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
    public float[,] FastRopeCos { get; }
    public float[,] FastRopeSin { get; }

    /// <summary>[CodebookSize, FastEmbeddingDim] -- single shared table, plain (non-offset) lookup by raw codebook value.</summary>
    public float[] FastEmbeddings { get; }
    public FishSpeechFastLayerWeights[] FastLayers { get; }
    public float[] FastNormWeight { get; }
    /// <summary>[CodebookSize, FastEmbeddingDim], real Q8_0 block format (see Q8_0WeightQuantizer) -- separate, NOT tied to FastEmbeddings (fast_tie_word_embeddings=false).</summary>
    public byte[] FastOutputWeight { get; }

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
        RmsNormEps = Model.Metadata.TryGetValue("fish-speech.attention.layer_norm_rms_epsilon", out var epsMain) ? Convert.ToSingle(epsMain) : 1e-6f;

        Embeddings = GetTensor("embeddings.weight");
        CodebookEmbeddings = GetTensor("codebook_embeddings.weight");
        NormWeight = GetTensor("norm.weight");

        FastEmbeddingDim = GetU32("fish_speech.fast_embedding_length", 2560);
        FastHeadCount = GetU32("fish_speech.fast_head_count", 32);
        FastHeadCountKv = GetU32("fish_speech.fast_head_count_kv", 8);
        FastHeadDim = GetU32("fish_speech.fast_head_dim", 128);
        FastBlockCount = GetU32("fish_speech.fast_block_count", 4);
        FastRopeFreqBase = Model.Metadata.TryGetValue("fish_speech.fast_rope_freq_base", out var frb) ? Convert.ToSingle(frb) : 1_000_000f;
        FastContextLength = GetU32("fish_speech.fast_context_length", 11);
        FastRmsNormEps = Model.Metadata.TryGetValue("fish_speech.fast_layer_norm_rms_eps", out var eps) ? Convert.ToSingle(eps) : 1e-6f;
        FastAttentionQkNorm = Model.Metadata.TryGetValue("fish_speech.fast_attention_qk_norm", out var qkn) && qkn is bool qknb && qknb;

        int half = FastHeadDim / 2;
        int maxPos = Math.Max(FastContextLength, 16);
        FastRopeCos = new float[maxPos, half];
        FastRopeSin = new float[maxPos, half];
        for (int p = 0; p < maxPos; p++)
        {
            for (int i = 0; i < half; i++)
            {
                float freq = 1f / MathF.Pow(FastRopeFreqBase, 2f * i / FastHeadDim);
                float angle = p * freq;
                FastRopeCos[p, i] = MathF.Cos(angle);
                FastRopeSin[p, i] = MathF.Sin(angle);
            }
        }

        FastEmbeddings = GetTensor("fast_embeddings.weight");
        FastNormWeight = GetTensor("fast_norm.weight");
        // Q8_0-quantized at load time (once) -- see Q8_0WeightQuantizer's doc comment for why:
        // this is the sub-network this session's performance pass measured as memory-bandwidth-
        // bound on plain float32 weight re-reads (~40ms/call, re-read 9x/frame), and the only
        // quantization level this sub-network was already proven numerically safe at (Q8_0
        // cosine ~0.9995 vs. Q4_K_M's ~0.489, measured earlier this project).
        //
        // Tried and REVERTED: reading the source GGUF's native on-disk dtype directly (mirroring
        // OpenTail.Stingray.Vision's VisionTensorRef/MatVecAny pattern, see NativeGgufWeightRef,
        // still in this codebase for future use). Measured WORSE for this pipeline's actual
        // default checkpoint (`s2-pro-q4_k_m.gguf`, real on-disk dtype Q4_K): 13907ms -> 14459ms,
        // a ~4% regression -- Q4_K's per-element decode (nested sub-block scales, nibble
        // unpacking) is compute-heavier than Q8_0's simple format despite being smaller on disk,
        // and that extra decode cost outweighed the bandwidth saved by skipping the float32 round
        // trip. Always normalizing to Q8_0 at load time (this approach) pays the cheap Q8_0
        // decode cost on every call regardless of the source file's on-disk format, which measured
        // faster for the checkpoint this pipeline actually uses. See docs/audio-review-
        // progress.md's performance-pass entries for the full measurement.
        int qSize = FastHeadCount * FastHeadDim;
        int kvSize = FastHeadCountKv * FastHeadDim;
        FastOutputWeight = Q8_0WeightQuantizer.Quantize(GetTensor("fast_output.weight"), CodebookSize, FastEmbeddingDim);

        FastLayers = new FishSpeechFastLayerWeights[FastBlockCount];
        for (int i = 0; i < FastBlockCount; i++)
        {
            string p = $"fast_layers.{i}";
            int ffnDim = GetTensor($"{p}.feed_forward.w1.weight").Length / FastEmbeddingDim;
            FastLayers[i] = new FishSpeechFastLayerWeights
            {
                AttentionNormWeight = GetTensor($"{p}.attention_norm.weight"),
                WqkvWeight = Q8_0WeightQuantizer.Quantize(GetTensor($"{p}.attention.wqkv.weight"), qSize + 2 * kvSize, FastEmbeddingDim),
                WoWeight = Q8_0WeightQuantizer.Quantize(GetTensor($"{p}.attention.wo.weight"), FastEmbeddingDim, qSize),
                FfnNormWeight = GetTensor($"{p}.ffn_norm.weight"),
                W1Weight = Q8_0WeightQuantizer.Quantize(GetTensor($"{p}.feed_forward.w1.weight"), ffnDim, FastEmbeddingDim),
                W2Weight = Q8_0WeightQuantizer.Quantize(GetTensor($"{p}.feed_forward.w2.weight"), FastEmbeddingDim, ffnDim),
                W3Weight = Q8_0WeightQuantizer.Quantize(GetTensor($"{p}.feed_forward.w3.weight"), ffnDim, FastEmbeddingDim),
                FfnDim = ffnDim,
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

/// <summary>Real fast-AR per-layer weights. The 5 big matrices (Wqkv/Wo/W1/W2/W3) are stored in
/// real Q8_0 block format (see <see cref="Q8_0WeightQuantizer"/>) -- this session's performance
/// pass measured them as the dominant, memory-bandwidth-bound cost at plain float32, and this
/// sub-network was already separately proven numerically safe at Q8_0 precision (unlike Q4_K_M).
/// A native-GGUF-dtype variant (skipping this quantize step, reading the source's own on-disk
/// format directly) was tried and reverted -- measured ~4% SLOWER for this pipeline's actual
/// default checkpoint, since that checkpoint's real on-disk format (Q4_K) has a more expensive
/// per-element decode than Q8_0 despite being smaller, see FishSpeechWeights's loader for the
/// full measurement. The two small RMSNorm weight vectors stay plain float32 (negligible size,
/// not worth quantizing).</summary>
public sealed class FishSpeechFastLayerWeights
{
    public required float[] AttentionNormWeight { get; init; }
    public required byte[] WqkvWeight { get; init; }
    public required byte[] WoWeight { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required byte[] W1Weight { get; init; } // gate
    public required byte[] W2Weight { get; init; } // down
    public required byte[] W3Weight { get; init; } // up
    public required int FfnDim { get; init; }
}
