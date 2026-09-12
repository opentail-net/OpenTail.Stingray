
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real weight loader for Voxtral Realtime's text decoder (`language_model.model.*` in
/// `model.safetensors`), transcribed directly from
/// `examples/audio.cpp/src/models/voxtral_realtime/text_decoder.cpp`'s `FullContextGraph` (not
/// guessed). A real Mistral GQA decoder (26 layers, 32/8 heads, head_dim=128, hidden=3072,
/// RoPE theta=1e6) PLUS a real AdaLN-Zero-style delay-conditioning gate per layer
/// (`ada_rms_norm.linear{1,2}.weight`, no bias, confirmed via a real tensor dump) that this
/// checkpoint's config alone does not reveal -- see `docs/audio-review-progress.md`'s Voxtral
/// section for the correction this represents (an earlier pass in this session wrongly assumed
/// the decoder needed zero new code). Tied embeddings (`config.json`'s
/// `text_config.tie_word_embeddings=true`, confirmed by the real tensor list having no separate
/// `lm_head.weight`).
///
/// <para><b>Perf-sweep Phase 1.2</b> (docs/perf-sweep-plan.md): every matvec weight matrix is
/// quantized to Q8_0 at load time (verified converter, see
/// `ConvertF32ToQ8_0VerificationTests`) so <see cref="VoxtralTextDecoder"/>'s linear layers hit
/// <c>SimdKernels.MatVecQ8_0</c> instead of a full-F32 matvec -- 4x less weight-matrix memory
/// traffic per call. <see cref="EmbedTokensWeight"/> keeps its original F32 form too (needed for
/// per-token embedding-row lookup, which is a direct index, not a matvec) alongside a separate
/// quantized copy for its tied-lm_head use.</para>
/// </summary>
public sealed class VoxtralTextDecoderWeights
{
    public const int HiddenSize = 3072;
    public const int IntermediateSize = 9216;
    public const int NumLayers = 26;
    public const int NumHeads = 32;
    public const int NumKvHeads = 8;
    public const int HeadDim = 128;
    public const float RmsNormEps = 1e-5f;
    public const float RopeTheta = 1_000_000f;
    public const int VocabSize = 131072;
    public const int AdaHiddenDim = 32;

    public float[] EmbedTokensWeight { get; } // [131072, 3072] -- raw F32, for embedding-row lookup
    public byte[] EmbedTokensWeightQ8_0 { get; } // same tensor, Q8_0-quantized, for the tied lm_head matvec
    public VoxtralTextLayerWeights[] Layers { get; } = new VoxtralTextLayerWeights[NumLayers];
    public float[] NormWeight { get; }

    public VoxtralTextDecoderWeights(SafetensorsLoader loader)
    {
        EmbedTokensWeight = loader.ReadF32("language_model.model.embed_tokens.weight");
        EmbedTokensWeightQ8_0 = QuantizeQ8_0(EmbedTokensWeight, HiddenSize);
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"language_model.model.layers.{i}";
            Layers[i] = new VoxtralTextLayerWeights
            {
                InputNorm = loader.ReadF32($"{p}.input_layernorm.weight"),
                QWeight = QuantizeQ8_0(loader.ReadF32($"{p}.self_attn.q_proj.weight"), HiddenSize),
                KWeight = QuantizeQ8_0(loader.ReadF32($"{p}.self_attn.k_proj.weight"), HiddenSize),
                VWeight = QuantizeQ8_0(loader.ReadF32($"{p}.self_attn.v_proj.weight"), HiddenSize),
                OWeight = QuantizeQ8_0(loader.ReadF32($"{p}.self_attn.o_proj.weight"), NumHeads * HeadDim),
                PostNorm = loader.ReadF32($"{p}.post_attention_layernorm.weight"),
                GateWeight = QuantizeQ8_0(loader.ReadF32($"{p}.mlp.gate_proj.weight"), HiddenSize),
                UpWeight = QuantizeQ8_0(loader.ReadF32($"{p}.mlp.up_proj.weight"), HiddenSize),
                DownWeight = QuantizeQ8_0(loader.ReadF32($"{p}.mlp.down_proj.weight"), IntermediateSize),
                Ada1Weight = QuantizeQ8_0(loader.ReadF32($"{p}.ada_rms_norm.linear1.weight"), HiddenSize),
                Ada2Weight = QuantizeQ8_0(loader.ReadF32($"{p}.ada_rms_norm.linear2.weight"), AdaHiddenDim),
            };
        }
        NormWeight = loader.ReadF32("language_model.model.norm.weight");
    }

    /// <summary>Quantizes a row-major <c>[rows, cols]</c> F32 matrix to Q8_0 (34 bytes/32-element
    /// block per row), using the perf-sweep-verified <see
    /// cref="OpenTail.Stingray.Core.FastVectorTypeConverter.ConvertF32ToQ8_0"/>.</summary>
    internal static byte[] QuantizeQ8_0(float[] src, int cols)
    {
        int rows = src.Length / cols;
        int bytesPerRow = (cols / 32) * 34;
        var dst = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            OpenTail.Stingray.Core.FastVectorTypeConverter.ConvertF32ToQ8_0(
                src.AsSpan(r * cols, cols), dst.AsSpan(r * bytesPerRow, bytesPerRow));
        return dst;
    }
}

public sealed class VoxtralTextLayerWeights
{
    public float[] InputNorm { get; set; } = [];
    public byte[] QWeight { get; set; } = [];
    public byte[] KWeight { get; set; } = [];
    public byte[] VWeight { get; set; } = [];
    public byte[] OWeight { get; set; } = [];
    public float[] PostNorm { get; set; } = [];
    public byte[] GateWeight { get; set; } = [];
    public byte[] UpWeight { get; set; } = [];
    public byte[] DownWeight { get; set; } = [];
    public byte[] Ada1Weight { get; set; } = []; // [32, 3072], no bias
    public byte[] Ada2Weight { get; set; } = []; // [3072, 32], no bias
}
