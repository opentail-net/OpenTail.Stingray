
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Real `facebook/musicgen-small` config, transcribed from the real checkpoint's own
/// `config.json` (fetched 2026-09-02 -- see docs/062-musicgen-implementation-plan.md), NOT
/// re-derived from memory. Three components share one checkpoint file
/// (`model.safetensors`, single-file, ~2.36GB, all F32): a `t5-base` text encoder (loaded from
/// `t5-base`'s OWN checkpoint per HF's real `MusicgenForConditionalGeneration.from_sub_models_pretrained`
/// convention -- the text encoder is NOT included in musicgen-small's own safetensors, only the
/// decoder transformer and the EnCodec audio codec are), a 24-layer OPT-style decoder-only
/// transformer over 4 parallel audio codebooks, and a 32kHz EnCodec neural codec.
/// </summary>
public static class MusicGenConfig
{
    // ── decoder (the MusicGen LM itself) ──────────────────────────────────
    public const int DecoderHiddenSize = 1024;
    public const int DecoderNumLayers = 24;
    public const int DecoderNumHeads = 16;
    public const int DecoderHeadDim = DecoderHiddenSize / DecoderNumHeads; // 64
    public const int DecoderFfnDim = 4096;
    public const int NumCodebooks = 4;
    public const int CodebookSize = 2048; // real audio vocab (0..2047)
    public const int CodebookVocabWithPad = CodebookSize + 1; // 2049: embed_tokens rows, index 2048 = pad/bos
    public const int PadTokenId = 2048;
    public const int BosTokenId = 2048; // real musicgen: bos_token_id == pad_token_id
    public const int MaxPositionEmbeddings = 2048;
    public const float LayerNormEps = 1e-5f; // real MusicgenDecoder uses nn.LayerNorm default eps

    // ── text_encoder (t5-base, loaded from the SEPARATE t5-base checkpoint) ──
    public const int TextDModel = 768;
    public const int TextDFf = 3072;
    public const int TextDKv = 64;
    public const int TextNumLayers = 12;
    public const int TextNumHeads = 12;
    public const int TextRelativeAttentionNumBuckets = 32;
    public const int TextRelativeAttentionMaxDistance = 128;
    public const float TextLayerNormEps = 1e-6f;

    // ── audio_encoder (EnCodec 32kHz) ─────────────────────────────────────
    public const int EncodecHiddenSize = 128; // == codebook_dim
    public const int EncodecCodebookSize = 2048;
    public const int EncodecNumFilters = 64;
    public static readonly int[] EncodecUpsamplingRatios = [8, 5, 4, 4]; // decoder order
    public const int EncodecLstmLayers = 2;
    public const int EncodecCompress = 2; // residual-block bottleneck divisor
    public const int EncodecKernelSize = 7;
    public const int EncodecLastKernelSize = 7;
    public const int EncodecResidualKernelSize = 3;
    public const int SampleRate = 32000;
    public const int FrameRate = 50; // real EnCodec 32kHz: 32000 / (8*5*4*4) = 50 frames/sec

    // ── generation defaults (real generation_config.json) ─────────────────
    public const float DefaultGuidanceScale = 3.0f;
    public const int DefaultTopK = 250;
    public const float DefaultTemperature = 1.0f;
}
