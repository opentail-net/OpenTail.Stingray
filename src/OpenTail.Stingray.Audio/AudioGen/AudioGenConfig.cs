
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Real `facebook/audiogen-medium` config, extracted directly from the real checkpoint's own
/// embedded training config (`xp.cfg`, an OmegaConf-serialized string baked into the real
/// AudioCraft `state_dict.bin`/`compression_state_dict.bin` "Solver" checkpoints -- see
/// docs/063-audiogen-implementation-plan.md for the full derivation), NOT re-derived from memory
/// or from MusicGen's numbers. AudioGen follows the exact same LM architecture MusicGen
/// introduced (delayed multi-codebook pattern over a discrete EnCodec-token autoregressive
/// Transformer) -- see <see cref="AudioGenConfig"/> vs `OpenTail.Stingray.Audio.MusicGen.MusicGenConfig`
/// for what's genuinely shared (delay pattern, codebook count/size, CFG formula) vs what
/// differs (dims, T5 variant, EnCodec ratios, checkpoint format/tensor-naming convention: native
/// AudioCraft, not HF `transformers`).
/// </summary>
public static class AudioGenConfig
{
    // ── LM (transformer_lm config block) ──────────────────────────────────
    public const int HiddenSize = 1536;
    public const int NumLayers = 48;
    public const int NumHeads = 24;
    public const int HeadDim = HiddenSize / NumHeads; // 64
    public const int FfnDim = HiddenSize * 4; // hidden_scale=4 -> 6144
    public const int NumCodebooks = 4;
    public const int CodebookSize = 2048; // "card"
    public const int CodebookVocabWithPad = CodebookSize + 1; // 2049: emb.{q} rows, index 2048 = special/pad/bos token
    public const int PadTokenId = CodebookSize; // real `special_token_id` property == card == 2048
    public const int BosTokenId = CodebookSize; // same value; AudioCraft has no separate BOS, the special token doubles as both
    public const float SinusoidalMaxPeriod = 10000f;

    // ── text conditioner (t5-large, external/frozen -- NOT bundled in the LM checkpoint) ──
    public const int TextDModel = 1024;
    public const int TextDFf = 4096;
    public const int TextDKv = 64;
    public const int TextNumLayers = 24;
    public const int TextNumHeads = 16;
    public const int TextRelativeAttentionNumBuckets = 32;
    public const int TextRelativeAttentionMaxDistance = 128;
    public const float TextLayerNormEps = 1e-6f;

    // ── audio_encoder (EnCodec 16kHz, separately-trained "compression" checkpoint) ─────
    public const int EncodecHiddenSize = 128; // == codebook_dim, real `seanet.dimension`
    public const int EncodecCodebookSize = 2048; // real `rvq.bins`
    public const int EncodecNumFilters = 64; // real `seanet.n_filters`
    public static readonly int[] EncodecUpsamplingRatios = [8, 5, 4, 2]; // decoder order, real `seanet.ratios` -- differs from MusicGen's [8,5,4,4]
    public const int SampleRate = 16000;
    public const int FrameRate = 50; // real: 16000 / (8*5*4*2) = 50 Hz -- same frame rate as MusicGen despite the different sample rate

    // ── generation defaults (real `classifier_free_guidance`/`generate.lm` config blocks) ──
    public const float DefaultGuidanceScale = 3.0f; // real `classifier_free_guidance.inference_coef`
    public const int DefaultTopK = 250;
    public const float DefaultTemperature = 1.0f;
}
