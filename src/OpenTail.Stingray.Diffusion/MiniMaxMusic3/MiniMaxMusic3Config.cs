namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 config, read directly from the real checkpoint's own per-component
/// `config.json` files (`MiniMaxAI/MiniMax-Music3` on Hugging Face) and the real, already-installed
/// `diffusers==0.40.0` source -- see docs/066-minimax-music3-future-plan.md for the full
/// archaeology (Phase A). Only the real components the actual inference pipeline uses are modeled
/// here (`condition_encoder`, `language_model`, `rvq_depth_decoder`, `transformer`, `vocoder`,
/// `scheduler`) -- the checkpoint repo bundles several large additional files
/// (`qwen_7B/`, `flowmatching_vae.pth`, `dav.pth`, ~29GB combined) that the real
/// `modular_model_index.json` component graph never references; not modeled here.
/// </summary>
public static class MiniMaxMusic3Config
{
    // ── Vocoder (`vocoder/config.json`, real `MiniMaxMusic3Vocoder`) ──────
    public const int VocoderLatentChannels = 128;
    public const int VocoderDecoderInputDim = 1024;
    public const int VocoderDecoderHiddenDim = 1536;
    public static readonly int[] VocoderUpsamplingRatios = [8, 8, 4, 2]; // hop = 512
    public const int VocoderSampleRate = 44100;

    // ── RVQ depth decoder (`rvq_depth_decoder/config.json`) ───────────────
    public const int RvqDepthDecoderHiddenSize = 4096;
    public const int RvqDepthDecoderIntermediateSize = 6144;
    public const int RvqDepthDecoderNumHeads = 16;
    public const int RvqDepthDecoderNumLayers = 4;
    public const int RvqDepthDecoderNumCodebooks = 8;
    public const int RvqDepthDecoderAudioVocabSize = 1024;
    public const int RvqDepthDecoderMaxPositionEmbeddings = 16;

    // ── Condition encoder (`condition_encoder/config.json`) ───────────────
    public const int ConditionEncoderHiddenDim = 4096;
    public const int ConditionEncoderOutDim = 2048;
    public const int ConditionEncoderNumLayers = 8;
    public const int ConditionEncoderInputHopLength = 960;
    public const int ConditionEncoderInputSampleRate = 24000;
    public const int ConditionEncoderOutputHopLength = 512;
    public const int ConditionEncoderOutputSampleRate = 44100;

    // ── Transformer (`transformer/config.json`, the flow-matching DiT) ────
    public const int TransformerConditionDim = 2048;
    public const int TransformerAttentionHeadDim = 64;
    public const int TransformerNumAttentionHeads = 32;
    public const int TransformerNumLayers = 36;
    public const int TransformerInChannels = 128;
    public const int TransformerFourierEmbeddingDim = 256;
    public const int TransformerFfInnerDim = 8192;
    public const int TransformerRotaryDim = 32;

    // ── Language model (`language_model/config.json`, real STOCK Qwen3ForCausalLM) ─
    public const int LanguageModelHiddenSize = 4096;
    public const int LanguageModelIntermediateSize = 12288;
    public const int LanguageModelNumLayers = 36;
    public const int LanguageModelNumAttentionHeads = 32;
    public const int LanguageModelNumKeyValueHeads = 8;
    public const int LanguageModelHeadDim = 128;
    public const float LanguageModelRmsNormEps = 1e-6f;
    public const float LanguageModelRopeTheta = 1_000_000f;

    // ── Real chunked long-form generation (confirmed from `MiniMaxMusic3CoreDenoiseStep`) ──
    public const int ChunkFrames = 200;
}
