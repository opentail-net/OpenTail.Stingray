
namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Real `ACE-Step/Ace-Step1.5` config, read directly from the checkpoint's own real
/// `acestep-v15-turbo/config.json`, `vae/config.json`, and `Qwen3-Embedding-0.6B/config.json`
/// (2026-09-03 -- see docs/064-acestep-implementation-plan.md for the full derivation, including
/// the real `modeling_acestep_v15_turbo.py`/`configuration_acestep_v15.py` reference source
/// bundled in that HF repo as `custom_code`, read directly rather than guessed).
///
/// <para><b>Status: config-only, no forward pass implemented yet.</b> This is Phase A
/// (reconnaissance) output per the implementation plan -- the DiT/condition-encoder/VAE/scheduler
/// classes referenced by the plan's suggested layout do not exist yet. See the plan doc's
/// "Immediate next steps" for the real work still ahead (weight download, Snake1d formula
/// verification, causal-vs-bidirectional Qwen3 usage verification, then the actual port).</para>
/// </summary>
public static class AceStepConfig
{
    // ── DiT (acestep-v15-turbo/config.json) ───────────────────────────────
    public const int HiddenSize = 2048;
    public const int HeadDim = 128;
    public const int NumAttentionHeads = 16;
    public const int NumKeyValueHeads = 8; // GQA, 2:1 ratio
    public const int IntermediateSize = 6144;
    public const int NumHiddenLayers = 24;
    public const int InChannels = 192; // = audio_acoustic_hidden_dim(64) * 3 (noisy latent + src_latents + chunk_masks)
    public const int AudioAcousticHiddenDim = 64;
    public const int PatchSize = 2;
    public const float RmsNormEps = 1e-6f;
    public const float RopeTheta = 1_000_000f;
    public const int SlidingWindow = 128;
    public const bool UseSlidingWindow = true;
    public const int TextHiddenDim = 1024; // == Qwen3-Embedding-0.6B's own hidden_size
    public const int NumLyricEncoderHiddenLayers = 8;
    public const int NumTimbreEncoderHiddenLayers = 4;
    public const int NumAttentionPoolerHiddenLayers = 2;
    public const int TimbreHiddenDim = 64;
    public const int PoolWindowSize = 5;
    public const int FsqDim = 2048;
    public static readonly int[] FsqInputLevels = [8, 8, 8, 5, 5, 5];
    public const int FsqInputNumQuantizers = 1;
    public const float TimestepMu = -0.4f;
    public const float TimestepSigma = 1.0f;
    public const bool IsTurbo = true;

    /// <summary>Real `layer_types` from the checkpoint: starts sliding, alternates every layer. Index 0 = sliding, 1 = full, 2 = sliding, ... (24 layers total, matching <see cref="NumHiddenLayers"/>).</summary>
    public static bool IsSlidingLayer(int layerIndex) => layerIndex % 2 == 0;

    // ── Qwen3-Embedding-0.6B (standalone checkpoint, real config.json) ────
    public const int TextEncoderHiddenSize = 1024;
    public const int TextEncoderNumLayers = 28;
    public const int TextEncoderNumAttentionHeads = 16;
    public const int TextEncoderNumKeyValueHeads = 8;
    public const int TextEncoderHeadDim = 128;
    public const int TextEncoderIntermediateSize = 3072;
    public const float TextEncoderRmsNormEps = 1e-6f;
    public const float TextEncoderRopeTheta = 1_000_000f;

    // ── VAE (vae/config.json, real AutoencoderOobleck) ────────────────────
    public const int VaeSampleRate = 48_000;
    public const int VaeAudioChannels = 2; // stereo
    public const int VaeDecoderChannels = 128;
    public const int VaeDecoderInputChannels = 64;
    public static readonly int[] VaeChannelMultiples = [1, 2, 4, 8, 16];
    public static readonly int[] VaeDownsamplingRatios = [2, 4, 4, 6, 10]; // product = 1920

    // ── Turbo inference defaults (real generate_audio) ─────────────────────
    public const int DefaultInferenceSteps = 8; // "fix_nfe"
    public const float DefaultShift = 3.0f;
    public const string DefaultInferMethod = "ode";

    /// <summary>Real hardcoded 8-step Euler-ODE timestep schedules, transcribed verbatim from `AceStepConditionGenerationModel.generate_audio`'s `SHIFT_TIMESTEPS` table. Only shift values 1, 2, or 3 are real/supported -- any other requested shift snaps to the nearest of these three in the real reference, do not interpolate a new schedule for other shifts.</summary>
    public static readonly IReadOnlyDictionary<float, float[]> ShiftTimestepSchedules = new Dictionary<float, float[]>
    {
        [1.0f] = [1.0f, 0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f],
        [2.0f] = [1.0f, 0.9333333333333333f, 0.8571428571428571f, 0.7692307692307693f, 0.6666666666666666f, 0.5454545454545454f, 0.4f, 0.2222222222222222f],
        [3.0f] = [1.0f, 0.9545454545454546f, 0.9f, 0.8333333333333334f, 0.75f, 0.6428571428571429f, 0.5f, 0.3f],
    };
}
