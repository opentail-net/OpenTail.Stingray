namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Real per-frame feature-generation loop wiring EVERY VoxCPM2 piece real-weight verified this
/// session (`base_lm`, `residual_lm`, <see cref="VoxCpm2StepProjection"/>,
/// <see cref="VoxCpm2CfmSolver"/>, <see cref="VoxCpm2LocalEncoder"/>) into the real algorithm
/// from `generator.cpp`'s `Impl::generate_once` (not guessed) -- a reference-audio-free
/// (`generate_zero_shot`-shaped) subset: no prompt/reference audio rows, only plain text prefill.
///
/// <para><b>Real prefill, reference-audio-free case</b> (`build_prefill_sequence` with
/// `prompt == nullptr`): every prefill row is a TEXT row (`text_mask=1`, `audio_mask=0`,
/// `feature=zero_patch`). Tracing `VoxCPM2PromptPrefillRuntime::Impl::build`'s real graph under
/// that condition (not guessed -- `audio_mask` zeroing the FSQ/current-embedding branches at
/// every position is exact, not an approximation): `masked_fsq` and `masked_current` are
/// identically zero at every step, so `lm_hidden_t = base_hidden_t` (raw post-final-norm
/// `base_lm` hidden, no FSQ contribution) and `residual_input_t = fusion_concat_proj(concat(
/// base_hidden_t, zeros))` at every step -- exactly reproducible by running `base_lm` one text
/// token at a time (`IForwardPass.Prefill([token], startPos: i)`, capturing `LastHidden` each
/// step) and feeding each step's `[baseHidden_t, zero_2048]` concatenation through the real
/// `fusion_concat_proj` into `residual_lm.Step`, WITHOUT needing a from-scratch port of
/// `VoxCPM2PromptPrefillRuntime`'s own ggml graph. The seed `lm_hidden`/`residual_hidden` for
/// `generate_once`'s loop are simply `base_hidden`/`residual_lm.Step`'s output at the LAST text
/// position (real `SliceModule({1, steps-1, 1})`), and the initial CFM `prefix_cond` is the real
/// all-zero patch (every prefill row's `feature` is `zero_patch` in this reference-audio-free
/// case, so `prefix_cond`'s last real assignment before the loop is unconditionally zero).</para>
///
/// <para><b>Real per-iteration algorithm</b> (`generate_once`, lines ~1568-1607, not guessed):
/// each iteration's FIRST <see cref="VoxCpm2StepProjection.Run"/> call uses a CONSTANT ZERO
/// vector for `currentEmbed` (`zero_hidden` in the reference) -- the real just-generated patch
/// embedding is only threaded into the SECOND `Run` call at the bottom of the SAME iteration,
/// whose `FsqHidden`/`ResidualInput` outputs seed the NEXT iteration's `lm_hidden`/
/// `residual_hidden`. Getting this backwards would silently produce wrong DiT conditioning.</para>
/// </summary>
public static class VoxCpm2Generator
{
    public const int HiddenDim = 2048;
    public const int DitHiddenDim = VoxCpm2MiniCpmBidirectionalStack.HiddenDim; // 1024
    public const int PatchSize = VoxCpm2DiTEstimator.PatchSize;
    public const int FeatDim = VoxCpm2DiTEstimator.FeatDim;

    public readonly struct Result(float[][][] patches, bool stoppedEarly)
    {
        /// <summary>Real per-frame `[PatchSize][FeatDim]`-shaped patches (real
        /// `generated_features`, patch-major -- concatenate/flatten before feeding the AudioVAE
        /// decoder, which expects channel-major `[LatentDim * frames]`).</summary>
        public float[][][] Patches { get; } = patches;
        public bool StoppedEarly { get; } = stoppedEarly;
    }

    /// <summary>
    /// Runs the real text-prefill + per-frame `generate_once` loop. `baseLmFwd` must be a fresh
    /// (just-constructed, empty-KV-cache) <see cref="IForwardPass"/> over VoxCPM2's `base_lm`;
    /// `residualLm` must be freshly `Reset()`.
    /// </summary>
    public static Result Generate(
        IForwardPass baseLmFwd,
        VoxCpm2ResidualLm residualLm,
        VoxCpm2StepProjectionWeights projWeights,
        VoxCpm2DiTEstimatorWeights ditWeights,
        VoxCpm2LocalEncoderWeights encoderWeights,
        int[] promptTokenIds,
        int maxPatches,
        int cfmTimesteps,
        float cfgValue,
        int minTokens,
        Random noiseRng)
    {
        if (promptTokenIds.Length == 0) throw new ArgumentException("Prompt must be non-empty.", nameof(promptTokenIds));

        var zeroHidden = new float[HiddenDim];
        float[] lastBaseHidden = zeroHidden;
        float[] lastResidualHidden = zeroHidden;
        for (int i = 0; i < promptTokenIds.Length; i++)
        {
            baseLmFwd.Prefill([promptTokenIds[i]], startPos: i);
            lastBaseHidden = baseLmFwd.LastHidden.ToArray();
            var concat = new float[HiddenDim * 2];
            Array.Copy(lastBaseHidden, concat, HiddenDim);
            var residualInput = Linear(concat, projWeights.FusionConcatProjWeight, projWeights.FusionConcatProjBias, HiddenDim * 2, HiddenDim);
            lastResidualHidden = residualLm.Step(residualInput);
        }

        var lmHidden = lastBaseHidden;
        var residualHidden = lastResidualHidden;
        var prefixCond = new float[PatchSize][];
        for (int p = 0; p < PatchSize; p++) prefixCond[p] = new float[FeatDim];

        var patches = new List<float[][]>();
        bool stoppedEarly = false;
        int textPosition = promptTokenIds.Length;
        for (int index = 0; index < maxPatches; index++)
        {
            var projected = VoxCpm2StepProjection.Run(projWeights, lmHidden, residualHidden, zeroHidden,
                HiddenDim, DitHiddenDim, latentDim: projWeights.FsqInProjBias.Length, scalarQuantizationScale: 1000);
            var mu = new float[2][] { projected.CurrentLmDitHidden, projected.ResidualDitHidden };

            var patch = VoxCpm2CfmSolver.GeneratePatch(ditWeights, mu, prefixCond, cfmTimesteps, cfgValue, meanMode: false, noiseRng);
            patches.Add(patch);
            prefixCond = patch;

            if (index > minTokens && StopClass(projected.CurrentStopLogits) == 1)
            {
                stoppedEarly = true;
                break;
            }

            var currEmbed = VoxCpm2LocalEncoder.EncodePatch(encoderWeights, patch, HiddenDim);
            baseLmFwd.ForwardEmbedding(currEmbed, textPosition++);
            var nextLm = baseLmFwd.LastHidden.ToArray();
            var nextProjected = VoxCpm2StepProjection.Run(projWeights, nextLm, residualHidden, currEmbed,
                HiddenDim, DitHiddenDim, latentDim: projWeights.FsqInProjBias.Length, scalarQuantizationScale: 1000);
            lmHidden = nextProjected.FsqHidden;
            residualHidden = residualLm.Step(nextProjected.ResidualInput);
        }

        return new Result([.. patches], stoppedEarly);
    }

    /// <summary>Real `stop_class`: argmax of the real 2-logit stop head.</summary>
    private static int StopClass(float[] stopLogits) => stopLogits[1] > stopLogits[0] ? 1 : 0;

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }
}
