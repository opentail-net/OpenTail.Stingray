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

    /// <summary>Real prefill row: either a TEXT token, or an AUDIO patch (real continuous
    /// `[PatchSize][FeatDim]` latent features from <see cref="VoxCpm2AudioVaeDecoder.Encode"/>,
    /// chunked into <see cref="PatchSize"/>-frame patches -- `AudioFeature` is the raw patch used
    /// for `prefix_cond` seeding, `AudioEmbedding` is that SAME patch already projected through
    /// <see cref="VoxCpm2LocalEncoder.EncodePatch"/>, the real `current_embeddings` row value).</summary>
    public readonly struct PrefillRow
    {
        public int? TextToken { get; init; }
        public float[][]? AudioFeature { get; init; }
        public float[]? AudioEmbedding { get; init; }
        public bool IsAudio => AudioFeature != null;

        public static PrefillRow Text(int token) => new() { TextToken = token };
        public static PrefillRow Audio(float[][] feature, float[] embedding) => new() { AudioFeature = feature, AudioEmbedding = embedding };
    }

    /// <summary>
    /// Real, general reference-audio-AWARE prefill + `generate_once` loop, ported from
    /// `VoxCPM2PromptPrefillRuntime::Impl::build`'s full real graph (not the text-only
    /// simplification <see cref="Generate"/> uses) -- supports real voice-cloning/reference-audio
    /// conditioning via mixed TEXT/AUDIO prefill rows (build the row sequence with
    /// <see cref="PrefillRow.Text"/>/<see cref="PrefillRow.Audio"/>, matching the real reference's
    /// `build_prefill_sequence` order: optional `[refAudioStart, ...reference patches...,
    /// refAudioEnd]`, then the target text tokens, then `audioStartTokenId`, then optional
    /// `[...prompt patches...]`).
    ///
    /// <para>Real per-row formula (traced from the reference's actual graph, not approximated):
    /// at every row, `base_hidden = base_lm(row's embedding)`; `fsq = FSQ(base_hidden)` (the SAME
    /// bottleneck <see cref="VoxCpm2StepProjection.Run"/> already computes, reused here via its
    /// `FsqHidden` output on a throwaway call); `lm_hidden = row.IsAudio ? fsq : base_hidden`
    /// (TEXT rows keep the raw hidden state, AUDIO rows are FSQ-quantized); `masked_current =
    /// row.IsAudio ? row.AudioEmbedding : zero`; `residual_input = fusion_concat_proj(concat(
    /// lm_hidden, masked_current))`; `residual_hidden = residual_lm.Step(residual_input)`. The
    /// loop's seed `lm_hidden`/`residual_hidden` are the LAST row's values; the initial CFM
    /// `prefix_cond` is the LAST row's real feature (`zero_patch` for a text row, the real raw
    /// audio patch for an audio row -- matches the reference's own unconditional
    /// `prefix_cond = row.feature` assignment on every row).</para>
    /// </summary>
    public static Result GenerateWithPrompt(
        IForwardPass baseLmFwd,
        VoxCpm2ResidualLm residualLm,
        VoxCpm2StepProjectionWeights projWeights,
        VoxCpm2DiTEstimatorWeights ditWeights,
        VoxCpm2LocalEncoderWeights encoderWeights,
        IReadOnlyList<PrefillRow> rows,
        int maxPatches,
        int cfmTimesteps,
        float cfgValue,
        int minTokens,
        Random noiseRng)
    {
        if (rows.Count == 0) throw new ArgumentException("Prefill rows must be non-empty.", nameof(rows));

        var zeroHidden = new float[HiddenDim];
        var zeroPatch = new float[PatchSize][];
        for (int p = 0; p < PatchSize; p++) zeroPatch[p] = new float[FeatDim];
        int latentDim = projWeights.FsqInProjBias.Length;

        var lmHidden = zeroHidden;
        var residualHidden = zeroHidden;
        var prefixCond = zeroPatch;

        for (int i = 0; i < rows.Count; i++)
        {
            var row = rows[i];
            float[] baseHidden;
            if (row.IsAudio)
            {
                baseLmFwd.ForwardEmbedding(row.AudioEmbedding!, i);
                baseHidden = baseLmFwd.LastHidden.ToArray();
            }
            else
            {
                baseLmFwd.Prefill([row.TextToken!.Value], startPos: i);
                baseHidden = baseLmFwd.LastHidden.ToArray();
            }

            var fsq = VoxCpm2StepProjection.Run(projWeights, baseHidden, zeroHidden, zeroHidden,
                HiddenDim, DitHiddenDim, latentDim, scalarQuantizationScale: 1000).FsqHidden;
            lmHidden = row.IsAudio ? fsq : baseHidden;

            var maskedCurrent = row.IsAudio ? row.AudioEmbedding! : zeroHidden;
            var concat = new float[HiddenDim * 2];
            Array.Copy(lmHidden, concat, HiddenDim);
            Array.Copy(maskedCurrent, 0, concat, HiddenDim, HiddenDim);
            var residualInput = Linear(concat, projWeights.FusionConcatProjWeight, projWeights.FusionConcatProjBias, HiddenDim * 2, HiddenDim);
            residualHidden = residualLm.Step(residualInput);

            prefixCond = row.IsAudio ? row.AudioFeature! : zeroPatch;
        }

        var patches = new List<float[][]>();
        bool stoppedEarly = false;
        int position = rows.Count;
        for (int index = 0; index < maxPatches; index++)
        {
            var projected = VoxCpm2StepProjection.Run(projWeights, lmHidden, residualHidden, zeroHidden,
                HiddenDim, DitHiddenDim, latentDim, scalarQuantizationScale: 1000);
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
            baseLmFwd.ForwardEmbedding(currEmbed, position++);
            var nextLm = baseLmFwd.LastHidden.ToArray();
            var nextProjected = VoxCpm2StepProjection.Run(projWeights, nextLm, residualHidden, currEmbed,
                HiddenDim, DitHiddenDim, latentDim, scalarQuantizationScale: 1000);
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
