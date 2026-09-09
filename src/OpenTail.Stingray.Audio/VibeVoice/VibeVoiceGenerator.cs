using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real special-token selection for VibeVoice TTS's interleaved text/diffusion decode
/// loop, ported from `generator.cpp`'s `select_vibevoice_constrained_token` (not guessed): only
/// EVER considers 4 candidate tokens (`speechStart`/`speechEnd`/`speechDiffusion`/`eos`) --
/// argmax among them when not sampling, else real temperature/top-k/top-p multinomial sampling
/// restricted to just those 4 logits.</summary>
public static class VibeVoiceGenerationTokenSelector
{
    public static int SelectArgmax(ReadOnlySpan<float> logits, int speechStartId, int speechEndId, int speechDiffusionId, int eosId)
    {
        Span<int> candidates = [speechStartId, speechEndId, speechDiffusionId, eosId];
        int best = candidates[0];
        float bestScore = float.NegativeInfinity;
        foreach (int token in candidates)
        {
            float score = logits[token];
            if (score > bestScore) { bestScore = score; best = token; }
        }
        return best;
    }

    /// <summary>Real `options.do_sample` path from `select_vibevoice_constrained_token`: argmax
    /// among just the 4 real candidate tokens when `options` is `null` (matches
    /// <see cref="SelectArgmax"/> exactly), otherwise real temperature/top-k/top-p sampling
    /// restricted to those same 4 candidates -- implemented by masking every OTHER vocabulary
    /// entry to `-infinity` before delegating to `OpenTail.Stingray.Engine.Sampler.Sample` (top-k/
    /// top-p naturally exclude `-infinity` entries, so this is equivalent to the reference's own
    /// restricted-candidate scoring without a bespoke small-N sampler).</summary>
    public static int Select(ReadOnlySpan<float> logits, int speechStartId, int speechEndId, int speechDiffusionId, int eosId,
        SamplingParams? options, Random? rng = null)
    {
        if (options is null) return SelectArgmax(logits, speechStartId, speechEndId, speechDiffusionId, eosId);

        Span<int> candidates = [speechStartId, speechEndId, speechDiffusionId, eosId];
        var masked = new float[logits.Length];
        Array.Fill(masked, float.NegativeInfinity);
        foreach (int token in candidates) masked[token] = logits[token];
        return Sampler.Sample(masked, options, rng);
    }
}

/// <summary>
/// Real interleaved text/diffusion generation loop for VibeVoice TTS, ported from
/// `generator.cpp`'s `generate_vibevoice` (not guessed): the LLM autoregressively emits ONE OF
/// 4 real control tokens per step (`speech_start`/`speech_end`/`speech_diffusion`/`eos` -- real,
/// non-obvious detail: these REUSE the base Qwen2 checkpoint's own vision special tokens
/// `&lt;|vision_start|&gt;`/`&lt;|vision_end|&gt;`/`&lt;|vision_pad|&gt;`/`&lt;|endoftext|&gt;`,
/// confirmed from `tokenizer_text.cpp`, not new tokens). On a `speech_diffusion` step: sample
/// one acoustic-latent frame via the real CFG diffusion sampler (conditioned on BOTH the
/// positive prompt's hidden state and a negative branch seeded from just the `speech_start`
/// token), unscale it (`latent/speechScalingFactor - speechBiasFactor`, real per-checkpoint
/// scalars), decode it to a waveform chunk, re-encode that chunk through the semantic tokenizer,
/// project both the acoustic latent and the semantic features through their own real connectors,
/// SUM the two projections into the next step's embedding. On any other step: embed the emitted
/// token normally via ordinary vocabulary lookup.
///
/// <para><b>Update, 2026-09-07</b>: the real STATEFUL streaming decode/encode
/// (`decode_acoustic_streaming`/`encode_semantic_streaming`, real per-layer conv-history caching
/// across chunks) is now implemented -- see <see cref="VibeVoiceTokenizerStreamingState"/> and
/// `VibeVoiceTokenizerDecoder.DecodeStreaming`/`VibeVoiceTokenizerEncoder.EncodeStreaming`. This
/// method constructs ONE persistent streaming state per real generation call and reuses it across
/// every diffusion chunk, matching the reference's real chunk-boundary continuity -- real,
/// confirmed-via-listening improvement over the earlier one-shot-per-chunk simplification (which
/// produced structurally valid but audibly gibberish-sounding output once a sample spanned more
/// than a couple of chunks).</para>
///
/// <para><b>Bug fix, 2026-09-08</b>: the CFG negative/unconditional branch now runs on its own
/// separate <see cref="IForwardPass"/> instance (<c>negativeFwd</c>), not the same instance as
/// the positive branch. Cross-checking against `generator.cpp` found the negative branch is
/// supposed to run on a fully independent decoder cache (`negative_cache`) seeded ONLY with the
/// `speech_start` token -- it never processes the text prompt, and it is reset from scratch
/// every time `speech_start` is re-emitted. The prior single-`fwd`, shared-cache version had two
/// compounding bugs: (1) the "negative" branch's `speech_start` embedding was written into the
/// SAME cache as the positive branch, right after the full prompt, so it wasn't actually
/// unconditional -- it attended over the whole prompt; (2) its position counter didn't advance
/// on every `speech_diffusion` step, so successive diffusion steps wrote into the SAME cache slot
/// as each other, and collided with the positive branch's own advancing position counter,
/// corrupting both branches' KV history. This is a strong candidate for the root cause previously
/// attributed to generic "compounding floating-point drift" in <see cref="VibeVoiceDiffusionSampler"/>'s
/// 2026-09-08 bisection note -- a real semantic/cache bug, not FP drift, would also produce
/// "close at frame 0, diverging by frame 5+" (small early perturbation compounding through the
/// closed generation loop). Callers MUST pass a genuinely separate <see cref="IForwardPass"/>
/// instance for <c>negativeFwd</c> (e.g. a second <c>new ForwardPass(sameModel, sameBackend, sameHp)</c>)
/// -- <see cref="IForwardPass.CreateContext"/> is NOT usable for this today; every real backend's
/// override is the interface's `=> this` default (unimplemented per
/// `docs/010-...Forward-Pass Context Isolation...md`), so calling it would silently alias back to
/// the same shared cache this fix removes.</para>
/// </summary>
public static class VibeVoiceGenerator
{
    public readonly struct Result(float[] audioSamples, int[] generatedTokens)
    {
        public float[] AudioSamples { get; } = audioSamples;
        public int[] GeneratedTokens { get; } = generatedTokens;
    }

    public static Result Generate(
        IForwardPass fwd,
        IForwardPass negativeFwd,
        int[] promptTokenIds,
        float[] textEmbeddingTable, // [vocabSize, hiddenDim] row-major, real model.language_model.embed_tokens.weight
        int hiddenDim,
        int speechStartId, int speechEndId, int speechDiffusionId, int eosId,
        VibeVoiceDiffusionHeadWeights diffusionHeadWeights,
        VibeVoiceTokenizerDecoderWeights acousticDecoderWeights,
        VibeVoiceTokenizerEncoderWeights semanticEncoderWeights,
        VibeVoiceConnectorWeights acousticConnectorWeights,
        VibeVoiceConnectorWeights semanticConnectorWeights,
        float speechScalingFactor, float speechBiasFactor,
        float layerNormEps,
        int ddpmNumSteps, int inferenceSteps, float guidanceScale,
        int maxSteps, Random rng, SamplingParams? tokenSelectionOptions = null)
    {
        var promptLogits = fwd.Prefill(promptTokenIds);
        var positiveHidden = fwd.LastHidden.ToArray();

        return GenerateFromPrefilledState(
            fwd, negativeFwd, promptTokenIds.Length, promptLogits.ToArray(), positiveHidden,
            textEmbeddingTable, hiddenDim, speechStartId, speechEndId, speechDiffusionId, eosId,
            diffusionHeadWeights, acousticDecoderWeights, semanticEncoderWeights,
            acousticConnectorWeights, semanticConnectorWeights, speechScalingFactor, speechBiasFactor,
            layerNormEps, ddpmNumSteps, inferenceSteps, guidanceScale, maxSteps, rng, tokenSelectionOptions);
    }

    /// <summary>
    /// Real voice-cloning entry point: identical interleaved decode loop to <see cref="Generate"/>,
    /// but the prompt is prefilled POSITION-BY-POSITION via <paramref name="fwd"/>'s
    /// `ForwardEmbedding` (not the ordinary token-id `Prefill`) so that the `Voice input:` section's
    /// `speechDiffusion` placeholder positions can carry real spliced per-frame embeddings (real
    /// acoustic-connector-projected reference-audio latents) instead of an ordinary vocabulary
    /// lookup -- ported from `generator.cpp`'s `prepare_vibevoice_prompt`'s real speaker-audio
    /// branch (not guessed): only the ACOUSTIC connector is used for prompt splicing (unlike the
    /// per-diffusion-step loop below, which sums BOTH acoustic and semantic connector outputs --
    /// confirmed real, asymmetric, not a mistake).
    /// </summary>
    public static Result GenerateWithVoiceCloning(
        IForwardPass fwd,
        IForwardPass negativeFwd,
        int[] promptTokenIds, bool[] speechInputMask, float[][][] speakerAcousticMeansChannelMajor /* per speaker: [dim][frames] real encoder mean output */, int[] speakerSpeechTokenCounts,
        float[] textEmbeddingTable, int hiddenDim,
        int speechStartId, int speechEndId, int speechDiffusionId, int eosId,
        VibeVoiceDiffusionHeadWeights diffusionHeadWeights,
        VibeVoiceTokenizerDecoderWeights acousticDecoderWeights,
        VibeVoiceTokenizerEncoderWeights semanticEncoderWeights,
        VibeVoiceConnectorWeights acousticConnectorWeights,
        VibeVoiceConnectorWeights semanticConnectorWeights,
        float speechScalingFactor, float speechBiasFactor, float fixStd,
        float layerNormEps,
        int ddpmNumSteps, int inferenceSteps, float guidanceScale,
        int maxSteps, Random rng, SamplingParams? tokenSelectionOptions = null)
    {
        if (promptTokenIds.Length != speechInputMask.Length)
            throw new ArgumentException("promptTokenIds/speechInputMask length mismatch.");

        // Real per-speaker: Gaussian-sample the encoder's mean output (real `fix_std` reparameterization,
        // see VibeVoiceAcousticLatentSampler's own doc comment for the real formula + precision-gap note),
        // scale for the connector, then project -- real `[hiddenDim][frames]` per speaker.
        var projectedPerSpeaker = new float[speakerAcousticMeansChannelMajor.Length][][];
        for (int s = 0; s < speakerAcousticMeansChannelMajor.Length; s++)
        {
            var sampled = VibeVoiceAcousticLatentSampler.Sample(speakerAcousticMeansChannelMajor[s], fixStd, rng);
            int dim = sampled.Length, frames = sampled[0].Length;
            var scaled = new float[dim][];
            for (int c = 0; c < dim; c++)
            {
                var row = new float[frames];
                for (int t = 0; t < frames; t++) row[t] = (sampled[c][t] + speechBiasFactor) * speechScalingFactor;
                scaled[c] = row;
            }
            projectedPerSpeaker[s] = VibeVoiceConnector.Project(acousticConnectorWeights, scaled);
        }

        // Real splice: walk the prompt in order, and for every speech-mask position consume the next
        // real per-speaker projected frame (in speaker order, matching `build_selected_prompt_features`
        // + `splice_speech_embeddings`'s combined real effect for the non-batched single-request case).
        int[] frameCursor = new int[speakerAcousticMeansChannelMajor.Length];
        int speakerCursor = 0, framesConsumedForCurrentSpeaker = 0;

        ReadOnlySpan<float> lastLogits = default;
        float[] lastHidden = [];
        for (int pos = 0; pos < promptTokenIds.Length; pos++)
        {
            float[] embedding;
            if (speechInputMask[pos])
            {
                while (speakerCursor < speakerSpeechTokenCounts.Length &&
                       framesConsumedForCurrentSpeaker >= speakerSpeechTokenCounts[speakerCursor])
                {
                    speakerCursor++;
                    framesConsumedForCurrentSpeaker = 0;
                }
                var proj = projectedPerSpeaker[speakerCursor];
                int frame = frameCursor[speakerCursor]++;
                embedding = new float[hiddenDim];
                for (int c = 0; c < hiddenDim; c++) embedding[c] = proj[c][frame];
                framesConsumedForCurrentSpeaker++;
            }
            else
            {
                embedding = EmbedToken(textEmbeddingTable, promptTokenIds[pos], hiddenDim);
            }

            lastLogits = fwd.ForwardEmbedding(embedding, pos);
            lastHidden = fwd.LastHidden.ToArray();
        }

        return GenerateFromPrefilledState(
            fwd, negativeFwd, promptTokenIds.Length, lastLogits.ToArray(), lastHidden,
            textEmbeddingTable, hiddenDim, speechStartId, speechEndId, speechDiffusionId, eosId,
            diffusionHeadWeights, acousticDecoderWeights, semanticEncoderWeights,
            acousticConnectorWeights, semanticConnectorWeights, speechScalingFactor, speechBiasFactor,
            layerNormEps, ddpmNumSteps, inferenceSteps, guidanceScale, maxSteps, rng, tokenSelectionOptions);
    }

    private static Result GenerateFromPrefilledState(
        IForwardPass fwd, IForwardPass negativeFwd, int promptLength, float[] promptLogits, float[] positiveHidden,
        float[] textEmbeddingTable, int hiddenDim,
        int speechStartId, int speechEndId, int speechDiffusionId, int eosId,
        VibeVoiceDiffusionHeadWeights diffusionHeadWeights,
        VibeVoiceTokenizerDecoderWeights acousticDecoderWeights,
        VibeVoiceTokenizerEncoderWeights semanticEncoderWeights,
        VibeVoiceConnectorWeights acousticConnectorWeights,
        VibeVoiceConnectorWeights semanticConnectorWeights,
        float speechScalingFactor, float speechBiasFactor,
        float layerNormEps,
        int ddpmNumSteps, int inferenceSteps, float guidanceScale,
        int maxSteps, Random rng, SamplingParams? tokenSelectionOptions)
    {
        var generatedTokens = new List<int>();
        var audioSamples = new List<float>();

        // Real reference detail (`generator.cpp`'s `negative_cache`): the CFG negative/
        // unconditional branch runs on its OWN independent decoder cache, seeded ONLY with the
        // `speech_start` token -- it never sees the text prompt at all. `negativeFwd` MUST be a
        // separate IForwardPass instance from `fwd` (own KV cache/position state, weights may be
        // shared) so this branch is not contaminated by -- or overwritten by -- the positive
        // branch's prompt-conditioned cache. Position 0 here because negativeFwd's cache starts
        // empty (it never processes the prompt).
        var negativeStartEmbedding = EmbedToken(textEmbeddingTable, speechStartId, hiddenDim);
        negativeFwd.ForwardEmbedding(negativeStartEmbedding, 0);
        var negativeHidden = negativeFwd.LastHidden.ToArray();
        int negativePosition = 1;

        var scheduler = new VibeVoiceDpmSolverScheduler(ddpmNumSteps);
        scheduler.SetTimesteps(inferenceSteps);

        // Real streaming state (ported this session): ONE persistent cache set per real stream,
        // reused across every diffusion chunk -- matches the reference's real
        // `decode_acoustic_streaming`/`encode_semantic_streaming` chunk-boundary continuity.
        // Constructing fresh state per chunk (this class's earlier, real, documented
        // simplification) is equivalent to the non-streaming Encode/Decode and produces audible
        // boundary artifacts once a sample spans more than a couple of chunks.
        var acousticDecoderState = VibeVoiceTokenizerStreamingState.ForDecoder(acousticDecoderWeights);
        var semanticEncoderState = VibeVoiceTokenizerStreamingState.ForEncoder(semanticEncoderWeights);

        var currentLogits = promptLogits;
        var currentHidden = positiveHidden;
        int position = promptLength;

        for (int step = 0; step < maxSteps; step++)
        {
            if (Environment.GetEnvironmentVariable("STINGRAY_TTS_TRACE") is not null)
                Console.WriteLine($"vibevoice_tts.step.{step}.control_logits start={currentLogits[speechStartId]:F4} end={currentLogits[speechEndId]:F4} diff={currentLogits[speechDiffusionId]:F4} eos={currentLogits[eosId]:F4}");

            int token = VibeVoiceGenerationTokenSelector.Select(currentLogits, speechStartId, speechEndId, speechDiffusionId, eosId, tokenSelectionOptions, rng);
            generatedTokens.Add(token);
            if (token == eosId) break;

            if (token == speechStartId)
            {
                // Real reference: `negative_cache` is FULLY RESET here (a fresh
                // `prefill_embeddings(negative_start, 1)`, discarding all prior negative-branch
                // history), not merely appended to -- ported via a full TruncateTo(0) rewind of
                // negativeFwd's own independent cache before re-seeding it.
                negativeFwd.TruncateTo(0);
                var restart = EmbedToken(textEmbeddingTable, speechStartId, hiddenDim);
                negativeFwd.ForwardEmbedding(restart, 0);
                negativeHidden = negativeFwd.LastHidden.ToArray();
                negativePosition = 1;
            }

            if (token == speechEndId)
            {
                // Real reference: `speech_end` zeroes BOTH streaming codec caches in place --
                // a real per-segment lifecycle boundary (not a one-time reset), matching
                // `generator.cpp`'s `acoustic_streaming_state.set_to_zero()`/
                // `semantic_streaming_state.set_to_zero()`. Previously missing entirely in this
                // port (speechEndId was threaded through every signature but never compared
                // against `token`): a multi-segment generation (multiple speaker turns, or any
                // intra-utterance speech_start/speech_end pair the LLM emits) would carry stale
                // convolution-history state across a segment boundary that the reference always
                // clears.
                acousticDecoderState.ResetToZero();
                semanticEncoderState.ResetToZero();
            }

            float[] nextEmbedding;
            if (token == speechDiffusionId)
            {
                // VibeVoiceDiffusionSampler.Sample expects a single latent-sized noise vector
                // (it internally duplicates it into the real CFG cond/uncond pair each step, per
                // duplicate_positive_half) -- matches this class's own already-verified real-
                // weight test, not the reference's raw 2*latent_size buffer size (which stores
                // room for both halves but only ever reads/duplicates the first).
                var initialNoise = RandnBoxMuller(rng, diffusionHeadWeights.LatentSize);

                var speechLatent = VibeVoiceDiffusionSampler.Sample(
                    diffusionHeadWeights, scheduler, currentHidden, negativeHidden, initialNoise, guidanceScale);

                var unscaled = new float[speechLatent.Length];
                for (int i = 0; i < unscaled.Length; i++) unscaled[i] = speechLatent[i] / speechScalingFactor - speechBiasFactor;

                var latentChannelMajor = ToChannelMajorSingleFrame(unscaled);
                var chunkChannelMajor = VibeVoiceTokenizerDecoder.DecodeStreaming(acousticDecoderWeights, acousticDecoderState, latentChannelMajor, layerNormEps);
                var chunk = chunkChannelMajor[0];
                audioSamples.AddRange(chunk);

                var semanticFeatures = VibeVoiceTokenizerEncoder.EncodeStreaming(semanticEncoderWeights, semanticEncoderState, chunk, layerNormEps);
                var acousticEmbedding = VibeVoiceConnector.Project(acousticConnectorWeights, latentChannelMajor);
                var semanticEmbedding = VibeVoiceConnector.Project(semanticConnectorWeights, semanticFeatures);

                nextEmbedding = new float[hiddenDim];
                for (int c = 0; c < hiddenDim; c++) nextEmbedding[c] = acousticEmbedding[c][0] + semanticEmbedding[c][0];
            }
            else
            {
                nextEmbedding = EmbedToken(textEmbeddingTable, token, hiddenDim);
            }

            if (token == speechDiffusionId)
            {
                // Real reference: the negative cache only ever advances on a speech_diffusion
                // step, fed the SAME combined next_embedding as the positive branch that step
                // -- and its position counter is its own independent sequence length, not the
                // shared prompt-relative position (this MUST increment every such call; the
                // pre-fix version left it fixed, causing successive diffusion steps to overwrite
                // the same cache slot on the shared instance).
                negativeFwd.ForwardEmbedding(nextEmbedding, negativePosition);
                negativeHidden = negativeFwd.LastHidden.ToArray();
                negativePosition++;
            }

            currentLogits = fwd.ForwardEmbedding(nextEmbedding, position + 1).ToArray();
            currentHidden = fwd.LastHidden.ToArray();
            position++;
        }

        if (audioSamples.Count == 0) throw new InvalidOperationException("VibeVoice generation produced no audio.");
        return new Result([.. audioSamples], [.. generatedTokens]);
    }

    private static float[] EmbedToken(float[] textEmbeddingTable, int token, int hiddenDim)
    {
        var row = new float[hiddenDim];
        Array.Copy(textEmbeddingTable, (long)token * hiddenDim, row, 0, hiddenDim);
        return row;
    }

    private static float[][] ToChannelMajorSingleFrame(float[] latent)
    {
        var output = new float[latent.Length][];
        for (int c = 0; c < latent.Length; c++) output[c] = [latent[c]];
        return output;
    }

    /// <summary>Standard Box-Muller transform over .NET's own `Random` -- NOT the reference's real
    /// torch-compatible RNG. Known, deliberately accepted precision gap (same class as
    /// `VibeVoiceAcousticLatentSampler`'s own documented gap): drives every real diffusion step's
    /// noise, so it is NOT bit-identical to the reference for a given seed, but porting a
    /// bit-identical torch RNG is a large, likely-low-value undertaking (see this session's
    /// MOSS-TTS-Nano/VibeVoice ASR RNG-gap entries in docs/audio-review-progress.md for why).
    ///
    /// <para><b>2026-09-08 bisection note</b>: a real cross-engine trace with IDENTICAL injected
    /// noise (bypassing this RNG gap entirely, via the reference's `diffusion_noise_file` request
    /// option) confirmed the RNG gap is NOT the primary cause of VibeVoice TTS's audible-quality
    /// issue -- see the "VibeVoice TTS: real per-step cross-engine diffusion bisection" entry in
    /// `docs/audio-review-progress.md` for the full writeup and numbers. Root cause is compounding
    /// floating-point drift through the closed generation loop, not this RNG implementation.</para>
    /// </summary>
    private static float[] RandnBoxMuller(Random rng, int count)
    {
        var output = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            output[i] = (float)(mag * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < count) output[i + 1] = (float)(mag * Math.Sin(2.0 * Math.PI * u2));
        }
        return output;
    }
}
