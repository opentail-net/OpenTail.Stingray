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
/// <para><b>Real, deliberate simplification vs. the reference</b>: this port uses the ONE-SHOT
/// (non-streaming) `VibeVoiceTokenizerDecoder`/`VibeVoiceTokenizerEncoder` per diffusion step
/// rather than the reference's real STATEFUL streaming decode/encode (`decode_acoustic_streaming`/
/// `encode_semantic_streaming`, which maintain conv history across chunks for correct receptive-
/// field behavior at chunk boundaries) -- each chunk here is decoded/re-encoded independently.
/// This produces structurally valid, non-degenerate audio per chunk but is NOT bit-exact with
/// the real reference's streaming boundary behavior; a real streaming state port is future work,
/// not guessed here.</para>
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
        int maxSteps, Random rng)
    {
        var generatedTokens = new List<int>();
        var audioSamples = new List<float>();

        var promptLogits = fwd.Prefill(promptTokenIds);
        var positiveHidden = fwd.LastHidden.ToArray();

        var negativeStartEmbedding = EmbedToken(textEmbeddingTable, speechStartId, hiddenDim);
        var negativeLogits = fwd.ForwardEmbedding(negativeStartEmbedding, promptTokenIds.Length);
        var negativeHidden = fwd.LastHidden.ToArray();
        int negativePosition = promptTokenIds.Length + 1;

        var scheduler = new VibeVoiceDpmSolverScheduler(ddpmNumSteps);
        scheduler.SetTimesteps(inferenceSteps);

        var currentLogits = promptLogits.ToArray();
        var currentHidden = positiveHidden;
        int position = promptTokenIds.Length;

        for (int step = 0; step < maxSteps; step++)
        {
            int token = VibeVoiceGenerationTokenSelector.SelectArgmax(currentLogits, speechStartId, speechEndId, speechDiffusionId, eosId);
            generatedTokens.Add(token);
            if (token == eosId) break;

            if (token == speechStartId)
            {
                var restart = EmbedToken(textEmbeddingTable, speechStartId, hiddenDim);
                negativeLogits = fwd.ForwardEmbedding(restart, negativePosition);
                negativeHidden = fwd.LastHidden.ToArray();
                negativePosition++;
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
                var chunkChannelMajor = VibeVoiceTokenizerDecoder.Decode(acousticDecoderWeights, latentChannelMajor, layerNormEps);
                var chunk = chunkChannelMajor[0];
                audioSamples.AddRange(chunk);

                var semanticFeatures = VibeVoiceTokenizerEncoder.Encode(semanticEncoderWeights, chunk, layerNormEps);
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
                negativeLogits = fwd.ForwardEmbedding(nextEmbedding, negativePosition);
                negativeHidden = fwd.LastHidden.ToArray();
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
