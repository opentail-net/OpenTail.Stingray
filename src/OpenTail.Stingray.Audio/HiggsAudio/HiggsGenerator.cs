using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real, full zero-shot (no reference-audio conditioning) text-to-waveform generation loop for
/// Higgs Audio TTS, ported from `generator.cpp`'s `HiggsGenerator::generate` (not guessed):
/// prefill the real text prompt, sample the FIRST codes directly from the prefill's own last
/// hidden state, then repeatedly embed the previous (delay-masked) codes and step forward until
/// <see cref="HiggsCodebookSampler"/>'s real stop state machine signals done (or `maxTokens` is
/// hit), reverse the real delay pattern, clamp any still-reserved id (`BocId`/`EocId`) to `0`
/// (confirmed real: `generator.cpp` does exactly this, not an error), then decode through the
/// real acoustic codec.
///
/// <para><see cref="GenerateWithReferenceAudio"/> extends this with real reference-audio
/// conditioning (fused prompt positions carrying delayed reference codebook embeddings). Real,
/// deliberate scope limit still remaining: KV-cache reuse across repeated calls with the SAME
/// reference audio (`reference_prefix_cache_`/`reference_kv_ready_` in the reference) is a real
/// perf optimization, not implemented -- every call re-runs the full reference prefix.</para>
/// </summary>
public static class HiggsGenerator
{
    public readonly struct Result(float[] audioSamples, int[][] rawCodes)
    {
        public float[] AudioSamples { get; } = audioSamples;
        public int[][] RawCodes { get; } = rawCodes;
    }

    public static Result Generate(
        IForwardPass fwd, HiggsLlmTensorSource llm, HiggsTtsTextTokenizer tokenizer,
        HiggsCodecDecoderWeights codecWeights,
        string text, int numCodebooks, int audioVocabSize,
        int maxTokens, SamplingParams? options = null, Random? rng = null)
    {
        var prompt = tokenizer.EncodePrompt(text, referenceText: "", delayedReferenceTokens: 0);
        fwd.Prefill(prompt.TokenIds);
        return DecodeFromPrefilledState(fwd, llm, codecWeights, prompt.TokenIds.Length, numCodebooks, audioVocabSize, maxTokens, options, rng);
    }

    /// <summary>
    /// Real reference-audio-conditioned generation, ported from `generator.cpp`'s real
    /// `make_prompt_input`/`make_prepared_prompt` prompt-fusion (not guessed): the reference
    /// waveform is encoded to raw RVQ codes (<see cref="HiggsCodecEncoder"/>), delayed
    /// (<see cref="HiggsCodebooks.ApplyDelayPattern"/>), and each delayed reference FRAME becomes
    /// one PROMPT position (real `audio_token_id` placeholder, real config value `-100` --
    /// confirmed via `assets.cpp`'s own validation, not a sentinel) whose embedding is the real
    /// sum-of-per-codebook-modality-embedding formula (the SAME formula
    /// <see cref="HiggsArStepper.Step"/> already uses for decode-step embedding, just applied to a
    /// reference frame's codes instead of a just-generated one) rather than an ordinary text
    /// lookup. Real, confirmed: prompt-position gating is STATIC (a position is either a text
    /// token OR a full reference-code row, decided purely by `token_ids[position]==audio_token_id`
    /// in `make_prompt_input`) -- no learned gate predictor is needed for prompt construction, only
    /// for decode-step modality selection (already unconditional here, since generation only ever
    /// emits audio codes after the prompt). Since some prompt positions now carry non-vocabulary
    /// embeddings, prefills POSITION-BY-POSITION via `ForwardEmbedding` instead of the ordinary
    /// token-id `Prefill` <see cref="Generate"/> uses.
    /// </summary>
    public static Result GenerateWithReferenceAudio(
        IForwardPass fwd, HiggsLlmTensorSource llm, HiggsTtsTextTokenizer tokenizer,
        HiggsCodecDecoderWeights codecWeights, float[] textEmbeddingTable,
        string text, string referenceText, int[][] referenceRawCodes /* [frame][codebook] */,
        int numCodebooks, int audioVocabSize, int audioTokenId,
        int maxTokens, SamplingParams? options = null, Random? rng = null)
    {
        int hiddenDim = llm.HiddenDim;
        int rawFrames = referenceRawCodes.Length;
        var rawFlat = new int[rawFrames * numCodebooks];
        for (int t = 0; t < rawFrames; t++) Array.Copy(referenceRawCodes[t], 0, rawFlat, t * numCodebooks, numCodebooks);
        var delayedFlat = HiggsCodebooks.ApplyDelayPattern(rawFlat, rawFrames, numCodebooks);
        int delayedReferenceFrames = HiggsCodebooks.DelayedFrameCount(rawFrames, numCodebooks);

        var prompt = tokenizer.EncodePrompt(text, referenceText, delayedReferenceFrames);
        var modality = llm.ModalityEmbeddingWeight;

        float[] EmbedTextToken(int token)
        {
            var row = new float[hiddenDim];
            Array.Copy(textEmbeddingTable, (long)token * hiddenDim, row, 0, hiddenDim);
            return row;
        }

        float[] EmbedReferenceFrame(int frame)
        {
            var embedding = new float[hiddenDim];
            for (int cb = 0; cb < numCodebooks; cb++)
            {
                int code = delayedFlat[frame * numCodebooks + cb];
                long rowBase = (long)(cb * audioVocabSize + code) * hiddenDim;
                for (int d = 0; d < hiddenDim; d++) embedding[d] += modality[rowBase + d];
            }
            return embedding;
        }

        int referenceFrame = 0;
        for (int pos = 0; pos < prompt.TokenIds.Length; pos++)
        {
            var embedding = prompt.TokenIds[pos] == audioTokenId
                ? EmbedReferenceFrame(referenceFrame++)
                : EmbedTextToken(prompt.TokenIds[pos]);
            fwd.ForwardEmbedding(embedding, pos);
        }
        if (referenceFrame != delayedReferenceFrames)
            throw new InvalidOperationException("Higgs TTS prompt did not consume every delayed reference frame.");

        return DecodeFromPrefilledState(fwd, llm, codecWeights, prompt.TokenIds.Length, numCodebooks, audioVocabSize, maxTokens, options, rng);
    }

    private static Result DecodeFromPrefilledState(
        IForwardPass fwd, HiggsLlmTensorSource llm, HiggsCodecDecoderWeights codecWeights,
        int promptLength, int numCodebooks, int audioVocabSize,
        int maxTokens, SamplingParams? options, Random? rng)
    {
        var sampler = new HiggsCodebookSampler(numCodebooks);
        var delayedFrames = new List<int[]>();

        var first = HiggsArStepper.SampleFromHidden(fwd.LastHidden, llm, numCodebooks, audioVocabSize, options, rng);
        var maskedFirst = sampler.Step(first);
        delayedFrames.Add(maskedFirst);

        int position = promptLength;
        while (!sampler.GenerationDone && delayedFrames.Count < maxTokens)
        {
            var raw = HiggsArStepper.Step(fwd, llm, sampler.LastCodes, position, numCodebooks, audioVocabSize, options, rng);
            position++;
            var masked = sampler.Step(raw);
            if (masked.Length > 0 && masked[0] != HiggsCodebookSampler.StopCode)
                delayedFrames.Add(masked);
        }
        if (!sampler.GenerationDone)
            throw new InvalidOperationException("Higgs TTS generation reached maxTokens before EOC.");

        int delayedFrameCount = delayedFrames.Count;
        var delayedFlat = new int[delayedFrameCount * numCodebooks];
        for (int t = 0; t < delayedFrameCount; t++)
            Array.Copy(delayedFrames[t], 0, delayedFlat, t * numCodebooks, numCodebooks);

        var rawFlat = HiggsCodebooks.ReverseDelayPattern(delayedFlat, delayedFrameCount, numCodebooks);
        int rawFrameCount = delayedFrameCount - (numCodebooks - 1);

        int codecVocab = audioVocabSize - 2; // real: excludes BocId/EocId, matches HiggsCodecDecoderWeights.CodebookSize
        for (int i = 0; i < rawFlat.Length; i++)
            if (rawFlat[i] >= codecVocab) rawFlat[i] = 0;

        var rawCodes = new int[rawFrameCount][];
        for (int t = 0; t < rawFrameCount; t++)
        {
            var row = new int[numCodebooks];
            Array.Copy(rawFlat, t * numCodebooks, row, 0, numCodebooks);
            rawCodes[t] = row;
        }

        var waveform = HiggsCodecDecoder.Decode(codecWeights, rawCodes);
        return new Result(waveform, rawCodes);
    }
}
