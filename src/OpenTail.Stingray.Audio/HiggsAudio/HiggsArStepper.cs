namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real per-step audio-token decode primitive for Higgs Audio TTS's AR generation loop, ported
/// from `ar.cpp`'s `build_higgs_decode_code_embedding`/`build_modality_logits` (not guessed):
/// embeds the previous step's 8 codebook ids via the real shared
/// <see cref="HiggsLlmTensorSource.ModalityEmbeddingWeight"/> table (each codebook occupying its
/// own disjoint `[codebook*audioVocabSize, (codebook+1)*audioVocabSize)` row range, summed
/// across codebooks -- confirmed real, not the "gated fusion" formula `build_higgs_prefill_
/// input_embedding` uses for TEXT positions, since a pure decode step has `text_gate=0` and this
/// simplified function has no gate parameters at all), runs one `ForwardEmbedding` step, then
/// projects the resulting hidden state back through the SAME modality-embedding table (real
/// weight tying, confirmed via `build_modality_logits`'s reuse of `weights.modality_embedding`
/// as the output projection) to get `[numCodebooks, audioVocabSize]` logits and argmax-samples
/// each codebook independently.
///
/// <para>Deliberately a bounded, callable-per-step primitive rather than a full generation loop
/// with stopping/EOS logic -- this session did not read `generator.cpp`'s real initial-seed-code
/// convention (what codes bootstrap the very first decode step after the text prefill) closely
/// enough to port that without guessing, so the caller supplies `previousCodes` every call
/// (including the first) rather than this class inventing a BOS convention.</para>
/// </summary>
public static class HiggsArStepper
{
    /// <summary>Embeds `previousCodes[numCodebooks]` (already-offset-free raw per-codebook ids,
    /// each in `[0, audioVocabSize)`) via the real sum-of-codebook-embeddings formula, runs one
    /// `ForwardEmbedding` step at `position`, then argmax-samples the next step's codes from the
    /// real modality-logits projection.</summary>
    public static int[] Step(IForwardPass fwd, HiggsLlmTensorSource llm, int[] previousCodes, int position,
        int numCodebooks, int audioVocabSize)
    {
        if (previousCodes.Length != numCodebooks) throw new ArgumentException("previousCodes length must equal numCodebooks.");

        int hiddenDim = llm.HiddenDim;
        var modality = llm.ModalityEmbeddingWeight; // [numCodebooks*audioVocabSize, hiddenDim]

        var embedding = new float[hiddenDim];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            int code = previousCodes[cb];
            if ((uint)code >= (uint)audioVocabSize) throw new ArgumentOutOfRangeException(nameof(previousCodes));
            long rowBase = (long)(cb * audioVocabSize + code) * hiddenDim;
            for (int d = 0; d < hiddenDim; d++) embedding[d] += modality[rowBase + d];
        }

        fwd.ForwardEmbedding(embedding, position);
        var lastHidden = fwd.LastHidden;

        var nextCodes = new int[numCodebooks];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            float best = float.NegativeInfinity;
            int bestIdx = 0;
            long rowBase = (long)cb * audioVocabSize * hiddenDim;
            for (int v = 0; v < audioVocabSize; v++)
            {
                long row = rowBase + (long)v * hiddenDim;
                float dot = 0f;
                for (int d = 0; d < hiddenDim; d++) dot += modality[row + d] * lastHidden[d];
                if (dot > best) { best = dot; bestIdx = v; }
            }
            nextCodes[cb] = bestIdx;
        }
        return nextCodes;
    }
}
