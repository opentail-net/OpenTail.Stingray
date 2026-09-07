namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real per-step audio-token decode primitives for Higgs Audio TTS's AR generation loop, ported
/// from `ar.cpp`'s `build_higgs_decode_code_embedding`/`build_modality_logits` and
/// `generator.cpp`'s real prefill-to-first-sample flow (not guessed).
///
/// <para><b>Real, confirmed sequence (`generator.cpp` ~line 436-448)</b>: the very FIRST sampled
/// codes come directly from `prefill.codebook_logits` -- the prefill's own last-position hidden
/// state (the real prompt ends in a literal `&lt;|audio|&gt;` text token) projected through the
/// modality-embedding table, sampled BEFORE any decode-step `ForwardEmbedding` call runs at all.
/// Only SUBSEQUENT codes come from feeding the previous step's REAL sampled codes through
/// `ForwardEmbedding`. This class splits the "project hidden state to codebook logits and
/// sample" step (<see cref="SampleFromHidden"/>) from the "embed previous codes and advance one
/// position" step (<see cref="Step"/>) so a caller can use the former directly on the prefill's
/// own `LastHidden` for the first sample, then the latter for every step after.</para>
/// </summary>
public static class HiggsArStepper
{
    /// <summary>Projects `hidden` through the real shared modality-embedding table (weight-tied
    /// output head, confirmed via `build_modality_logits`'s reuse of `weights.modality_embedding`)
    /// to get `[numCodebooks, audioVocabSize]` logits and argmax-samples each codebook
    /// independently. Real reference formula: `logits = Linear(hidden, modality_embedding, no
    /// bias)`, reshaped per codebook.</summary>
    public static int[] SampleFromHidden(ReadOnlySpan<float> hidden, HiggsLlmTensorSource llm, int numCodebooks, int audioVocabSize)
    {
        int hiddenDim = llm.HiddenDim;
        var modality = llm.ModalityEmbeddingWeight; // [numCodebooks*audioVocabSize, hiddenDim]

        var codes = new int[numCodebooks];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            float best = float.NegativeInfinity;
            int bestIdx = 0;
            long rowBase = (long)cb * audioVocabSize * hiddenDim;
            for (int v = 0; v < audioVocabSize; v++)
            {
                long row = rowBase + (long)v * hiddenDim;
                float dot = 0f;
                for (int d = 0; d < hiddenDim; d++) dot += modality[row + d] * hidden[d];
                if (dot > best) { best = dot; bestIdx = v; }
            }
            codes[cb] = bestIdx;
        }
        return codes;
    }

    /// <summary>Embeds `previousCodes[numCodebooks]` (raw per-codebook ids, each in
    /// `[0, audioVocabSize)`) via the real sum-of-codebook-embeddings formula (confirmed real,
    /// distinct from the "gated fusion" formula `build_higgs_prefill_input_embedding` uses for
    /// TEXT positions -- a pure decode step always has `text_gate=0`), runs one
    /// `ForwardEmbedding` step at `position`, then samples the NEXT step's codes via
    /// <see cref="SampleFromHidden"/> on the resulting hidden state.</summary>
    public static int[] Step(IForwardPass fwd, HiggsLlmTensorSource llm, int[] previousCodes, int position,
        int numCodebooks, int audioVocabSize)
    {
        if (previousCodes.Length != numCodebooks) throw new ArgumentException("previousCodes length must equal numCodebooks.");

        int hiddenDim = llm.HiddenDim;
        var modality = llm.ModalityEmbeddingWeight;

        var embedding = new float[hiddenDim];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            int code = previousCodes[cb];
            if ((uint)code >= (uint)audioVocabSize) throw new ArgumentOutOfRangeException(nameof(previousCodes));
            long rowBase = (long)(cb * audioVocabSize + code) * hiddenDim;
            for (int d = 0; d < hiddenDim; d++) embedding[d] += modality[rowBase + d];
        }

        fwd.ForwardEmbedding(embedding, position);
        return SampleFromHidden(fwd.LastHidden, llm, numCodebooks, audioVocabSize);
    }
}
