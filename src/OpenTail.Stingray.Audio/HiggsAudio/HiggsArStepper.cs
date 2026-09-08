using OpenTail.Stingray.Engine;

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
    /// to get `[numCodebooks, audioVocabSize]` logits and samples each codebook independently.
    /// Real reference formula: `logits = Linear(hidden, modality_embedding, no bias)`, reshaped
    /// per codebook.
    ///
    /// <para>Real sampling formula, ported from `sampler.cpp`'s `sample_codebook_row` (not
    /// guessed): `temperature &lt;= 0` (the reference's `kGreedyTemperatureThreshold`) or
    /// `top_k==1` is argmax; otherwise `scores = logits/temperature`, real top-k on those SCORES
    /// (logit space, before softmax) when set, softmax, then real top-p on the resulting
    /// probabilities, then a multinomial draw. This reuses <see cref="Sampler.Sample"/> (same
    /// logit-space top-k-then-softmax-then-top-p ordering via its top-k fast path) rather than a
    /// bespoke reimplementation -- the one real, flagged gap is RNG: the reference draws via a
    /// real seeded SGLang Gumbel-max trick or Torch-CUDA multinomial (`sample_seeded_sglang_gumbel`/
    /// `sample_unseeded_torch_multinomial`), this port uses .NET's own `Random`-driven categorical
    /// draw -- same non-bit-exact-RNG gap already accepted elsewhere this session (VoxCPM2's CFM
    /// solver, VibeVoice's diffusion sampler). `options: null` (default) preserves the exact
    /// previous argmax-only behavior byte-for-byte.</para>
    /// </summary>
    public static int[] SampleFromHidden(ReadOnlySpan<float> hidden, HiggsLlmTensorSource llm, int numCodebooks, int audioVocabSize,
        SamplingParams? options = null, Random? rng = null)
    {
        int hiddenDim = llm.HiddenDim;
        var modality = llm.ModalityEmbeddingWeight; // [numCodebooks*audioVocabSize, hiddenDim]
        var normWeight = llm.NormWeight;

        // Output RMSNorm: ForwardPass.LastHidden is pre-norm, so normalize before projecting to modality logits (matches ar.cpp).
        double sumSq = 0;
        for (int d = 0; d < hiddenDim; d++) sumSq += (double)hidden[d] * hidden[d];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / hiddenDim + 1e-6));
        var normedHidden = new float[hiddenDim];
        for (int d = 0; d < hiddenDim; d++) normedHidden[d] = hidden[d] * invRms * normWeight[d];

        var codes = new int[numCodebooks];
        var logitsRow = options is null ? null : new float[audioVocabSize];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            long rowBase = (long)cb * audioVocabSize * hiddenDim;
            if (options is null)
            {
                float best = float.NegativeInfinity;
                int bestIdx = 0;
                for (int v = 0; v < audioVocabSize; v++)
                {
                    long row = rowBase + (long)v * hiddenDim;
                    float dot = 0f;
                    for (int d = 0; d < hiddenDim; d++) dot += modality[row + d] * normedHidden[d];
                    if (dot > best) { best = dot; bestIdx = v; }
                }
                codes[cb] = bestIdx;
            }
            else
            {
                for (int v = 0; v < audioVocabSize; v++)
                {
                    long row = rowBase + (long)v * hiddenDim;
                    float dot = 0f;
                    for (int d = 0; d < hiddenDim; d++) dot += modality[row + d] * normedHidden[d];
                    logitsRow![v] = dot;
                }
                codes[cb] = Sampler.Sample(logitsRow!, options, rng);
            }
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
        int numCodebooks, int audioVocabSize, SamplingParams? options = null, Random? rng = null)
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
        return SampleFromHidden(fwd.LastHidden, llm, numCodebooks, audioVocabSize, options, rng);
    }
}
