using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 autoregressive mel-token generation loop: repeatedly calls
/// <see cref="XttsGptGenerator.NextMelLogits"/> and samples the next code with this codebase's
/// existing <see cref="Sampler"/> (temperature/top-k/top-p/repetition-penalty -- the exact same
/// sampling family HF's `GenerationMixin.generate()` uses under the hood for XTTS's real
/// `**hf_generate_kwargs`, so this reuses the engine's own battle-tested sampler rather than
/// hand-rolling a second implementation), until the real stop-audio-token is sampled or the
/// real max-length cap is hit.
///
/// <para>Real sampling defaults confirmed from `coqui/XTTS-v2`'s own `config.json`:
/// `temperature=0.75, top_k=50, top_p=0.85, repetition_penalty=5.0`. Real max length from
/// `model_args.gpt_max_audio_tokens=605` (a hard safety cap; real generation almost always stops
/// much earlier via the stop token).</para>
///
/// <para><b>One known, minor divergence from the reference</b>: HF's `RepetitionPenaltyLogitsProcessor`
/// operates over the FULL `input_ids` tensor passed to `generate()`, which (per XTTS's own
/// `compute_embeddings`) includes dummy placeholder ids (value `1`) for the entire
/// conditioning+text prefix region, not just the real generated mel tokens -- an artifact of how
/// the reference structures its `input_ids` for HF's generation API, not a deliberate modeling
/// choice. This port applies the repetition penalty only over the REAL generated mel token
/// history (excluding the leading start-audio-token and any prefix placeholders), which is the
/// sensible/intended behavior -- flagged here rather than blindly reproducing the placeholder
/// artifact.</para>
/// </summary>
public static class XttsGptSampler
{
    public const int MaxAudioTokens = 605;

    public static readonly SamplingParams DefaultParams = new()
    {
        Temperature = 0.75f,
        TopK = 50,
        TopP = 0.85f,
        RepetitionPenalty = 5.0f,
        RepeatLastN = 0, // real reference's repetition penalty is unwindowed (whole history)
    };

    /// <summary>Generates real mel/audio codebook token ids autoregressively. Does NOT include the leading start_audio_token or the trailing stop_audio_token in the returned list.</summary>
    public static List<int> Generate(XttsGptWeights trunkWeights, XttsGptEmbeddings embWeights, float[] prefixTokenMajor, int prefixLen, Random rng, SamplingParams? p = null, int maxTokens = MaxAudioTokens)
    {
        p ??= DefaultParams;
        var melTokensSoFar = new List<int> { XttsGptEmbeddings.AudioStartToken };
        var generated = new List<int>();

        for (int step = 0; step < maxTokens; step++)
        {
            float[] logits = XttsGptGenerator.NextMelLogits(trunkWeights, embWeights, prefixTokenMajor, prefixLen, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(melTokensSoFar));

            var samplingParams = generated.Count > 0 ? p with { PreviousTokens = generated } : p;
            int next = Sampler.Sample(logits, samplingParams, rng);

            if (next == XttsGptEmbeddings.AudioStopToken)
                break;

            generated.Add(next);
            melTokensSoFar.Add(next);
        }

        return generated;
    }
}
