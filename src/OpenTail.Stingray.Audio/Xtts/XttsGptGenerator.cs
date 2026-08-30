using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 GPT generation orchestration: builds the real prefix (conditioning latents +
/// text embeddings), then computes next-mel-token logits given the mel tokens generated so far.
/// Supports high-speed O(1) single-step generation via <see cref="XttsGptCache"/>.
/// </summary>
public static class XttsGptGenerator
{
    /// <summary>
    /// Real prefix construction (`GPT.compute_embeddings`): text is padded with
    /// [start_text_token, ...ids, stop_text_token], embedded (token+positional), and concatenated
    /// AFTER the real conditioning latents. condLatents is channel-first [ModelDim, 32] (as
    /// returned by <see cref="XttsConditioningEncoder.Encode"/>); textIds are RAW (unpadded) real
    /// tokenizer ids. Returns the prefix, token-major [PrefixLen, ModelDim].
    /// </summary>
    public static float[] BuildPrefix(XttsGptEmbeddings emb, float[] condLatents, int numCondLatents, ReadOnlySpan<int> textIds, out int prefixLen)
    {
        int dim = XttsGptWeights.ModelDim;

        var paddedText = new int[textIds.Length + 2];
        paddedText[0] = XttsGptEmbeddings.TextStartToken;
        for (int i = 0; i < textIds.Length; i++) paddedText[i + 1] = textIds[i];
        paddedText[^1] = XttsGptEmbeddings.TextStopToken;

        var textEmb = emb.EmbedText(paddedText); // token-major [paddedText.Length, dim]

        prefixLen = numCondLatents + paddedText.Length;
        var prefix = new float[prefixLen * dim];

        // condLatents is channel-first [dim, numCondLatents] -- transpose to token-major.
        for (int li = 0; li < numCondLatents; li++)
            for (int d = 0; d < dim; d++)
                prefix[li * dim + d] = condLatents[d * numCondLatents + li];

        Array.Copy(textEmb, 0, prefix, numCondLatents * dim, textEmb.Length);
        return prefix;
    }

    /// <summary>
    /// Fast KV-cached autoregressive mel-token generation loop:
    /// Prefills the prefix once into XttsGptCache, then evaluates single-token steps in O(1) time
    /// per layer, accumulating the exact vocoder latents along the way.
    /// </summary>
    public static (List<int> GeneratedCodes, float[] Latents) Generate(
        XttsGptWeights trunkWeights,
        XttsGptEmbeddings emb,
        XttsGptCache cache,
        float[] prefixTokenMajor,
        int prefixLen,
        Random rng,
        SamplingParams? p = null,
        int maxTokens = XttsGptSampler.MaxAudioTokens)
    {
        p ??= XttsGptSampler.DefaultParams;
        cache.Reset();

        int dim = XttsGptWeights.ModelDim;

        // 1. Prefill prefix tokens (cond latents + text tokens)
        for (int i = 0; i < prefixLen; i++)
        {
            var tokenVec = prefixTokenMajor.AsSpan(i * dim, dim);
            XttsGptTrunk.Step(trunkWeights, cache, tokenVec);
        }

        // 2. Feed start audio token (token 0 in mel modality)
        int startAudioToken = XttsGptEmbeddings.AudioStartToken;
        var startMelEmb = emb.EmbedSingleMel(startAudioToken, 0);
        var lastHidden = XttsGptTrunk.Step(trunkWeights, cache, startMelEmb);

        var melLatentsList = new List<float[]>();
        // Add start_audio_token's final-normed latent
        melLatentsList.Add(emb.FinalNormOnly(lastHidden.ToArray()));

        var melTokensSoFar = new List<int> { startAudioToken };
        var generated = new List<int>();

        for (int step = 0; step < maxTokens; step++)
        {
            float[] logits = emb.MelLogits(lastHidden.ToArray());

            var samplingParams = generated.Count > 0 ? p with { PreviousTokens = generated } : p;
            int next = Sampler.Sample(logits, samplingParams, rng);

            if (next == XttsGptEmbeddings.AudioStopToken)
                break;

            generated.Add(next);
            melTokensSoFar.Add(next);

            // Step with the next generated token at mel position (step + 1)
            var melVec = emb.EmbedSingleMel(next, step + 1);
            lastHidden = XttsGptTrunk.Step(trunkWeights, cache, melVec);
            melLatentsList.Add(emb.FinalNormOnly(lastHidden.ToArray()));
        }

        // Format latents: channel-first [dim, melLen]
        int melLen = melLatentsList.Count;
        var latents = new float[dim * melLen];
        for (int mi = 0; mi < melLen; mi++)
        {
            var h = melLatentsList[mi];
            for (int d = 0; d < dim; d++)
                latents[d * melLen + mi] = h[d];
        }

        return (generated, latents);
    }

    /// <summary>
    /// Real next-mel-token logits given the prefix and the mel tokens generated so far (INCLUDING
    /// the leading `start_audio_token` -- caller must include it as the first entry of
    /// `melTokensSoFar`). Recomputes the whole sequence through the trunk (see this class's own
    /// doc comment for why). Returns [NumAudioTokens] logits for the NEXT token.
    /// </summary>
    public static float[] NextMelLogits(XttsGptWeights trunkWeights, XttsGptEmbeddings emb, float[] prefixTokenMajor, int prefixLen, ReadOnlySpan<int> melTokensSoFar)
    {
        int dim = XttsGptWeights.ModelDim;
        var melEmb = emb.EmbedMel(melTokensSoFar); // token-major [melTokensSoFar.Length, dim]

        int totalT = prefixLen + melTokensSoFar.Length;
        var fullTokenMajor = new float[totalT * dim];
        Array.Copy(prefixTokenMajor, 0, fullTokenMajor, 0, prefixTokenMajor.Length);
        Array.Copy(melEmb, 0, fullTokenMajor, prefixTokenMajor.Length, melEmb.Length);

        // XttsGptTrunk.Forward expects channel-first [dim, T].
        var channelFirst = new float[dim * totalT];
        for (int ti = 0; ti < totalT; ti++)
            for (int d = 0; d < dim; d++)
                channelFirst[d * totalT + ti] = fullTokenMajor[ti * dim + d];

        var trunkOut = XttsGptTrunk.Forward(trunkWeights, channelFirst, totalT);

        var lastHidden = new float[dim];
        for (int d = 0; d < dim; d++) lastHidden[d] = trunkOut[d * totalT + (totalT - 1)];

        return emb.MelLogits(lastHidden);
    }
}
