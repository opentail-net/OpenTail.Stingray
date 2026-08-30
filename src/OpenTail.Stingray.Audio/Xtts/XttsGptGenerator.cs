
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 GPT generation orchestration: builds the real prefix (conditioning latents +
/// text embeddings), then computes next-mel-token logits given the mel tokens generated so far.
///
/// <para><b>Correctness-first, non-KV-cached design</b>: recomputes the FULL sequence (prefix +
/// all mel tokens so far) through the trunk on every call, rather than porting the real
/// reference's KV-cache-based single-step `GPT2InferenceModel.forward` (see
/// `TTS/tts/layers/xtts/gpt_inference.py`). Mathematically equivalent (a causal transformer's
/// output at the last position is identical whether computed incrementally with a KV cache or by
/// re-running the whole prefix), just O(T²) instead of O(T) per generated token -- matches this
/// codebase's own established pattern of porting a stateless/correct version first, then adding a
/// real KV cache as a follow-up performance pass once correctness is golden-verified (see
/// `FishSpeechFastAr.Forward` vs. `.ForwardStep`/`FishSpeechFastArCache` for the exact same
/// staged approach on a different pipeline, CLAUDE.md rule 7).</para>
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
