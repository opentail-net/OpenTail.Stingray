
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 vocoder-input extraction: the GPT trunk's own 1024-dim hidden states at the mel
/// positions (channel-first [ModelDim, T]) -- confirmed from the real reference
/// (`TTS/tts/models/xtts.py`'s `inference`/`full_inference` methods) that the vocoder does NOT
/// consume the DVAE decoder's output. Real inference calls `gpt_latents = self.gpt(text_tokens,
/// text_len, gpt_codes, expected_output_len, cond_latents=gpt_cond_latent, return_latent=True)`
/// -- a SECOND, non-generation forward pass over the already-sampled codes (via `GPT.forward`,
/// not `.generate()`), with `return_latent=True` making `GPT.get_logits` return hidden states
/// (after BOTH the trunk's own `ln_f` and the separate `gpt.final_norm` -- see
/// <see cref="XttsGptTrunk"/>/<see cref="XttsGptEmbeddings"/>'s doc comments for that detail)
/// instead of projecting through `mel_head`.
///
/// <para><b>Deliberately does not replicate `GPT.forward`'s real padding/trim arithmetic</b>
/// (`code_lengths = ceil(wav_lengths/code_stride_len)+3`, `set_mel_padding`, a `sub=-5` trailing
/// trim the reference's own authors flagged "don't ask me why 😄"). This is safe: causal
/// self-attention means a position's hidden state depends only on itself and EARLIER positions,
/// never on tokens appended after it -- so feeding exactly `[start_audio_token, ...generated
/// codes]` (no trailing stop/padding tokens) and taking every position's hidden state gives
/// EXACTLY the same values the padded reference would produce at the same relative positions,
/// just without the reference's extra trailing padding positions to trim away. Verified directly:
/// see `XttsGptLatentsTests`, which compares against the real reference's own (pre-trim-length)
/// output over the overlapping prefix of positions.</para>
/// </summary>
public static class XttsGptLatents
{
    /// <summary>
    /// generatedCodes are the real sampled mel/audio codes (e.g. from
    /// <see cref="XttsGptSampler.Generate"/>'s output), WITHOUT the leading start_audio_token or
    /// a trailing stop_audio_token -- both are handled internally. Returns per-position hidden
    /// states for the sequence `[start_audio_token, ...generatedCodes]`, channel-first
    /// [ModelDim, generatedCodes.Length + 1].
    /// </summary>
    public static float[] ComputeLatents(XttsGptWeights trunkWeights, XttsGptEmbeddings embWeights, float[] prefixTokenMajor, int prefixLen, ReadOnlySpan<int> generatedCodes)
    {
        int dim = XttsGptWeights.ModelDim;
        int melLen = generatedCodes.Length + 1;

        var melIds = new int[melLen];
        melIds[0] = XttsGptEmbeddings.AudioStartToken;
        for (int i = 0; i < generatedCodes.Length; i++) melIds[i + 1] = generatedCodes[i];

        var melEmb = embWeights.EmbedMel(melIds); // token-major [melLen, dim]

        int totalT = prefixLen + melLen;
        var fullTokenMajor = new float[totalT * dim];
        Array.Copy(prefixTokenMajor, 0, fullTokenMajor, 0, prefixTokenMajor.Length);
        Array.Copy(melEmb, 0, fullTokenMajor, prefixTokenMajor.Length, melEmb.Length);

        var channelFirst = new float[dim * totalT];
        for (int ti = 0; ti < totalT; ti++)
            for (int d = 0; d < dim; d++)
                channelFirst[d * totalT + ti] = fullTokenMajor[ti * dim + d];

        var trunkOut = XttsGptTrunk.Forward(trunkWeights, channelFirst, totalT); // includes ln_f, channel-first [dim, totalT]

        // Real `get_logits`: enc = final_norm(trunk_output[offset:]); return enc[-melLen:] --
        // apply the SEPARATE gpt.final_norm (the same one MelLogits/TextLogits use) to just the
        // mel positions, position by position.
        var latents = new float[dim * melLen];
        var posBuffer = new float[dim];
        for (int mi = 0; mi < melLen; mi++)
        {
            int ti = prefixLen + mi;
            for (int d = 0; d < dim; d++) posBuffer[d] = trunkOut[d * totalT + ti];
            var normed = embWeights.FinalNormOnly(posBuffer);
            for (int d = 0; d < dim; d++) latents[d * melLen + mi] = normed[d];
        }

        return latents;
    }
}
