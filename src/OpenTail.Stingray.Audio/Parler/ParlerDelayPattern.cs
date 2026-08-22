namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real MusicGen-style delayed multi-codebook pattern, transcribed directly from the real
/// `parler_tts` package's `build_delay_pattern_mask`/`apply_delay_pattern_mask`
/// (`modeling_parler_tts.py`, from the already-local
/// `scratch-llamacpp-ref/parler-pkg/parler_tts-0.2.3` source download -- NOT guessed). Confirmed
/// this fire (per ChatGPT's sourced guidance, cross-checked directly against this real source)
/// that Parler-TTS inherits MusicGen's exact staggering scheme unchanged: codebook <c>c</c> is
/// offset by exactly <c>c</c> positions relative to codebook 0, so codebook 0 is predicted first,
/// codebook 1 one step later, etc. See docs/audio-review-progress.md's Parler-TTS generation-loop
/// section for the full derivation.
///
/// <para><b>Real docstring example (4 codebooks, max_length=8, no prompt), reproduced exactly by
/// <see cref="Build"/> and asserted in <c>ParlerDelayPatternTests</c></b>:</para>
/// <code>
/// [B, -1, -1, -1, -1,  P,  P,  P]
/// [B,  B, -1, -1, -1, -1,  P,  P]
/// [B,  B,  B, -1, -1, -1, -1,  P]
/// [B,  B,  B,  B, -1, -1, -1, -1]
/// </code>
/// <para>where <c>B</c>=bos_token_id, <c>P</c>=pad_token_id, and <c>-1</c> marks a position whose
/// real value is not yet known (either "still to be predicted" during generation, or "carries the
/// real prompt value" when a prompt has non-empty content there).</para>
/// </summary>
public static class ParlerDelayPattern
{
    /// <summary>
    /// Real `build_delay_pattern_mask`, single-batch (bsz=1) form. <paramref name="inputIds"/> is
    /// `[numCodebooks][seqLen]` (the not-yet-delayed prompt, typically just one BOS-token column
    /// for zero-shot generation). Returns <c>(InputIds, PatternMask)</c>: <c>InputIds</c> is the
    /// real initial decoder input truncated to the first not-yet-known position (matches the real
    /// source's `first_start_id` truncation -- what the autoregressive loop should feed first);
    /// <c>PatternMask</c> is the FULL `[numCodebooks][maxLength]` mask (BOS/PAD/-1) that
    /// <see cref="Apply"/> later uses to force known BOS/PAD values over model-generated ones.
    /// </summary>
    public static (int[][] InputIds, int[][] PatternMask) Build(int[][] inputIds, int bosTokenId, int padTokenId, int maxLength, int numCodebooks)
    {
        int seqLen = inputIds[0].Length;

        if (maxLength < 2 * numCodebooks - 1)
        {
            // Real early-return: too short to apply a delay pattern at all -- return as-is.
            var identityMask = new int[numCodebooks][];
            for (int cb = 0; cb < numCodebooks; cb++)
            {
                identityMask[cb] = new int[maxLength];
                Array.Fill(identityMask[cb], -1);
            }
            return (inputIds, identityMask);
        }

        var shifted = new int[numCodebooks][];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            shifted[cb] = new int[maxLength];
            Array.Fill(shifted[cb], -1);
            for (int i = 0; i < seqLen; i++) shifted[cb][cb + i] = inputIds[cb][i];
        }

        // Real torch.tril(ones(numCodebooks, maxLength)): true where pos <= cb (BOS region).
        // Real torch.triu(ones(...), diagonal=maxLength-numCodebooks+1): true where pos - cb >= maxLength-numCodebooks+1 (EOS/PAD region).
        int eosDiagonal = maxLength - numCodebooks + 1;

        var pattern = new int[numCodebooks][];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            pattern[cb] = new int[maxLength];
            for (int pos = 0; pos < maxLength; pos++)
            {
                bool isBos = pos <= cb;
                bool isEos = pos - cb >= eosDiagonal;
                pattern[cb][pos] = isBos ? bosTokenId : isEos ? padTokenId : shifted[cb][pos];
            }
        }

        // Real "find the first position to start generating": first -1 along codebook 0's row
        // (codebook 0 has no offset, so if none exists the whole matrix is already fully known).
        int firstStartId = maxLength;
        for (int pos = 0; pos < maxLength; pos++)
        {
            if (pattern[0][pos] == -1) { firstStartId = pos; break; }
        }

        var truncated = new int[numCodebooks][];
        for (int cb = 0; cb < numCodebooks; cb++)
            truncated[cb] = pattern[cb][..firstStartId];

        return (truncated, pattern);
    }

    /// <summary>
    /// Real `apply_delay_pattern_mask`: wherever <paramref name="patternMask"/> is <c>-1</c>, keep
    /// the model-generated value in <paramref name="inputIds"/>; everywhere else, force the
    /// mask's own BOS/PAD value, overriding whatever the model predicted there.
    /// </summary>
    public static int[][] Apply(int[][] inputIds, int[][] patternMask)
    {
        int numCodebooks = inputIds.Length;
        int seqLen = inputIds[0].Length;
        var output = new int[numCodebooks][];
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            output[cb] = new int[seqLen];
            for (int pos = 0; pos < seqLen; pos++)
                output[cb][pos] = patternMask[cb][pos] == -1 ? inputIds[cb][pos] : patternMask[cb][pos];
        }
        return output;
    }
}
