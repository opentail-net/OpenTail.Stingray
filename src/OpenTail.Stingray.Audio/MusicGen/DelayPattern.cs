
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// MusicGen's delayed multi-codebook pattern, transcribed from the real `transformers`
/// `MusicgenForConditionalGeneration.build_delay_pattern_mask`/`apply_delay_pattern_mask`.
/// Codebook <c>q</c> is offset by <c>q</c> frames so a single decoder forward step produces one
/// diagonal column across all codebooks -- this is why generation needs only `frames + codebooks -
/// 1` steps instead of `codebooks * frames` sequential per-codebook steps.
///
/// <para>Real layout (codebooks=4, PAD token = pad_token_id from decoder config, BOS = same id in
/// real MusicGen since bos_token_id == pad_token_id == 2048 for musicgen-small):</para>
/// <code>
/// CB0:  A0  A1  A2  A3  PAD PAD PAD
/// CB1:  PAD B0  B1  B2  B3  PAD PAD
/// CB2:  PAD PAD C0  C1  C2  C3  PAD
/// CB3:  PAD PAD PAD D0  D1  D2  D3
/// </code>
/// Note the real pattern also pads the TAIL (not just the head) up to `frames + codebooks - 1`
/// columns, unlike a naive triangular stagger -- <see cref="BuildInput"/> reflects that.
/// </summary>
public static class DelayPattern
{
    /// <summary>
    /// Builds the delayed input grid from `[codebooks][frames]` clean tokens. Returns
    /// `[codebooks][frames + codebooks - 1]`, with codebook `q`'s real tokens written starting at
    /// column `q` and every other cell set to <paramref name="padToken"/>.
    /// </summary>
    public static int[][] BuildInput(int[][] tokens, int padToken)
    {
        int codebooks = tokens.Length;
        int frames = tokens[0].Length;
        int seqLen = frames + codebooks - 1;

        var result = new int[codebooks][];
        for (int q = 0; q < codebooks; q++)
        {
            result[q] = new int[seqLen];
            Array.Fill(result[q], padToken);
            Array.Copy(tokens[q], 0, result[q], q, frames);
        }
        return result;
    }

    /// <summary>
    /// Reverses <see cref="BuildInput"/>: given the delayed `[codebooks][frames + codebooks - 1]`
    /// grid (as generated, one diagonal column at a time), extracts the clean
    /// `[codebooks][frames]` token streams by reading codebook `q` starting at column `q`.
    /// </summary>
    public static int[][] RemoveDelay(int[][] delayedTokens, int frames)
    {
        int codebooks = delayedTokens.Length;
        var result = new int[codebooks][];
        for (int q = 0; q < codebooks; q++)
        {
            result[q] = new int[frames];
            Array.Copy(delayedTokens[q], q, result[q], 0, frames);
        }
        return result;
    }

    /// <summary>
    /// Returns the delayed INPUT column fed to the transformer to predict target column
    /// <paramref name="step"/> (0-based) -- i.e. the ALREADY-known delayed column `step - 1`
    /// (causal LM: the input at a position predicts the NEXT position's output), or an all-BOS
    /// column when `step == 0` (real MusicGen's `decoder_start_token_id` -- which happens to
    /// equal <paramref name="bosOrPadToken"/> for musicgen-small, since `bos_token_id ==
    /// pad_token_id == 2048` there). Do NOT read `generated[q][step - q]` here -- that is the
    /// value THIS call is trying to help predict, not something already known; the correct
    /// lookback index into codebook `q`'s own real-token stream is `step - 1 - q`.
    /// </summary>
    public static int[] InputColumnForStep(int codebooks, int step, int[][] generated, int bosOrPadToken)
    {
        var column = new int[codebooks];
        for (int q = 0; q < codebooks; q++)
        {
            int localIndex = step - 1 - q;
            column[q] = (localIndex >= 0 && localIndex < generated[q].Length) ? generated[q][localIndex] : bosOrPadToken;
        }
        return column;
    }
}
