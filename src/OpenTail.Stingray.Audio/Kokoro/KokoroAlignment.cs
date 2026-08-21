using System;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// StyleTTS2's length regulator (model.py forward lines 96-103): converts a per-token duration
/// prediction into a one-hot `[T, totalFrames]` alignment matrix via `repeat_interleave` +
/// scatter, then expands token-rate features to frame-rate via `features @ alignment`. Since
/// the matrix is one-hot (exactly one 1 per column -- each output frame belongs to exactly one
/// input token), the matmul is equivalent to, and implemented as, a per-frame gather -- no need
/// to materialize the full sparse matrix.
/// </summary>
public static class KokoroAlignment
{
    /// <summary>
    /// model.py line 96-97: `duration = sigmoid(duration_proj(x)).sum(-1) / speed; pred_dur =
    /// round(duration).clamp(min=1).long()`. durationSums is `KokoroProsodyPredictor.PredictDurations`'s
    /// PRE-divide output.
    /// </summary>
    public static int[] ToPredDur(float[] durationSums, float speed)
    {
        var predDur = new int[durationSums.Length];
        for (int i = 0; i < durationSums.Length; i++)
        {
            float duration = durationSums[i] / speed;
            int rounded = (int)MathF.Round(duration, MidpointRounding.ToEven);
            predDur[i] = Math.Max(rounded, 1);
        }
        return predDur;
    }

    /// <summary>
    /// Builds the frame-rate -> token-index map implied by `repeat_interleave(arange(T), pred_dur)`:
    /// frameToToken[f] = the token index that output frame f belongs to. Length = sum(predDur).
    /// </summary>
    public static int[] BuildFrameToTokenMap(int[] predDur)
    {
        int totalFrames = 0;
        for (int i = 0; i < predDur.Length; i++) totalFrames += predDur[i];

        var map = new int[totalFrames];
        int frame = 0;
        for (int tok = 0; tok < predDur.Length; tok++)
            for (int r = 0; r < predDur[tok]; r++)
                map[frame++] = tok;
        return map;
    }

    /// <summary>
    /// Expands token-rate channel-first features `[channels, T]` to frame-rate `[channels, totalFrames]`
    /// via the frame-to-token map (equivalent to `features @ pred_aln_trg`).
    /// </summary>
    public static float[] Expand(float[] channelFirstSource, int channels, int[] frameToToken)
    {
        int totalFrames = frameToToken.Length;
        int t = channelFirstSource.Length / channels;
        var output = new float[channels * totalFrames];
        for (int c = 0; c < channels; c++)
        {
            int srcRowBase = c * t;
            for (int f = 0; f < totalFrames; f++)
                output[c * totalFrames + f] = channelFirstSource[srcRowBase + frameToToken[f]];
        }
        return output;
    }
}
