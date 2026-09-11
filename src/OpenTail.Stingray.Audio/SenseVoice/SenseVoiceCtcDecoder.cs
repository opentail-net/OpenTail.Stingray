namespace OpenTail.Stingray.Audio.SenseVoice;

/// <summary>
/// Real greedy CTC decode for SenseVoice, ported verbatim from sherpa-onnx's real
/// `OfflineCtcGreedySearchDecoder::Decode` (`offline-ctc-greedy-search-decoder.cc`): per-frame
/// argmax over the vocab dimension, then collapse consecutive repeated ids and drop `blank_id`
/// (standard CTC decode -- note the check is `y != blank_id_ &amp;&amp; y != prev_id`, i.e. a
/// blank frame still updates `prev_id`, so `blank, a, blank, a` decodes to two `a`s, not one).
/// </summary>
public static class SenseVoiceCtcDecoder
{
    /// <summary>`logitsFlat` is row-major `[numFrames, vocabSize]`.</summary>
    public static int[] GreedyDecode(float[] logitsFlat, int numFrames, int vocabSize, int blankId)
    {
        var ids = new List<int>(numFrames);
        int prevId = -1;
        for (int t = 0; t < numFrames; t++)
        {
            int baseIdx = t * vocabSize;
            int best = 0;
            float bestValue = logitsFlat[baseIdx];
            for (int c = 1; c < vocabSize; c++)
            {
                float v = logitsFlat[baseIdx + c];
                if (v > bestValue) { bestValue = v; best = c; }
            }

            if (best != blankId && best != prevId)
                ids.Add(best);
            prevId = best;
        }
        return [.. ids];
    }
}
