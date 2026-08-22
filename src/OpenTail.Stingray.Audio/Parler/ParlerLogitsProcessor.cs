namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real Parler-TTS EOS/stopping logic, transcribed directly from the real `parler_tts` package's
/// `ParlerTTSLogitsProcessor` (`logits_processors.py`, from the already-local
/// `scratch-llamacpp-ref/parler-pkg/parler_tts-0.2.3` source -- NOT guessed, per the earlier
/// entry's explicit flag that this needed direct source verification). Single-batch (bsz=1) form
/// only -- this engine generates one sequence at a time, so the real per-batch-item vectorized
/// bookkeeping collapses to a single scalar `_firstUnfinished` codebook pointer.
///
/// <para><b>Real algorithm</b>: because of the delay pattern (see <see cref="ParlerDelayPattern"/>),
/// codebook 0 reaches its own real end-of-audio position before codebook 1 does, which reaches it
/// before codebook 2, etc. -- so codebook streams must be allowed to emit EOS in that same
/// cascading order, not independently. A single pointer `_firstUnfinished` (initially codebook 0)
/// tracks which codebook is currently allowed to end; every step, if that codebook's OWN generated
/// history so far already contains an EOS token AND it isn't the last codebook, the pointer
/// advances to the next codebook. Every codebook's EOS logit is forced to `-infinity` UNLESS its
/// index is `&lt;= _firstUnfinished` (already-finished codebooks stay allowed too, matching the
/// real `codebook_idx &gt; first_codebooks_unfinished` strict-greater mask).</para>
/// </summary>
public sealed class ParlerLogitsProcessor
{
    private readonly int _numCodebooks;
    private readonly int _eosTokenId;
    private int _firstUnfinished;

    public ParlerLogitsProcessor(int eosTokenId, int numCodebooks)
    {
        _eosTokenId = eosTokenId;
        _numCodebooks = numCodebooks;
        _firstUnfinished = 0;
    }

    /// <summary>
    /// Real per-step update + mask application. <paramref name="historyPerCodebook"/> is each
    /// codebook's own generated token history so far (NOT including this step's not-yet-chosen
    /// token). <paramref name="scoresPerCodebook"/>'s EOS logit is set to
    /// <see cref="float.NegativeInfinity"/> in place for every codebook whose index exceeds the
    /// current unfinished pointer.
    /// </summary>
    public void Apply(int[][] historyPerCodebook, float[][] scoresPerCodebook)
    {
        bool firstUnfinishedHasEos = Contains(historyPerCodebook[_firstUnfinished], _eosTokenId);
        if (firstUnfinishedHasEos && _firstUnfinished < _numCodebooks - 1) _firstUnfinished++;

        for (int cb = 0; cb < _numCodebooks; cb++)
            if (cb > _firstUnfinished) scoresPerCodebook[cb][_eosTokenId] = float.NegativeInfinity;
    }

    private static bool Contains(int[] history, int value)
    {
        foreach (var v in history) if (v == value) return true;
        return false;
    }
}
