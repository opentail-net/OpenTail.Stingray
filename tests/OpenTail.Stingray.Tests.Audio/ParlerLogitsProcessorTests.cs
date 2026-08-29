
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Golden verification for <see cref="ParlerLogitsProcessor"/> against a real trace of the real
/// `parler_tts.logits_processors.ParlerTTSLogitsProcessor` (run directly via PyTorch, 4
/// codebooks, single batch item, eos_token_id=5 -- see the exact trace this test replicates in
/// docs/audio-review-progress.md's Parler-TTS generation-loop section). Confirms the cascading
/// EOS-unlock order: only codebook 0 may emit EOS until it actually has, then codebook 1 unlocks,
/// then codebook 2, then codebook 3 (the last codebook stays unlocked once the pointer reaches it,
/// matching the real `first_codebooks_unfinished &lt; max_codebooks` guard).
/// </summary>
public sealed class ParlerLogitsProcessorTests
{
    private const int EosId = 5;
    private const int NumCodebooks = 4;
    private const int Vocab = 6;

    [Fact]
    public void Apply_RealPyTorchTrace_MatchesGoldenEosBlockingSequence()
    {
        var proc = new ParlerLogitsProcessor(EosId, NumCodebooks);
        var history = new List<int>[NumCodebooks];
        for (int cb = 0; cb < NumCodebooks; cb++) history[cb] = [];

        // (new tokens to append AFTER this step's check, expected eos-column-blocked pattern BEFORE appending)
        var steps = new (int[] NewTokens, bool[] ExpectedBlocked)[]
        {
            ([10, 10, 10, 10],        [false, true, true, true]),
            ([EosId, 10, 10, 10],     [false, true, true, true]),
            ([EosId, 10, 10, 10],     [false, false, true, true]),
            ([EosId, EosId, 10, 10],  [false, false, true, true]),
            ([EosId, EosId, 10, 10],  [false, false, false, true]),
            ([EosId, EosId, EosId, 10], [false, false, false, true]),
            ([EosId, EosId, EosId, EosId], [false, false, false, false]),
            ([EosId, EosId, EosId, EosId], [false, false, false, false]),
        };

        foreach (var (newTokens, expectedBlocked) in steps)
        {
            var historyArrays = new int[NumCodebooks][];
            for (int cb = 0; cb < NumCodebooks; cb++) historyArrays[cb] = [.. history[cb]];

            var scores = new float[NumCodebooks][];
            for (int cb = 0; cb < NumCodebooks; cb++) scores[cb] = new float[Vocab];

            proc.Apply(historyArrays, scores);

            for (int cb = 0; cb < NumCodebooks; cb++)
            {
                bool blocked = float.IsNegativeInfinity(scores[cb][EosId]);
                Assert.True(blocked == expectedBlocked[cb], $"codebook {cb}: expected blocked={expectedBlocked[cb]}, got {blocked}");
            }

            for (int cb = 0; cb < NumCodebooks; cb++) history[cb].Add(newTokens[cb]);
        }
    }
}
