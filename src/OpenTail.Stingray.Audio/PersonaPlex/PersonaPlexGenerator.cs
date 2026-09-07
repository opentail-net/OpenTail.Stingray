namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Real per-frame generation loop wiring PersonaPlex's temporal LM to its Depformer, ported from
/// `lm_runtime.cpp`'s `build_token_embedding`/`run_token_step` and `session.cpp`'s real initial
/// tokens (not guessed): each frame, the temporal LM's step input is
/// `textEmbedding[frameTextToken] + SUM_over_16(audioEmbeddings[cb][frameAudioCode[cb]])` (plain
/// additive fusion, real per-codebook tables -- confirmed via `build_token_embedding`), producing
/// a hidden state used BOTH to predict the temporal LM's own next TEXT token (via its tied
/// `output.weight`/text logits) AND, together with that just-predicted next text token, to
/// condition the Depformer's generation of the NEXT frame's 16 audio codebook tokens (real
/// Moshi-lineage design: the Depformer fills in one frame's audio codes conditioned on the
/// temporal step that decided that frame's text token).
///
/// <para>Real, confirmed initial/bootstrap tokens (session.cpp's `kTextInitialToken`/
/// `kAudioInitialToken` -- NOT guessed, unlike this session's earlier wrong zero-code bootstrap
/// for Higgs Audio TTS before its real value was found): `kTextInitialToken=32000` (exactly
/// `textVocabSize`, the reserved LAST row of the real `[textVocabSize+1, hidden]` embedding
/// table) and `kAudioInitialToken=2048` (exactly `audioCodebookSize`, same reserved-last-row
/// convention for each `[audioCodebookSize+1, hidden]` audio embedding table).</para>
///
/// <para><b>Update, 2026-09-07</b>: the real staggered multi-stream DELAY pattern flagged above
/// is now implemented -- see <see cref="GenerateDelayed"/>, which drives the real
/// <see cref="PersonaPlexDelayState"/> ring-buffer state machine ported from `session.cpp`. This
/// class's original <see cref="Generate"/> method (all 16 codebooks fed for the same frame, no
/// delay offset) is KEPT as-is since `PersonaPlexFullPipelineRealWeightsTests` already depends on
/// it and it remains a real, valid (if simplified) structural smoke path -- `GenerateDelayed` is
/// the real, delay-correct alternative for new work.</para>
/// </summary>
public static class PersonaPlexGenerator
{
    public const int TextInitialToken = 32000; // real kTextInitialToken == textVocabSize
    public const int AudioInitialToken = 2048; // real kAudioInitialToken == audioCodebookSize

    // Real kSilenceTokens (session.cpp) -- used as a fixed real placeholder for the "user" stream
    // in GenerateDelayed's non-duplex (no live user audio) generation loop, matching the
    // reference's own use of this exact constant during its silence-padding phases.
    public static readonly int[] SilenceTokens = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];

    public readonly struct Frame(int textToken, int[] audioCodes)
    {
        public int TextToken { get; } = textToken;
        public int[] AudioCodes { get; } = audioCodes;
    }

    /// <summary>Generates `numFrames` real frames, each a (text token, 16 audio codebook codes)
    /// pair, greedy/argmax throughout (real sampling not ported, matching this session's other
    /// codebook-sampler stand-ins).</summary>
    public static Frame[] Generate(IForwardPass fwd, PersonaPlexLmTensorSource llm, PersonaPlexDepformer depformer,
        int numFrames, int textVocabSize, int audioCodebookSize)
    {
        int hiddenDim = llm.HiddenDim;
        var textEmbedding = llm.TextEmbeddingWeight();
        var audioEmbeddings = new float[llm.LmCodebooks][];
        for (int cb = 0; cb < llm.LmCodebooks; cb++) audioEmbeddings[cb] = llm.AudioEmbeddingWeight(cb);

        int currentTextToken = TextInitialToken;
        var currentAudioCodes = new int[llm.LmCodebooks];
        Array.Fill(currentAudioCodes, AudioInitialToken);

        var frames = new Frame[numFrames];
        for (int t = 0; t < numFrames; t++)
        {
            var embedding = new float[hiddenDim];
            long textRow = (long)currentTextToken * hiddenDim;
            for (int d = 0; d < hiddenDim; d++) embedding[d] = textEmbedding[textRow + d];
            for (int cb = 0; cb < llm.LmCodebooks; cb++)
            {
                long row = (long)currentAudioCodes[cb] * hiddenDim;
                var table = audioEmbeddings[cb];
                for (int d = 0; d < hiddenDim; d++) embedding[d] += table[row + d];
            }

            var textLogits = fwd.ForwardEmbedding(embedding, t);
            var hidden = fwd.LastHidden.ToArray();

            int nextTextToken = ArgMax(textLogits, textVocabSize);
            var nextAudioCodes = depformer.GenerateFrame(hidden, nextTextToken, audioCodebookSize);

            frames[t] = new Frame(nextTextToken, nextAudioCodes);
            currentTextToken = nextTextToken;
            currentAudioCodes = nextAudioCodes;
        }
        return frames;
    }

    /// <summary>
    /// Real delay-correct generation, ported from `session.cpp`'s `run_user_frame`/
    /// `run_prepared_token_step` (not guessed): drives <see cref="PersonaPlexDelayState"/>'s real
    /// ring-buffer bootstrap/advance logic so each model step's 17-stream input frame
    /// (`step.Tokens`) reflects the real per-stream delay offsets, not a naive same-instant
    /// bundle. Real, deliberate simplification vs. the reference: this is a NON-duplex ("speak
    /// only", no live user audio) loop -- the "user" stream (9-16) is fed the real fixed
    /// `SilenceTokens` constant every step (matching the reference's own silence-padding
    /// convention) rather than genuine live user Mimi codes, and the real voice-prompt/system-
    /// prompt bootstrap sequence (`start_conversation`'s real silence-frame counts, SentencePiece-
    /// tokenized system prompt, and voice-embedding priming) is NOT replayed -- generation starts
    /// straight from `PersonaPlexDelayState`'s own bootstrap step. Text and "moshi" audio are
    /// both self-predicted each step (`moshiTokens: null`, `textToken: null` passed to
    /// `Prepare`), matching `run_user_frame`'s own real `prepare(user_codes, nullptr,
    /// std::nullopt)` call.
    /// </summary>
    public static Frame[] GenerateDelayed(IForwardPass fwd, PersonaPlexLmTensorSource llm, PersonaPlexDepformer depformer,
        int numOutputFrames, int textVocabSize, int audioCodebookSize)
    {
        int hiddenDim = llm.HiddenDim;
        var textEmbedding = llm.TextEmbeddingWeight();
        var audioEmbeddings = new float[llm.LmCodebooks][];
        for (int cb = 0; cb < llm.LmCodebooks; cb++) audioEmbeddings[cb] = llm.AudioEmbeddingWeight(cb);

        var delayState = new PersonaPlexDelayState();
        var frames = new List<Frame>(numOutputFrames);
        int position = 0;
        int guard = 0;
        while (frames.Count < numOutputFrames)
        {
            if (++guard > numOutputFrames * 16 + 64)
                throw new InvalidOperationException("PersonaPlex delay-state loop did not converge.");

            var step = delayState.Prepare(SilenceTokens, null, null);
            if (step is null) continue; // real one-time offset==0 bootstrap round, no model step

            var tokens = step.Value.Tokens;
            var embedding = new float[hiddenDim];
            long textRow = (long)tokens[0] * hiddenDim;
            for (int d = 0; d < hiddenDim; d++) embedding[d] = textEmbedding[textRow + d];
            for (int cb = 0; cb < llm.LmCodebooks; cb++)
            {
                long row = (long)tokens[1 + cb] * hiddenDim;
                var table = audioEmbeddings[cb];
                for (int d = 0; d < hiddenDim; d++) embedding[d] += table[row + d];
            }

            var textLogits = fwd.ForwardEmbedding(embedding, position++);
            var hidden = fwd.LastHidden.ToArray();
            int sampledText = ArgMax(textLogits, textVocabSize);
            var sampledAudio = depformer.GenerateFrame(hidden, sampledText, audioCodebookSize);

            var output = delayState.FinishWithSampling(sampledText, sampledAudio);
            if (output != null) frames.Add(new Frame(sampledText, output));
        }
        return [.. frames];
    }

    private static int ArgMax(ReadOnlySpan<float> logits, int count)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < count; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
