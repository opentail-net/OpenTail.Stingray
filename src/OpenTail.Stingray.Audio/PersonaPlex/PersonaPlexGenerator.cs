using OpenTail.Stingray.Engine;

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

    // Real kSineTokens (session.cpp) -- the real "user" stream placeholder used specifically
    // during `start_conversation`'s bootstrap padding (distinct from SilenceTokens, which is the
    // "moshi" stream's placeholder there and the "user" stream's placeholder during ordinary
    // non-duplex generation).
    public static readonly int[] SineTokens = [430, 1268, 381, 1611, 1095, 1495, 56, 472];

    // Real kInitialAudioTokens (session.cpp) -- all-AudioInitialToken(2048), the real placeholder
    // used for BOTH user and moshi streams specifically during voice-prompt embedding replay
    // (distinct from SilenceTokens/SineTokens, which are the real silence-padding placeholders).
    public static readonly int[] InitialAudioTokens = [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048];

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
        int numOutputFrames, int textVocabSize, int audioCodebookSize,
        SamplingParams? textOptions = null, SamplingParams? audioOptions = null, Random? rng = null)
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
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);
            var sampledAudio = depformer.GenerateFrame(hidden, sampledText, audioCodebookSize, audioOptions, rng);

            var output = delayState.FinishWithSampling(sampledText, sampledAudio);
            if (output != null) frames.Add(new Frame(sampledText, output));
        }
        return [.. frames];
    }

    /// <summary>
    /// Real voice-id-conditioned generation, ported from `session.cpp`'s `start_conversation` (not
    /// guessed): replays the real per-voice-id bootstrap embeddings, imports the real delay-ring-
    /// buffer cache snapshot, pads with real silence frames, then continues into the ordinary
    /// self-predicting generation loop (<see cref="GenerateDelayed"/>'s own per-step logic).
    ///
    /// <para><b>Real, derived simplification (not guessed)</b>: every bootstrap step -- voice-
    /// prompt embedding replay AND the surrounding silence-padding frames -- supplies EXPLICIT
    /// (non-null) user/moshi/text values to <see cref="PersonaPlexDelayState.Prepare"/>, which
    /// marks every stream `Provided` for that step. Read `depformer.cpp`'s real `run()`: when
    /// EVERY codebook is `Provided`, it short-circuits entirely (`last_unprovided_step &lt; 0`),
    /// returning `audio_target` verbatim with NO model computation (only RNG-advancing if
    /// `do_sample`, a real-but-skippable determinism detail this project's other sampler ports
    /// already treat as an accepted non-bit-exact-RNG gap). Likewise `run_prepared_embedding_step`'s
    /// real `next_text = provided ? target : sampled` always takes the `target` branch here. So
    /// every bootstrap step only needs the real LM forward pass (for correct hidden-state/KV-cache
    /// continuity) -- the "sample text, sample audio via Depformer" work real generation frames do
    /// is providably dead code for this specific always-provided call pattern, not skipped by
    /// guesswork.</para>
    ///
    /// <para><b>Real, deliberate scope limit</b>: the reference's real system-prompt text section
    /// (SentencePiece-tokenized, stepped between the two silence-padding halves) is NOT
    /// implemented -- `systemPrompt` must be empty, matching the reference's own
    /// `if (!prompt.empty())` skip for that case exactly (not a simplification of the empty case,
    /// a real skip already present in the reference for it). A non-empty system prompt throws.</para>
    /// </summary>
    public static Frame[] GenerateWithVoicePrompt(
        IForwardPass fwd, PersonaPlexLmTensorSource llm, PersonaPlexDepformer depformer,
        PersonaPlexVoicePrompt voicePrompt, float mimiFrameRate, string systemPrompt,
        int numOutputFrames, int textVocabSize, int audioCodebookSize,
        SamplingParams? textOptions = null, SamplingParams? audioOptions = null, Random? rng = null)
    {
        if (!string.IsNullOrWhiteSpace(systemPrompt))
            throw new NotSupportedException("PersonaPlex GenerateWithVoicePrompt: non-empty system prompts need SentencePiece tokenization, not yet ported.");

        int hiddenDim = llm.HiddenDim;
        var textEmbedding = llm.TextEmbeddingWeight();
        var audioEmbeddings = new float[llm.LmCodebooks][];
        for (int cb = 0; cb < llm.LmCodebooks; cb++) audioEmbeddings[cb] = llm.AudioEmbeddingWeight(cb);

        var delayState = new PersonaPlexDelayState();
        int position = 0;

        float[] BuildTokenEmbedding(int[] tokens)
        {
            var embedding = new float[hiddenDim];
            long textRow = (long)tokens[0] * hiddenDim;
            for (int d = 0; d < hiddenDim; d++) embedding[d] = textEmbedding[textRow + d];
            for (int cb = 0; cb < llm.LmCodebooks; cb++)
            {
                long row = (long)tokens[1 + cb] * hiddenDim;
                var table = audioEmbeddings[cb];
                for (int d = 0; d < hiddenDim; d++) embedding[d] += table[row + d];
            }
            return embedding;
        }

        // Real always-provided bootstrap step: run the LM forward (embedding either a precomputed
        // voice-prompt frame vector, or built from real forced token ids), then finish with the
        // real forced target values (no real sampling needed -- see this method's doc comment).
        void RunProvidedStep(float[] embedding, int[] userTokens, int[] moshiTokens, int textToken)
        {
            var step = delayState.Prepare(userTokens, moshiTokens, textToken);
            while (step is null) step = delayState.Prepare(userTokens, moshiTokens, textToken);
            fwd.ForwardEmbedding(embedding, position++);
            delayState.FinishWithSampling(step.Value.Target[0], step.Value.Target[1..]);
        }

        // 1. Real per-voice-id embedding replay.
        for (int frame = 0; frame < voicePrompt.Frames; frame++)
        {
            var embedding = new float[hiddenDim];
            Array.Copy(voicePrompt.Embeddings, (long)frame * hiddenDim, embedding, 0, hiddenDim);
            RunProvidedStep(embedding, InitialAudioTokens, InitialAudioTokens, PersonaPlexDelayState.ZeroTextToken);
        }
        delayState.ImportCache(voicePrompt.Cache);

        // 2. Real pre-system-prompt silence padding (`0.5 * mimi.frame_rate` real frames).
        int silenceFrames = (int)(0.5f * mimiFrameRate);
        for (int i = 0; i < silenceFrames; i++)
        {
            var tokens = new int[PersonaPlexDelayState.NumStreams];
            tokens[0] = PersonaPlexDelayState.ZeroTextToken;
            for (int cb = 0; cb < 8; cb++) tokens[1 + cb] = SilenceTokens[cb];
            for (int cb = 0; cb < 8; cb++) tokens[9 + cb] = SineTokens[cb];
            RunProvidedStep(BuildTokenEmbedding(tokens), SineTokens, SilenceTokens, PersonaPlexDelayState.ZeroTextToken);
        }

        // Real system-prompt section skipped (empty prompt only, see doc comment).

        // 3. Real post-system-prompt silence padding.
        for (int i = 0; i < silenceFrames; i++)
        {
            var tokens = new int[PersonaPlexDelayState.NumStreams];
            tokens[0] = PersonaPlexDelayState.ZeroTextToken;
            for (int cb = 0; cb < 8; cb++) tokens[1 + cb] = SilenceTokens[cb];
            for (int cb = 0; cb < 8; cb++) tokens[9 + cb] = SineTokens[cb];
            RunProvidedStep(BuildTokenEmbedding(tokens), SineTokens, SilenceTokens, PersonaPlexDelayState.ZeroTextToken);
        }

        // 4. Real ordinary self-predicting generation loop, continuing from the bootstrapped state.
        var frames = new List<Frame>(numOutputFrames);
        int guard = 0;
        while (frames.Count < numOutputFrames)
        {
            if (++guard > numOutputFrames * 16 + 64)
                throw new InvalidOperationException("PersonaPlex delay-state loop did not converge.");

            var step = delayState.Prepare(SilenceTokens, null, null);
            if (step is null) continue;

            var embedding = BuildTokenEmbedding(step.Value.Tokens);
            var textLogits = fwd.ForwardEmbedding(embedding, position++);
            var hidden = fwd.LastHidden.ToArray();
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);
            var sampledAudio = depformer.GenerateFrame(hidden, sampledText, audioCodebookSize, audioOptions, rng);

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
