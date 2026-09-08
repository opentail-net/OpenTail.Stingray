using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
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
    /// pair, greedy/argmax throughout. Real, deliberate simplification (all 16 codebooks fed for
    /// the same frame, no real delay pattern) kept only for this session's earlier structural
    /// real-weight tests -- <see cref="GenerateDelayed"/> is the real, reference-matching
    /// generation path (real per-stream delay offsets via <see cref="PersonaPlexDelayState"/>,
    /// real temperature/top-k/top-p sampling for both text and Depformer audio codes via
    /// <see cref="SamplingParams"/>), not this method.</summary>
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

        var embedding = new float[hiddenDim];
        var frames = new Frame[numFrames];
        for (int t = 0; t < numFrames; t++)
        {
            BuildEmbedding(embedding, textEmbedding, audioEmbeddings, currentTextToken, currentAudioCodes, hiddenDim, llm.LmCodebooks);

            var textLogits = fwd.ForwardEmbedding(embedding, t);

            int nextTextToken = ArgMax(textLogits, textVocabSize);
            var nextAudioCodes = depformer.GenerateFrame(fwd.LastHidden, nextTextToken, audioCodebookSize);

            frames[t] = new Frame(nextTextToken, nextAudioCodes);
            currentTextToken = nextTextToken;
            currentAudioCodes = nextAudioCodes;
        }
        return frames;
    }

    /// <summary>
    /// Real LIVE-DUPLEX generation, ported from `session.cpp`'s real top-level `run()` loop (not
    /// guessed): `for frame in user_frames: run_user_frame(state, user_codes[frame])` -- the exact
    /// same per-step algorithm as <see cref="GenerateDelayed"/>, except the "user" stream (9-16)
    /// is fed REAL per-frame Mimi codes from <paramref name="userCodesPerFrame"/> (produced by
    /// <see cref="MimiCodecEncoder.Encode"/> on real captured user audio) instead of the fixed
    /// `SilenceTokens` placeholder. One real `PersonaPlexDelayState.Prepare` call per user frame
    /// (matching the reference's real one-`prepare`-call-per-`run_user_frame`-invocation
    /// convention), which may legitimately return `null` for a bootstrap round (no model step that
    /// user frame) -- this is real, expected behavior, not an error, same as `GenerateDelayed`'s
    /// own `if (step is null) continue`.
    /// </summary>
    public static Frame[] GenerateWithUserAudio(IForwardPass fwd, PersonaPlexLmTensorSource llm, PersonaPlexDepformer depformer,
        int[][] userCodesPerFrame, int textVocabSize, int audioCodebookSize,
        SamplingParams? textOptions = null, SamplingParams? audioOptions = null, Random? rng = null)
    {
        int hiddenDim = llm.HiddenDim;
        var textEmbedding = llm.TextEmbeddingWeight();
        var audioEmbeddings = new float[llm.LmCodebooks][];
        for (int cb = 0; cb < llm.LmCodebooks; cb++) audioEmbeddings[cb] = llm.AudioEmbeddingWeight(cb);

        var delayState = new PersonaPlexDelayState();
        var frames = new List<Frame>(userCodesPerFrame.Length);
        var embedding = new float[hiddenDim];
        int position = 0;
        foreach (var userCodes in userCodesPerFrame)
        {
            if (userCodes.Length != 8)
                throw new ArgumentException("Each user-audio frame must carry exactly 8 real Mimi codebook codes.", nameof(userCodesPerFrame));

            var step = delayState.Prepare(userCodes, null, null);
            if (step is null) continue; // real one-time offset==0 bootstrap round, no model step

            BuildEmbedding(embedding, textEmbedding, audioEmbeddings, step.Value.Tokens[0], step.Value.Tokens.AsSpan(1), hiddenDim, llm.LmCodebooks);

            var textLogits = fwd.ForwardEmbedding(embedding, position++);
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);
            int nextText = step.Value.Provided[0] != 0 ? step.Value.Target[0] : sampledText;
            var audioTarget = step.Value.Target[1..];
            var audioProvided = step.Value.Provided[1..];
            var sampledAudio = depformer.GenerateFrame(fwd.LastHidden, nextText, audioCodebookSize, audioOptions, rng, audioTarget, audioProvided);

            var output = delayState.FinishWithSampling(sampledText, sampledAudio);
            if (output != null) frames.Add(new Frame(sampledText, output));
        }
        return [.. frames];
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
        var embedding = new float[hiddenDim];
        int position = 0;
        int guard = 0;
        while (frames.Count < numOutputFrames)
        {
            if (++guard > numOutputFrames * 16 + 64)
                throw new InvalidOperationException("PersonaPlex delay-state loop did not converge.");

            var step = delayState.Prepare(SilenceTokens, null, null);
            if (step is null) continue; // real one-time offset==0 bootstrap round, no model step

            BuildEmbedding(embedding, textEmbedding, audioEmbeddings, step.Value.Tokens[0], step.Value.Tokens.AsSpan(1), hiddenDim, llm.LmCodebooks);

            var textLogits = fwd.ForwardEmbedding(embedding, position++);
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);
            int nextText = step.Value.Provided[0] != 0 ? step.Value.Target[0] : sampledText;
            var audioTarget = step.Value.Target[1..];
            var audioProvided = step.Value.Provided[1..];
            var sampledAudio = depformer.GenerateFrame(fwd.LastHidden, nextText, audioCodebookSize, audioOptions, rng, audioTarget, audioProvided);

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
    /// <para><b>Real system-prompt text</b>: ported from `session.cpp`'s real `wrap_system_prompt`
    /// (not guessed) -- trims whitespace, and unless the trimmed text already starts AND ends with
    /// the literal string `"&lt;system&gt;"`, wraps it as `"&lt;system&gt; " + text + " &lt;system&gt;"`, then
    /// tokenizes via the real per-checkpoint SentencePiece model
    /// (<see cref="PersonaPlexSentencePieceModel"/>, confirmed real UNIGRAM) and steps one token at
    /// a time between the two silence-padding halves, same real always-provided
    /// (`Sine`/`Silence`/thisToken) pattern as the silence frames. Pass `tokenizer: null` only when
    /// `systemPrompt` is empty/whitespace-only (matches the reference's own real
    /// `if (!prompt.empty())` skip) -- a non-null `systemPrompt` with a null `tokenizer` throws.</para>
    /// </summary>
    public static Frame[] GenerateWithVoicePrompt(
        IForwardPass fwd, PersonaPlexLmTensorSource llm, PersonaPlexDepformer depformer,
        PersonaPlexVoicePrompt voicePrompt, float mimiFrameRate, string systemPrompt, UnigramTokenizer? tokenizer,
        int numOutputFrames, int textVocabSize, int audioCodebookSize,
        SamplingParams? textOptions = null, SamplingParams? audioOptions = null, Random? rng = null)
    {
        string wrappedPrompt = WrapSystemPrompt(systemPrompt);
        if (wrappedPrompt.Length > 0 && tokenizer is null)
            throw new ArgumentException("PersonaPlex GenerateWithVoicePrompt: a non-empty systemPrompt requires a tokenizer.", nameof(tokenizer));

        int hiddenDim = llm.HiddenDim;
        var textEmbedding = llm.TextEmbeddingWeight();
        var audioEmbeddings = new float[llm.LmCodebooks][];
        for (int cb = 0; cb < llm.LmCodebooks; cb++) audioEmbeddings[cb] = llm.AudioEmbeddingWeight(cb);

        var delayState = new PersonaPlexDelayState();
        int position = 0;
        var tokenEmbeddingBuffer = new float[hiddenDim];

        // Real always-provided bootstrap step: run the LM forward (embedding either a precomputed
        // voice-prompt frame vector, or built from real forced token ids), then finish with the
        // real forced target values (no real sampling needed -- see this method's doc comment).
        void RunPreparedTokenStep(int[] userTokens, int[] moshiTokens, int textToken)
        {
            var step = delayState.Prepare(userTokens, moshiTokens, textToken);
            if (step is null) return;
            BuildEmbedding(tokenEmbeddingBuffer, textEmbedding, audioEmbeddings, step.Value.Tokens[0], step.Value.Tokens.AsSpan(1), hiddenDim, llm.LmCodebooks);
            var textLogits = fwd.ForwardEmbedding(tokenEmbeddingBuffer, position++);
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);
            delayState.FinishWithSampling(sampledText, step.Value.Target[1..]);
        }

        // 1. Real per-voice-id embedding replay (zero allocations: slices directly into voicePrompt).
        for (int frame = 0; frame < voicePrompt.Frames; frame++)
        {
            var step = delayState.Prepare(InitialAudioTokens, InitialAudioTokens, PersonaPlexDelayState.ZeroTextToken);
            while (step is null) step = delayState.Prepare(InitialAudioTokens, InitialAudioTokens, PersonaPlexDelayState.ZeroTextToken);
            var textLogits = fwd.ForwardEmbedding(voicePrompt.Embeddings.AsSpan((int)((long)frame * hiddenDim), hiddenDim), position++);
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);
            delayState.FinishWithSampling(sampledText, step.Value.Target[1..]);
        }
        delayState.ImportCache(voicePrompt.Cache);

        // 2. Real pre-system-prompt silence padding (`0.5 * mimi.frame_rate` real frames).
        int silenceFrames = (int)(0.5f * mimiFrameRate);
        for (int i = 0; i < silenceFrames; i++)
        {
            RunPreparedTokenStep(SineTokens, SilenceTokens, PersonaPlexDelayState.ZeroTextToken);
        }

        // Real system-prompt section: one real token per step, forced (same always-provided
        // shortcut as the silence frames), Sine/Silence audio padding throughout.
        if (wrappedPrompt.Length > 0)
        {
            var promptTokens = tokenizer!.Encode(wrappedPrompt);
            foreach (int token in promptTokens)
            {
                RunPreparedTokenStep(SineTokens, SilenceTokens, token);
            }
        }

        // 3. Real post-system-prompt silence padding.
        for (int i = 0; i < silenceFrames; i++)
        {
            RunPreparedTokenStep(SineTokens, SilenceTokens, PersonaPlexDelayState.ZeroTextToken);
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

            BuildEmbedding(tokenEmbeddingBuffer, textEmbedding, audioEmbeddings, step.Value.Tokens[0], step.Value.Tokens.AsSpan(1), hiddenDim, llm.LmCodebooks);
            var textLogits = fwd.ForwardEmbedding(tokenEmbeddingBuffer, position++);
            int sampledText = textOptions is null ? ArgMax(textLogits, textVocabSize) : Sampler.Sample(textLogits[..textVocabSize], textOptions, rng);

            int nextText = step.Value.Provided[0] != 0 ? step.Value.Target[0] : sampledText;
            var audioTarget = step.Value.Target[1..];
            var audioProvided = step.Value.Provided[1..];

            var sampledAudio = depformer.GenerateFrame(fwd.LastHidden, nextText, audioCodebookSize, audioOptions, rng, audioTarget, audioProvided);

            var output = delayState.FinishWithSampling(sampledText, sampledAudio);
            if (output != null) frames.Add(new Frame(sampledText, output));
        }
        return [.. frames];
    }

    private static void BuildEmbedding(Span<float> dst, float[] textEmbedding, float[][] audioEmbeddings, int textToken, ReadOnlySpan<int> audioTokens, int hiddenDim, int lmCodebooks)
    {
        textEmbedding.AsSpan((int)((long)textToken * hiddenDim), hiddenDim).CopyTo(dst);
        for (int cb = 0; cb < lmCodebooks; cb++)
        {
            int row = (int)((long)audioTokens[cb] * hiddenDim);
            TensorPrimitives.Add((ReadOnlySpan<float>)dst, audioEmbeddings[cb].AsSpan(row, hiddenDim), dst);
        }
    }

    /// <summary>Real `wrap_system_prompt`: trim whitespace; if empty, return empty; if already
    /// wrapped in literal `"&lt;system&gt;"` markers, return as-is; else wrap as
    /// `"&lt;system&gt; " + text + " &lt;system&gt;"`.</summary>
    private static string WrapSystemPrompt(string text)
    {
        string trimmed = text.Trim();
        if (trimmed.Length == 0) return "";
        if (trimmed.StartsWith("<system>", StringComparison.Ordinal) && trimmed.EndsWith("<system>", StringComparison.Ordinal))
            return trimmed;
        return $"<system> {trimmed} <system>";
    }

    private static int ArgMax(ReadOnlySpan<float> logits, int count) =>
        TensorPrimitives.IndexOfMax(logits[..count]);
}
