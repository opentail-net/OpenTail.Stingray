using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// Native C# port of MOSS-TTS-Nano's local frame decoder forward pass, from
/// `examples/audio.cpp/src/models/moss/moss_tts_nano/local_frame_decoder.cpp`'s `TextGraph`/
/// `AudioGraph`/`generate_frame` (not guessed). Given one frame's global-transformer hidden state,
/// autoregressively produces: (1) a text-side choice between continuing the assistant audio turn
/// (`audio_assistant_slot_token_id`) or ending it (`audio_end_token_id`) -- an end choice means
/// generation for this utterance is finished; (2) if continuing, that frame's 16 RVQ
/// audio-codebook tokens, one at a time, each conditioned on [global hidden, text embedding, the
/// SAME frame's already-decoded earlier codebooks] through the shared local transformer (RoPE
/// positions reset to 0.. for each per-codebook run, matching the reference's `AudioGraph`, NOT
/// the frame's position in the overall sequence).
///
/// <para><b>Update, 2026-09-07</b>: real temperature/top-k/top-p sampling for the per-codebook
/// audio tokens is now wired via an optional <see cref="SamplingParams"/>, reusing this
/// codebase's existing `OpenTail.Stingray.Engine.Sampler` (same real HF-style formula the
/// reference's own `HfSampler` implements: temperature scaling, top-k, softmax, top-p) --
/// `options: null` (the default everywhere this was already called) preserves the exact previous
/// argmax-only behavior byte-for-byte. The text-side 2-way choice (<see cref="PredictTextChoice"/>)
/// is left as its existing restricted argmax-of-2 -- the reference's real sampling there is a
/// genuinely separate, still-unported "restricted vocabulary" HfSampler call, lower value than
/// the 16-codebook-per-frame audio path this update targets.</para>
/// </summary>
public static class MossTtsLocalFrameDecoder
{
    /// <summary>
    /// Runs the text-side sub-decision for one frame: returns
    /// <see cref="MossTtsGlobalTransformerWeights.AudioAssistantSlotTokenId"/> (continue) or
    /// <see cref="MossTtsGlobalTransformerWeights.AudioEndTokenId"/> (stop), by argmax over just
    /// those two candidate logits -- matches the reference's real 2-way restricted sampling.
    /// </summary>
    public static int PredictTextChoice(
        MossTtsGlobalTransformerWeights g,
        MossTtsLocalTransformerWeights l,
        float[] globalHidden)
    {
        var hiddenOut = MossTtsGlobalTransformer.RunTransformerStack(
            [globalHidden],
            l.Layers,
            l.FinalNormWeight,
            l.FinalNormBias);
        var logits = MossTtsGlobalTransformer.Linear(
            hiddenOut[0], g.TextLmHeadWeight, bias: [],
            MossTtsGlobalTransformerWeights.HiddenDim, MossTtsGlobalTransformerWeights.VocabSize);

        float assistantLogit = logits[MossTtsGlobalTransformerWeights.AudioAssistantSlotTokenId];
        float endLogit = logits[MossTtsGlobalTransformerWeights.AudioEndTokenId];
        return assistantLogit >= endLogit
            ? MossTtsGlobalTransformerWeights.AudioAssistantSlotTokenId
            : MossTtsGlobalTransformerWeights.AudioEndTokenId;
    }

    /// <summary>
    /// Generates one frame's <paramref name="activeCodebooks"/> RVQ tokens (greedy), or returns
    /// null if the text-side choice was to end generation (matches the reference's empty-vector
    /// stop signal).
    /// </summary>
    public static int[]? GenerateFrame(
        MossTtsGlobalTransformerWeights g,
        MossTtsLocalTransformerWeights l,
        float[] globalHidden,
        int activeCodebooks,
        SamplingParams? options = null,
        Random? rng = null)
    {
        if (activeCodebooks <= 0 || activeCodebooks > MossTtsGlobalTransformerWeights.NumCodebooks)
            throw new ArgumentOutOfRangeException(nameof(activeCodebooks));

        int bestText = PredictTextChoice(g, l, globalHidden);
        if (bestText == MossTtsGlobalTransformerWeights.AudioEndTokenId)
            return null;

        var frame = new int[MossTtsGlobalTransformerWeights.NumCodebooks];
        Array.Fill(frame, MossTtsGlobalTransformerWeights.AudioPadTokenId);

        var textEmb = MossTtsGlobalTransformer.EmbedRow(g.TextEmbedding, bestText, MossTtsGlobalTransformerWeights.HiddenDim);

        for (int q = 0; q < activeCodebooks; q++)
        {
            var rows = new float[q + 2][];
            rows[0] = globalHidden;
            rows[1] = textEmb;
            for (int k = 0; k < q; k++)
                rows[2 + k] = MossTtsGlobalTransformer.EmbedRow(g.AudioEmbeddings[k], frame[k], MossTtsGlobalTransformerWeights.HiddenDim);

            var hiddenOut = MossTtsGlobalTransformer.RunTransformerStack(rows, l.Layers, l.FinalNormWeight, l.FinalNormBias);
            var lastHidden = hiddenOut[^1];

            int codebookSize = MossTtsGlobalTransformerWeights.AudioCodebookSize;
            var logits = MossTtsGlobalTransformer.Linear(
                lastHidden, l.AudioLmHeads[q], bias: [],
                MossTtsGlobalTransformerWeights.HiddenDim, codebookSize);

            int best;
            if (options is null)
            {
                best = 0;
                for (int i = 1; i < logits.Length; i++)
                    if (logits[i] > logits[best]) best = i;
            }
            else
            {
                best = Sampler.Sample(logits, options, rng);
            }

            frame[q] = best;
        }

        return frame;
    }
}
