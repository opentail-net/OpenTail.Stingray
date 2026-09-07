namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>Real, in-range 16-codebook RVQ audio frame tokens for one generated timestep.</summary>
public readonly struct MossTtsAudioCodes(int frames, int codebooks, int[] tokenIds, bool hitMaxNewFrames)
{
    public int Frames { get; } = frames;
    public int Codebooks { get; } = codebooks;
    /// <summary>Flattened `[frame * Codebooks + codebook]` token ids.</summary>
    public int[] TokenIds { get; } = tokenIds;
    public bool HitMaxNewFrames { get; } = hitMaxNewFrames;
}

/// <summary>
/// Real multi-frame autoregressive generation loop for MOSS-TTS-Nano, ported from
/// `examples/audio.cpp/src/models/moss/moss_tts_nano/generator.cpp`'s
/// `MossTTSNanoGenerator::generate` (not guessed). Real per-step algorithm: recompute the global
/// transformer's last hidden state over the ENTIRE row sequence so far (prompt rows plus every
/// previously generated frame's row -- no incremental KV-cache reuse in the reference either),
/// decode one frame via the local frame decoder, and if that frame was non-empty (the text-side
/// choice was "continue", not "end") append a new row `[audio_assistant_slot_token_id, frame...]`
/// to the sequence and repeat; an empty frame (end-of-content) or `maxNewFrames` stops the loop.
/// </summary>
public static class MossTtsGenerator
{
    public static MossTtsAudioCodes Generate(
        MossTtsGlobalTransformerWeights g,
        MossTtsLocalTransformerWeights l,
        IReadOnlyList<MossTtsGlobalRow> prompt,
        int activeCodebooks,
        int maxNewFrames)
    {
        if (prompt.Count == 0) throw new ArgumentException("MOSS-TTS-Nano generator requires a non-empty prompt.", nameof(prompt));
        if (activeCodebooks <= 0 || activeCodebooks > MossTtsGlobalTransformerWeights.NumCodebooks)
            throw new ArgumentOutOfRangeException(nameof(activeCodebooks));
        if (maxNewFrames <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewFrames));

        var rows = new List<MossTtsGlobalRow>(prompt);
        var generated = new List<int>(maxNewFrames * MossTtsGlobalTransformerWeights.NumCodebooks);
        bool stoppedOnEoc = false;

        for (int step = 0; step < maxNewFrames; step++)
        {
            var hidden = MossTtsGlobalTransformer.ForwardLastHidden(g, rows);
            var frame = MossTtsLocalFrameDecoder.GenerateFrame(g, l, hidden, activeCodebooks);
            if (frame is null)
            {
                stoppedOnEoc = true;
                break;
            }

            generated.AddRange(frame);
            rows.Add(new MossTtsGlobalRow(MossTtsGlobalTransformerWeights.AudioAssistantSlotTokenId, frame));
        }

        if (generated.Count == 0)
            throw new InvalidOperationException("MOSS-TTS-Nano generation produced no audio frames.");

        int frames = generated.Count / MossTtsGlobalTransformerWeights.NumCodebooks;
        bool hitMax = !stoppedOnEoc && frames >= maxNewFrames;
        return new MossTtsAudioCodes(frames, MossTtsGlobalTransformerWeights.NumCodebooks, generated.ToArray(), hitMax);
    }
}
