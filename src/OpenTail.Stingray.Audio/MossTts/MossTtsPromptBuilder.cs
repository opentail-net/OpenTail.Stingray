namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// Builds MOSS-TTS-Nano's real zero-shot (no reference/prompt audio) prompt token sequence, ported
/// verbatim from `examples/audio.cpp/src/models/moss/moss_tts_nano/prompt_builder.cpp`'s
/// `MossTTSNanoPromptBuilder::build` (the `prompt_codes == nullptr` branch) -- real prompt template
/// strings (not guessed): a chat-style `user`/`assistant` turn wrapping the target text inside a
/// `&lt;user_inst&gt;...&lt;/user_inst&gt;` block with fixed `Reference(s): None` / `Instruction: None` /
/// etc. fields, followed by `im_end`/`im_start` role-switch tokens and a trailing
/// `audio_start_token_id` that hands control to the audio-generation loop.
///
/// <para><see cref="BuildVoiceClonePrompt"/> is the real `prompt_codes != null` branch (lines
/// 108-133 of the reference): now implementable end-to-end since
/// <see cref="MossTtsAudioCodecEncoder"/> supplies real reference-audio RVQ codes.</para>
/// </summary>
public static class MossTtsPromptBuilder
{
    private const string UserRolePrefix = "user\n";
    private const string UserTemplateReferencePrefix = "<user_inst>\n- Reference(s):\n";
    private const string UserTemplateAfterReference =
        "\n- Instruction:\nNone\n" +
        "- Tokens:\nNone\n" +
        "- Quality:\nNone\n" +
        "- Sound Event:\nNone\n" +
        "- Ambient Sound:\nNone\n" +
        "- Language:\nNone\n" +
        "- Text:\n";
    private const string UserTemplateSuffix = "\n</user_inst>";
    private const string AssistantTurnPrefix = "\n";
    private const string AssistantRolePrefix = "assistant\n";

    /// <summary>
    /// Builds the real zero-shot prompt as a list of already-tokenized text-only rows (no audio ids
    /// at any row -- matches the reference's `MossTokenRowBuilder::push_text_tokens`, which fills
    /// every codebook slot with the pad sentinel for pure-text rows), ending with
    /// <see cref="MossTtsGlobalTransformerWeights.AudioStartTokenId"/> to hand off to generation.
    /// </summary>
    public static List<MossTtsGlobalRow> BuildZeroShotPrompt(SentencePieceBpeTokenizer tokenizer, string text)
    {
        if (string.IsNullOrEmpty(text)) throw new ArgumentException("MOSS-TTS-Nano prompt requires target text.", nameof(text));

        var ids = new List<int> { MossTtsGlobalTransformerWeights.ImStartTokenId };
        ids.AddRange(tokenizer.Encode(UserRolePrefix));
        ids.AddRange(tokenizer.Encode(UserTemplateReferencePrefix));
        ids.AddRange(tokenizer.Encode("None"));
        ids.AddRange(tokenizer.Encode(UserTemplateAfterReference));
        ids.AddRange(tokenizer.Encode(text));
        ids.AddRange(tokenizer.Encode(UserTemplateSuffix));
        ids.Add(MossTtsGlobalTransformerWeights.ImEndTokenId);
        ids.AddRange(tokenizer.Encode(AssistantTurnPrefix));
        ids.Add(MossTtsGlobalTransformerWeights.ImStartTokenId);
        ids.AddRange(tokenizer.Encode(AssistantRolePrefix));
        ids.Add(MossTtsGlobalTransformerWeights.AudioStartTokenId);

        return ToTextOnlyRows(ids);
    }

    /// <summary>
    /// Builds the real voice-cloning prompt (reference audio supplied), ported verbatim from
    /// `prompt_builder.cpp`'s `build()`'s `prompt_codes != nullptr` branch (lines 108-133, not
    /// guessed): `[user-prefix, audioStartTokenId]` (text-only rows) -&gt; one real audio row per
    /// reference-audio frame, each tagged with <see cref="MossTtsGlobalTransformerWeights.
    /// AudioUserSlotTokenId"/> as its TEXT id (matching `push_audio_row`'s real convention: the
    /// text slot carries the role marker, not a pad) and that frame's `NumQuantizers` real codes
    /// as its audio ids -&gt; `[audioEndTokenId, ...same "After Reference" template.../assistant
    /// turn.../audioStartTokenId]` (text-only rows) to hand off to generation.
    /// </summary>
    public static List<MossTtsGlobalRow> BuildVoiceClonePrompt(
        SentencePieceBpeTokenizer tokenizer, string text, int[][] referenceCodesPerQuantizer)
    {
        if (string.IsNullOrEmpty(text)) throw new ArgumentException("MOSS-TTS-Nano prompt requires target text.", nameof(text));
        int numQuantizers = referenceCodesPerQuantizer.Length;
        if (numQuantizers != MossTtsGlobalTransformerWeights.NumCodebooks)
            throw new ArgumentException($"Expected {MossTtsGlobalTransformerWeights.NumCodebooks} reference codebooks, got {numQuantizers}.", nameof(referenceCodesPerQuantizer));
        int frames = referenceCodesPerQuantizer[0].Length;
        if (frames <= 0) throw new ArgumentException("MOSS-TTS-Nano voice-clone prompt requires a non-empty reference code sequence.", nameof(referenceCodesPerQuantizer));

        var prefix = new List<int> { MossTtsGlobalTransformerWeights.ImStartTokenId };
        prefix.AddRange(tokenizer.Encode(UserRolePrefix));
        prefix.AddRange(tokenizer.Encode(UserTemplateReferencePrefix));
        prefix.Add(MossTtsGlobalTransformerWeights.AudioStartTokenId);

        var rows = new List<MossTtsGlobalRow>(prefix.Count + frames + 32);
        rows.AddRange(ToTextOnlyRows(prefix));

        for (int f = 0; f < frames; f++)
        {
            var codes = new int[numQuantizers];
            for (int q = 0; q < numQuantizers; q++) codes[q] = referenceCodesPerQuantizer[q][f];
            rows.Add(new MossTtsGlobalRow(MossTtsGlobalTransformerWeights.AudioUserSlotTokenId, codes));
        }

        var suffix = new List<int> { MossTtsGlobalTransformerWeights.AudioEndTokenId };
        suffix.AddRange(tokenizer.Encode(UserTemplateAfterReference));
        suffix.AddRange(tokenizer.Encode(text));
        suffix.AddRange(tokenizer.Encode(UserTemplateSuffix));
        suffix.Add(MossTtsGlobalTransformerWeights.ImEndTokenId);
        suffix.AddRange(tokenizer.Encode(AssistantTurnPrefix));
        suffix.Add(MossTtsGlobalTransformerWeights.ImStartTokenId);
        suffix.AddRange(tokenizer.Encode(AssistantRolePrefix));
        suffix.Add(MossTtsGlobalTransformerWeights.AudioStartTokenId);
        rows.AddRange(ToTextOnlyRows(suffix));

        return rows;
    }

    private static List<MossTtsGlobalRow> ToTextOnlyRows(List<int> ids)
    {
        var pad = MossTtsGlobalTransformerWeights.AudioPadTokenId;
        var rows = new List<MossTtsGlobalRow>(ids.Count);
        foreach (int id in ids)
            rows.Add(new MossTtsGlobalRow(id, Enumerable.Repeat(pad, MossTtsGlobalTransformerWeights.NumCodebooks).ToArray()));
        return rows;
    }
}
