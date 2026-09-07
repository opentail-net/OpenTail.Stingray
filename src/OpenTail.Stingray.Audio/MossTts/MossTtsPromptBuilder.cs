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
/// <para><b>Not yet implemented</b>: the reference-audio continuation path (`prompt_codes != null`
/// in the reference -- voice cloning against a real reference audio clip's already-encoded RVQ
/// codes) -- a real, separate feature needing the audio codec (not yet ported) to produce those
/// reference codes in the first place. This class covers the reference's zero-shot path only.</para>
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

        var pad = MossTtsGlobalTransformerWeights.AudioPadTokenId;
        var rows = new List<MossTtsGlobalRow>(ids.Count);
        foreach (int id in ids)
            rows.Add(new MossTtsGlobalRow(id, Enumerable.Repeat(pad, MossTtsGlobalTransformerWeights.NumCodebooks).ToArray()));
        return rows;
    }
}
