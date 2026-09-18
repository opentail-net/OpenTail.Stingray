namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// Real Qwen2.5-VL-7B text-conditioning extraction for Qwen Image (docs/089). Confirmed against
/// the real diffusers source (`pipeline_qwenimage.py`'s `_get_qwen_prompt_embeds`):
/// a fixed, hand-built ChatML string (NOT the tokenizer's own `apply_chat_template` -- unlike
/// FLUX.2's Mistral recipe), tokenized with the real prompt substituted in, then the FINAL hidden
/// layer only (not FLUX.2's multi-layer concat), with the first 34 template tokens (`drop_idx`)
/// cropped back off before use -- the template's own wrapper tokens are not part of the real
/// per-token conditioning sequence.
/// </summary>
public static class QwenImageTextConditioning
{
    /// <summary>Real prompt template, verbatim from `pipeline_qwenimage.py`'s
    /// `prompt_template_encode`.</summary>
    public const string PromptTemplate =
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, " +
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
        "<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>Real `prompt_template_encode_start_idx` -- the number of leading template tokens
    /// to crop off the final hidden states (confirmed against the real reference, docs/089).</summary>
    public const int DropIdx = 34;

    /// <summary>
    /// Encodes a prompt into Qwen Image's real per-token conditioning sequence: final-layer
    /// hidden states, template tokens cropped off. Returns a flat
    /// `[nTokens * QwenImageModel.ContextDim]` array plus the real (post-crop) token count.
    /// </summary>
    /// <param name="forwardPass">A <see cref="Engine.ForwardPass"/> (or any
    /// <see cref="IForwardPass"/>) over the real Qwen2.5-VL-7B checkpoint (admitted as `qwen2vl`,
    /// text-only, 2026-09-18).</param>
    /// <param name="tokenizer">The same checkpoint's real tokenizer.</param>
    /// <param name="prompt">The raw user prompt (image description request).</param>
    public static (float[] embeds, int nTokens) Encode(IForwardPass forwardPass, GgufTokenizer tokenizer, string prompt)
    {
        string rendered = string.Format(PromptTemplate, prompt);
        var tokens = tokenizer.Encode(rendered).ToList();
        if (tokens.Count <= DropIdx)
            throw new InvalidOperationException($"QwenImageTextConditioning.Encode: rendered prompt tokenized to only {tokens.Count} tokens, need > {DropIdx} (drop_idx) to have any real content left after cropping the template.");

        int embDim = QwenImageModel.ContextDim;
        var allHidden = new float[tokens.Count * embDim];
        forwardPass.ExtractHiddenStates(tokens, allHidden);

        int nTokens = tokens.Count - DropIdx;
        var cropped = new float[nTokens * embDim];
        allHidden.AsSpan(DropIdx * embDim, nTokens * embDim).CopyTo(cropped);

        return (cropped, nTokens);
    }
}
