namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// Real LLaMA-3-8B (llava-llama-3-8b-v1_1 finetune) text-conditioning extraction for HunyuanVideo
/// (docs/088). Confirmed against the real reference
/// (examples/diffusers/.../pipeline_hunyuan_video.py's `_get_llama_prompt_embeds`):
/// wrap the prompt in a fixed literal template (NOT a Jinja chat template -- confirmed by direct
/// grep, this is a hardcoded Python format string, not `tokenizer.apply_chat_template`), tap
/// `hidden_states[-(num_hidden_layers_to_skip + 1)]` with the reference's default
/// `num_hidden_layers_to_skip=2` (= HF's `hidden_states[-3]`), then crop the FIRST `crop_start=95`
/// token positions off the resulting embeddings (the template's own tokens, not the user prompt's)
/// before handing them to the DiT -- get the crop-AFTER-encode convention right or conditioning is
/// silently offset by 95 positions.
/// </summary>
public static class HunyuanVideoTextConditioning
{
    /// <summary>
    /// Real HunyuanVideo default prompt template (`DEFAULT_PROMPT_TEMPLATE["template"]`, verbatim,
    /// confirmed in-repo). `{0}` is the user's raw prompt.
    /// </summary>
    public const string PromptTemplate =
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: " +
        "1. The main content and theme of the video." +
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects." +
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects." +
        "4. background environment, light, style and atmosphere." +
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>" +
        "<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|>";

    /// <summary>Real `DEFAULT_PROMPT_TEMPLATE["crop_start"]` -- token count of the template's own
    /// framing (minus `&lt;|eot_id|&gt;` and the `{}` placeholder), cropped off the front of the
    /// resulting embeddings after encoding.</summary>
    public const int CropStart = 95;

    /// <summary>Real `num_hidden_layers_to_skip=2` default, expressed in this codebase's
    /// `EnableHiddenTaps` "layer i's tap is that layer's own output" convention (= HF's
    /// `hidden_states[i+1]`). HF's `hidden_states[-(2+1)] = hidden_states[-3]`; for LLaMA-3-8B's
    /// 32 layers (33 total hidden_states entries, indices 0..32), index -3 = index 30 = layer 29's
    /// own output (0-indexed) -&gt; `EnableHiddenTaps([29])`.</summary>
    public static readonly int[] TapLayers = [29];

    /// <summary>
    /// Encodes a prompt into HunyuanVideo's real per-token conditioning sequence. Returns a flat
    /// `[nTokens * hiddenDim]` array (already cropped past the template's own tokens) plus the
    /// resulting token count, ready for <see cref="HunyuanVideoPipeline.Generate"/>'s
    /// <c>textContext</c> parameter.
    /// </summary>
    /// <param name="llamaForwardPass">A <see cref="Engine.ForwardPass"/> (or any
    /// <see cref="IForwardPass"/> with <see cref="IForwardPass.SupportsHiddenTaps"/>) over the real
    /// llava-llama-3-8b-v1_1 checkpoint's text tower.</param>
    /// <param name="tokenizer">The same checkpoint's real tokenizer.</param>
    /// <param name="prompt">The raw user prompt (video description request).</param>
    public static (float[] embeds, int nTokens) Encode(IForwardPass llamaForwardPass, GgufTokenizer tokenizer, string prompt)
    {
        if (!llamaForwardPass.SupportsHiddenTaps)
            throw new NotSupportedException("HunyuanVideoTextConditioning.Encode requires a ForwardPass with SupportsHiddenTaps.");

        string rendered = string.Format(PromptTemplate, prompt);
        var tokens = tokenizer.Encode(rendered).ToList();
        if (tokens.Count <= CropStart)
            throw new InvalidOperationException($"HunyuanVideoTextConditioning.Encode: template rendering produced only {tokens.Count} tokens, <= crop_start={CropStart}.");

        llamaForwardPass.EnableHiddenTaps(TapLayers);
        llamaForwardPass.Prefill(tokens);

        int tapDim = llamaForwardPass.HiddenTapDim;
        int nKept = tokens.Count - CropStart;
        var embeds = new float[nKept * tapDim];
        for (int i = 0; i < nKept; i++)
            llamaForwardPass.HiddenTapsAt(CropStart + i).CopyTo(embeds.AsSpan(i * tapDim, tapDim));

        return (embeds, nKept);
    }
}
