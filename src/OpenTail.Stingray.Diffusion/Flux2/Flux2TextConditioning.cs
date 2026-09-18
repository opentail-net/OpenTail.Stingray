namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Real Mistral-Small-24B text-conditioning extraction for FLUX.2 (docs/087). Confirmed against
/// the real BFL source (examples/flux2/src/flux2/text_encoder.py's `forward()`): wrap the prompt
/// in a real Jinja chat template (system + user turns, `add_generation_prompt=False`), run the
/// real Mistral forward pass with hidden-state taps at HF's `hidden_states[10, 20, 30]`, and
/// concatenate them per token position (feature-dim concat, NOT averaged/pooled) -- NO cropping of
/// template tokens (unlike Qwen Image's `drop_idx` convention, docs/089 -- confirmed by direct grep,
/// no `crop`/`drop_idx`/`start_idx` exists anywhere in text_encoder.py).
/// </summary>
public static class Flux2TextConditioning
{
    /// <summary>
    /// Real BFL system message (examples/flux2/src/flux2/system_messages.py's `SYSTEM_MESSAGE`,
    /// verbatim, confirmed in-repo).
    /// </summary>
    public const string SystemMessage =
        "You are an AI that reasons about image descriptions. You give structured responses " +
        "focusing on object relationships, object attribution and actions without speculation.";

    /// <summary>
    /// HF's literal `hidden_states[10, 20, 30]` indices, expressed in this codebase's
    /// `EnableHiddenTaps` "layer i's tap is that layer's own output" convention (= HF's
    /// `hidden_states[i+1]`) -- confirmed off-by-one correction, docs/087.
    /// </summary>
    public static readonly int[] TapLayers = [9, 19, 29];

    /// <summary>
    /// Encodes a prompt into FLUX.2's real 15360-dim-per-token conditioning sequence. Returns a
    /// flat `[nTokens * 15360]` array plus the token count, ready for <see cref="Flux2DiT.Forward"/>'s
    /// `textEmbeds`/`textPositions` parameters (caller builds `textPositions` separately, per the
    /// real 4D `(t=0,h=0,w=0,l=sequentialIndex)` scheme -- see `Flux2Pipeline.cs`).
    /// </summary>
    /// <param name="mistralForwardPass">A <see cref="Engine.ForwardPass"/> (or any
    /// <see cref="IForwardPass"/> with <see cref="IForwardPass.SupportsHiddenTaps"/>) over the real
    /// Mistral-Small-3.2-24B checkpoint.</param>
    /// <param name="tokenizer">The same checkpoint's real tokenizer (must expose a non-null
    /// <see cref="GgufTokenizer.ChatTemplate"/>).</param>
    /// <param name="prompt">The raw user prompt (image description request).</param>
    public static (float[] embeds, int nTokens) Encode(IForwardPass mistralForwardPass, GgufTokenizer tokenizer, string prompt)
    {
        if (!mistralForwardPass.SupportsHiddenTaps)
            throw new NotSupportedException("Flux2TextConditioning.Encode requires a ForwardPass with SupportsHiddenTaps.");
        if (tokenizer.ChatTemplate is not { } jinja)
            throw new InvalidOperationException("Flux2TextConditioning.Encode requires the checkpoint's real tokenizer.chat_template metadata (Mistral-Small ships one).");

        var messages = JinjaChatTemplate.BuildMessages(prompt, systemContent: SystemMessage);
        string rendered = jinja.Render(new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["add_generation_prompt"] = false, // real recipe: text_encoder.py's forward() uses False
        });

        var tokens = tokenizer.Encode(rendered).ToList();
        if (tokens.Count == 0) throw new InvalidOperationException("Flux2TextConditioning.Encode: chat-template rendering produced zero tokens.");

        mistralForwardPass.EnableHiddenTaps(TapLayers);
        mistralForwardPass.Prefill(tokens);

        int tapDim = mistralForwardPass.HiddenTapDim;
        var embeds = new float[tokens.Count * tapDim];
        for (int i = 0; i < tokens.Count; i++)
            mistralForwardPass.HiddenTapsAt(i).CopyTo(embeds.AsSpan(i * tapDim, tapDim));

        return (embeds, tokens.Count);
    }
}
