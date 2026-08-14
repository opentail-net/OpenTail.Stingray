using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Builds a production-equivalent instruction prompt for the Gemma 4 E4B-it model tests.
/// The checkpoint is instruction-tuned: raw continuation tokens are intentionally not used as a
/// coherence oracle because they bypass its GGUF-provided chat template.
/// </summary>
internal static class Gemma4TestPrompt
{
    public static int[] RenderInstruction(GgufModel model, string userMessage)
    {
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var template = tokenizer.ChatTemplate
            ?? throw new InvalidOperationException("Gemma 4 E4B-it fixture must provide a parsed chat template.");

        string rendered = template.Render(new Dictionary<string, object?>
        {
            ["messages"] = JinjaChatTemplate.BuildMessages(userMessage),
            ["add_generation_prompt"] = true,
            ["tools"] = null,
            // Match RunCommand's Gemma 4 default for stock instruction checkpoints.
            ["enable_thinking"] = false,
        });
        var tokens = tokenizer.Encode(rendered).ToArray();
        if (tokens.Length == 0)
            throw new InvalidOperationException("Gemma 4 E4B-it chat template rendered an empty prompt.");
        return tokens;
    }
}
