using System.ComponentModel;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Renders a model's chat template against a sample conversation:
/// <c>opentail-llm-cli show-template -m model.gguf</c>
///
/// <para>The read-only slice of §6 Phase 4's compatibility lab. It does NOT run inference, compare
/// against fixtures, or exercise tool-call envelopes — those need a fixture corpus. It answers the
/// one question that precedes all of those: <b>what does this model actually receive?</b></para>
///
/// <para>That question is otherwise unanswerable without running a generation and inferring the
/// formatting backwards from the output. Chat-template mismatches are a leading cause of a model
/// that loads correctly and then behaves badly, because the failure is invisible — the prompt is
/// well-formed, just not what the model was trained on.</para>
/// </summary>
public sealed class ShowTemplateCommand : Command<ShowTemplateCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("Path to a GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("-p|--prompt <TEXT>")]
        [Description("Sample user message (default: a short placeholder)")]
        public string? Prompt { get; init; }

        [CommandOption("--system <TEXT>")]
        [Description("Optional system message to include")]
        public string? System { get; init; }

        [CommandOption("--no-thinking")]
        [Description("Render with enable_thinking = false")]
        public bool NoThinking { get; init; }

        [CommandOption("--raw")]
        [Description("Print the raw Jinja template source instead of a rendered sample")]
        public bool Raw { get; init; }

        public override string? Validate() => ModelPath is null ? "Use -m <model.gguf>." : null;
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        try
        {
            using var model = GgufModel.Open(settings.ModelPath!);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var template = tokenizer.ChatTemplate;

            if (template is null)
            {
                Console.WriteLine("This model has no chat template (tokenizer.chat_template is absent).");
                Console.WriteLine("Prompts are passed through verbatim; a chat-style client will need to");
                Console.WriteLine("supply its own formatting.");
                return 0;   // absence is a fact about the model, not a failure of this command
            }

            if (settings.Raw)
            {
                // The parsed template does not retain its source, so read the original Jinja from
                // GGUF metadata — which is also the more useful thing to show, being byte-for-byte
                // what the model shipped rather than a re-serialisation of our parse of it.
                Console.WriteLine(
                    model.Metadata.TryGetValue("tokenizer.chat_template", out object? raw) && raw is string src
                        ? src
                        : "(the template parsed, but tokenizer.chat_template is not a string in this GGUF)");
                return 0;
            }

            var messages = JinjaChatTemplate.BuildMessages(
                settings.Prompt ?? "Hello!", systemContent: settings.System);

            string rendered = template.Render(new Dictionary<string, object?>
            {
                ["messages"] = messages,
                ["add_generation_prompt"] = true,
                ["tools"] = null,
                ["enable_thinking"] = !settings.NoThinking,
            });

            Console.WriteLine("--- rendered prompt (exactly what the model receives) ---");
            Console.WriteLine(rendered);
            Console.WriteLine("--- end ---");
            Console.WriteLine();
            Console.WriteLine($"{rendered.Length} characters. Use --raw for the template source, "
                              + "--system/--no-thinking to vary the render.");
            return 0;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Template rendering is the point of this command, so a failure here IS the finding:
            // it means the model ships a template this engine cannot execute.
            Console.Error.WriteLine($"show-template: {ex.GetType().Name}: {ex.Message}");
            return 2;
        }
    }
}
