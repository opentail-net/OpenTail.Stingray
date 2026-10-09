using System.ComponentModel;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Cli.Terminal;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray speak &lt;text&gt; [-o &lt;out.wav&gt;]</c>: synthesize speech audio using the recommended or specified voice model.
/// </summary>
public sealed class SpeakCommand : Command<SpeakCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|-p|--text|--prompt <TEXT>", Positional = true)]
        [Description("Input text to synthesize into speech audio.")]
        public string? Text { get; init; }

        [CommandOption("-o|--output <PATH>")]
        [Description("Output destination path (.wav). Default: speech.wav.")]
        public string OutputPath { get; init; } = "speech.wav";

        [CommandOption("-m|--model <ID>")]
        [Description("Model catalog id (default: piper-lessac).")]
        public string? Model { get; init; }

        [CommandOption("--model-file <PATH>")]
        [Description("Direct path to an ONNX voice model file (.onnx) or companion config (.json).")]
        public string? ModelFile { get; init; }

        [CommandOption("-v|--voice <VOICE>")]
        [Description("Voice preset or style name.")]
        public string? Voice { get; init; }

        [CommandOption("-s|--speed <SPEED>")]
        [Description("Speech generation speed multiplier. Default: 1.0.")]
        [DefaultValue(1.0f)]
        public float Speed { get; init; } = 1.0f;

        [CommandOption("-g|--gpu|--backend <BACKEND>")]
        [Description("Compute backend: auto (default), cpu, or vulkan.")]
        public string Backend { get; init; } = "auto";
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(s.Text))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] text is required for speak. Usage: stingray speak <text> [-o <out.wav>]");
            return ExitCodes.Usage;
        }

        if (!CatalogTaskResolver.TryResolve("speak", s.Model, s.ModelFile, out var resolved, out string? error))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(error ?? "Failed to resolve speech model.")}");
            return ExitCodes.Usage;
        }

        var ttsSettings = new TtsCommand.Settings
        {
            Engine = "piper",
            ModelPath = resolved!.ModelPath,
            Text = s.Text,
            OutputPath = s.OutputPath,
            Speed = s.Speed,
            Voice = s.Voice ?? "lessac",
            Backend = s.Backend
        };

        return new TtsCommand().ExecuteInternal(ttsSettings, cancellation);
    }
}
