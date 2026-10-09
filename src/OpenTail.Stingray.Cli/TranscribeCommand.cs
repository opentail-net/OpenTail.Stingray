using System.ComponentModel;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Cli.Terminal;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray transcribe &lt;audio.wav&gt; [-o &lt;out.txt&gt;]</c>: transcribe speech audio using the recommended or specified Whisper model.
/// </summary>
public sealed class TranscribeCommand : Command<TranscribeCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-i|--input <PATH>", Positional = true)]
        [Description("Input 16kHz WAV audio file path for Speech-to-Text transcription.")]
        public string? InputPath { get; init; }

        [CommandOption("-o|--output <PATH>")]
        [Description("Optional output file path to write transcribed text.")]
        public string? OutputPath { get; init; }

        [CommandOption("-m|--model <ID>")]
        [Description("Model catalog id (default: whisper-base).")]
        public string? Model { get; init; }

        [CommandOption("--model-file <PATH>")]
        [Description("Direct path to a Whisper checkpoint (.bin / GGML).")]
        public string? ModelFile { get; init; }

        [CommandOption("-l|--language <LANG>")]
        [Description("Spoken language code (e.g. en, es, fr, de, zh, ja). Default: auto/en.")]
        public string? Language { get; init; }

        [CommandOption("--no-timestamps")]
        [Description("Disable timestamp-aligned subtitle segment generation.")]
        public bool NoTimestamps { get; init; }

        [CommandOption("--vad")]
        [Description("Enable Silero VAD neural speech boundary detection.")]
        public bool UseVad { get; init; }

        [CommandOption("--temperature <TEMP>")]
        [Description("Decoding temperature (0.0 for greedy argmax). Default: 0.0.")]
        [DefaultValue(0.0f)]
        public float Temperature { get; init; } = 0.0f;
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(s.InputPath))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] input audio file is required. Usage: stingray transcribe <audio.wav> [-o <out.txt>]");
            return ExitCodes.Usage;
        }

        if (!File.Exists(s.InputPath))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] audio file not found: {Path.GetFullPath(s.InputPath)}");
            return ExitCodes.NoInput;
        }

        if (!CatalogTaskResolver.TryResolveOrOffer("transcribe", s.Model, s.ModelFile, out var resolved, out string? error, cancellation))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(error ?? "Failed to resolve transcription model.")}");
            return ExitCodes.Usage;
        }

        var sttSettings = new SttCommand.Settings
        {
            InputPath = s.InputPath,
            Model = "base",
            ModelFile = resolved!.ModelPath,
            OutputPath = s.OutputPath,
            Language = s.Language,
            Task = "transcribe",
            NoTimestamps = s.NoTimestamps,
            UseVad = s.UseVad,
            Temperature = s.Temperature
        };

        return new SttCommand().ExecuteInternal(sttSettings, cancellation);
    }
}
