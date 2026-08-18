using System.ComponentModel;
using System.Diagnostics;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Cli.CommandLine;

namespace OpenTail.Stingray.Cli;

public sealed class TtsCommand : Command<TtsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|-p|--text|--prompt <TEXT>")]
        [Description("Input text to synthesize into speech audio.")]
        public string? Text { get; init; }

        [CommandOption("-v|--voice <VOICE>")]
        [Description("Voice persona style preset (e.g. af_heart, af_bella, am_adam, bf_alice, bm_george). Default: af_heart.")]
        public string Voice { get; init; } = "af_heart";

        [CommandOption("-s|--speed <SPEED>")]
        [Description("Speech generation speed multiplier. Default: 1.0.")]
        public float Speed { get; init; } = 1.0f;

        [CommandOption("-o|--output <PATH>")]
        [Description("Output WAV file path. Default: speech.wav.")]
        public string OutputPath { get; init; } = "speech.wav";

        [CommandOption("--list-voices")]
        [Description("List all available registered voice styles.")]
        public bool ListVoices { get; init; }
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (s.ListVoices)
        {
            Console.WriteLine("Available Kokoro-82M Voice Presets:");
            foreach (var voice in KokoroVoices.AvailableVoices)
            {
                Console.WriteLine($"  • {voice}");
            }
            return 0;
        }

        if (string.IsNullOrWhiteSpace(s.Text))
        {
            Console.Error.WriteLine("Error: Text prompt is required. Use -t or --text \"Your sentence here\".");
            return 1;
        }

        Console.WriteLine("Kokoro-82M Native Text-to-Speech");
        Console.WriteLine($"Voice:    {s.Voice}");
        Console.WriteLine($"Speed:    {s.Speed:F2}x");
        Console.WriteLine($"Prompt:   \"{s.Text}\"");

        var sw = Stopwatch.StartNew();
        using var pipeline = new KokoroPipeline();

        var req = new AudioGenerationRequest
        {
            Text = s.Text,
            Voice = s.Voice,
            Speed = s.Speed,
            OutputPath = s.OutputPath
        };

        var result = pipeline.Generate(req);
        sw.Stop();

        double audioDuration = result.Duration.TotalSeconds;
        double rtf = sw.Elapsed.TotalSeconds / Math.Max(0.001, audioDuration);

        Console.WriteLine($"✓ Audio synthesized: {Path.GetFullPath(s.OutputPath)}");
        Console.WriteLine($"Duration:       {audioDuration:F2}s ({result.Samples.Length:N0} samples @ {result.SampleRate}Hz)");
        Console.WriteLine($"Inference time: {sw.Elapsed.TotalSeconds:F2}s ({1.0 / Math.Max(0.001, rtf):F1}x real-time)");

        return 0;
    }
}
