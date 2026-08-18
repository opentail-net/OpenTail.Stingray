using System.ComponentModel;
using System.Diagnostics;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Cli.CommandLine;

namespace OpenTail.Stingray.Cli;

public sealed class TtsCommand : Command<TtsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|-p|--text|--prompt <TEXT>")]
        [Description("Input text to synthesize into speech audio.")]
        public string? Text { get; init; }

        [CommandOption("-e|--engine <ENGINE>")]
        [Description("TTS architecture engine: kokoro (default), piper, f5tts, or chatterbox.")]
        public string Engine { get; init; } = "kokoro";

        [CommandOption("-v|--voice <VOICE>")]
        [Description("Voice persona style preset (e.g. af_heart, af_bella, am_adam, resemble_default, narrator). Default: af_heart.")]
        public string Voice { get; init; } = "af_heart";

        [CommandOption("-s|--speed <SPEED>")]
        [Description("Speech generation speed multiplier. Default: 1.0.")]
        public float Speed { get; init; } = 1.0f;

        [CommandOption("-c|--config <PATH>")]
        [Description("Path to optional Piper model config JSON (.onnx.json).")]
        public string? ConfigPath { get; init; }

        [CommandOption("--ref-audio <PATH>")]
        [Description("Path to reference audio file for zero-shot voice cloning (F5-TTS).")]
        public string? ReferenceAudioPath { get; init; }

        [CommandOption("--ref-text <TEXT>")]
        [Description("Reference transcription for the voice cloning audio (F5-TTS).")]
        public string? ReferenceText { get; init; }

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
            Console.WriteLine("Available Voice Presets:");
            Console.WriteLine("  Kokoro-82M:");
            foreach (var voice in KokoroVoices.AvailableVoices) Console.WriteLine($"    • {voice}");
            Console.WriteLine("  Chatterbox-Turbo:");
            foreach (var voice in ChatterboxVoices.AvailableVoices) Console.WriteLine($"    • {voice}");
            return 0;
        }

        if (string.IsNullOrWhiteSpace(s.Text))
        {
            Console.Error.WriteLine("Error: Text prompt is required. Use -t or --text \"Your sentence here\".");
            return 1;
        }

        bool isChatterbox = s.Engine.Contains("chatter", StringComparison.OrdinalIgnoreCase);
        bool isF5 = s.Engine.Contains("f5", StringComparison.OrdinalIgnoreCase) || !string.IsNullOrEmpty(s.ReferenceAudioPath);
        bool isPiper = s.Engine.Equals("piper", StringComparison.OrdinalIgnoreCase) ||
                       (!string.IsNullOrEmpty(s.ConfigPath) && s.ConfigPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase));

        ITextToSpeechPipeline pipeline = isChatterbox
            ? new ChatterboxPipeline()
            : (isF5
                ? new F5TtsPipeline()
                : (isPiper
                    ? (!string.IsNullOrEmpty(s.ConfigPath) ? PiperPipeline.FromConfigFile(s.ConfigPath) : new PiperPipeline())
                    : new KokoroPipeline()));

        using (pipeline)
        {
            Console.WriteLine($"{pipeline.Architecture} Native Text-to-Speech");
            Console.WriteLine($"Voice:    {s.Voice}");
            Console.WriteLine($"Speed:    {s.Speed:F2}x");
            if (!string.IsNullOrEmpty(s.ReferenceAudioPath))
            {
                Console.WriteLine($"Ref Audio: {s.ReferenceAudioPath} (Zero-Shot Voice Cloning)");
            }
            Console.WriteLine($"Prompt:   \"{s.Text}\"");

            var sw = Stopwatch.StartNew();

            AudioGenerationRequest req = isF5
                ? new F5AudioGenerationRequest
                {
                    Text = s.Text,
                    Voice = s.Voice,
                    Speed = s.Speed,
                    OutputPath = s.OutputPath,
                    ReferenceAudioPath = s.ReferenceAudioPath,
                    ReferenceText = s.ReferenceText
                }
                : new AudioGenerationRequest
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
        }

        return 0;
    }
}
