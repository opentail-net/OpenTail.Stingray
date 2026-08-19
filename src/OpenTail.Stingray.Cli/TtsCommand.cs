using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Audio.MeloTTS;
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
        [Description("TTS architecture engine: kokoro (default), piper, f5tts, chatterbox, or melo.")]
        public string Engine { get; init; } = "kokoro";

        [CommandOption("-v|--voice <VOICE>")]
        [Description("Voice persona style preset (e.g. af_heart, af_bella, resemble_default, EN-US, EN-BR, ZH). Default: af_heart.")]
        public string Voice { get; init; } = "af_heart";

        [CommandOption("-s|--speed <SPEED>")]
        [Description("Speech generation speed multiplier. Default: 1.0.")]
        public float Speed { get; init; } = 1.0f;

        [CommandOption("-o|--output <PATH>")]
        [Description("Output destination path (.wav). Default: speech.wav.")]
        public string OutputPath { get; init; } = "speech.wav";

        [CommandOption("-m|--model <PATH>")]
        [Description("Custom model checkpoint path (.gguf, .onnx, or .safetensors).")]
        public string? ModelPath { get; init; }

        [CommandOption("--voices-dir <PATH>")]
        [Description("Kokoro voice directory containing .bin / .gguf voice vectors.")]
        public string? VoicesDir { get; init; }

        [CommandOption("--ref-audio <PATH>")]
        [Description("Reference audio path (.wav) for Zero-Shot Voice Cloning (F5-TTS).")]
        public string? ReferenceAudioPath { get; init; }

        [CommandOption("--ref-text <TEXT>")]
        [Description("Reference audio transcript text for Zero-Shot Voice Cloning (F5-TTS).")]
        public string? ReferenceText { get; init; }

        [CommandOption("--vocab <PATH>")]
        [Description("Custom vocabulary / token file path (F5-TTS vocab.txt).")]
        public string? VocabPath { get; init; }

        [CommandOption("--nfe <NFE>")]
        [Description("Number of Function Evaluations / ODE solver steps for Flow-Matching DiT (default: 32).")]
        public int Nfe { get; init; } = 32;

        [CommandOption("--cfg <CFG>")]
        [Description("Classifier-Free Guidance strength for Flow-Matching DiT (default: 2.0).")]
        public float CfgStrength { get; init; } = 2.0f;
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(s.Text))
        {
            Console.Error.WriteLine("Error: --text (-t) is required for text-to-speech generation.");
            return 1;
        }

        ITextToSpeechPipeline pipeline;
        string engine = s.Engine.ToLowerInvariant();

        try
        {
            pipeline = engine switch
            {
                "kokoro" => KokoroPipeline.Load(s.ModelPath, s.VoicesDir),
                "piper" => PiperPipeline.Load(s.ModelPath),
                "f5" or "f5tts" or "f5-tts" => F5TtsPipeline.Load(
                    modelPath: s.ModelPath,
                    vocabPath: s.VocabPath,
                    odeSteps: s.Nfe,
                    cfgStrength: s.CfgStrength),
                "chatterbox" or "chatterbox-turbo" => ChatterboxPipeline.Load(s.ModelPath),
                "melo" or "melotts" => MeloPipeline.Load(s.ModelPath),
                _ => throw new ArgumentException($"Unknown TTS engine: '{s.Engine}'. Supported: kokoro, piper, f5tts, chatterbox, melo.")
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error initializing TTS pipeline '{s.Engine}': {ex.Message}");
            return 1;
        }

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

            var req = new AudioGenerationRequest
            {
                Text = s.Text,
                Voice = s.Voice,
                Speed = s.Speed,
                OutputPath = s.OutputPath,
                ReferenceAudioPath = s.ReferenceAudioPath,
                ReferenceText = s.ReferenceText
            };

            var result = pipeline.Generate(req);
            sw.Stop();

            double audioDuration = result.Duration.TotalSeconds;
            double rtf = sw.Elapsed.TotalSeconds / Math.Max(0.001, audioDuration);

            Console.WriteLine($"Generated {audioDuration:F2}s audio in {sw.Elapsed.TotalSeconds:F2}s ({rtf:F2}x RTF) -> {s.OutputPath}");
            return 0;
        }
    }
}
