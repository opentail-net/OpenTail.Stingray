using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;

namespace OpenTail.Stingray.Cli;

public sealed class SttCommand : Command<SttCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-i|--input <PATH>")]
        [Description("Input 16kHz WAV audio file path for Speech-to-Text transcription or translation.")]
        public string? InputPath { get; init; }

        [CommandOption("-l|--language <LANG>")]
        [Description("Spoken language code (e.g. en, es, fr, de, zh, ja). Default: auto/en.")]
        public string? Language { get; init; }

        [CommandOption("-t|--task <TASK>")]
        [Description("ASR task: 'transcribe' (default) or 'translate' (translate to English).")]
        public string Task { get; init; } = "transcribe";

        [CommandOption("-m|--model <VARIANT>")]
        [Description("Whisper model architecture preset: tiny (default), base, small, medium, large-v3, or turbo.")]
        public string Model { get; init; } = "tiny";

        [CommandOption("--model-file <PATH>")]
        [Description("Path to a whisper.cpp GGML .bin checkpoint with real weights. If omitted, a file matching --model's preset name is searched for under ./models (e.g. ggml-tiny.bin); if none is found, the pipeline runs with untrained placeholder weights and a warning is printed.")]
        public string? ModelFile { get; init; }

        [CommandOption("--no-timestamps")]
        [Description("Disable timestamp-aligned subtitle segment generation.")]
        public bool NoTimestamps { get; init; }

        [CommandOption("--vad")]
        [Description("Enable Silero VAD neural speech boundary detection and silence filtering.")]
        public bool UseVad { get; init; }

        [CommandOption("--temperature <TEMP>")]
        [Description("Decoding temperature (0.0 for greedy argmax). Default: 0.0.")]
        public float Temperature { get; init; } = 0.0f;

        [CommandOption("-o|--output <PATH>")]
        [Description("Optional output file path to write the transcribed text or subtitle segments.")]
        public string? OutputPath { get; init; }
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(s.InputPath))
        {
            Console.Error.WriteLine("Error: Input audio file is required. Use -i or --input <audio.wav>.");
            return 1;
        }

        if (!File.Exists(s.InputPath))
        {
            Console.Error.WriteLine($"Error: Audio file not found: {Path.GetFullPath(s.InputPath)}");
            return 1;
        }

        SpeechTask task = s.Task.Equals("translate", StringComparison.OrdinalIgnoreCase)
            ? SpeechTask.Translate
            : SpeechTask.Transcribe;

        string variant = s.Model.ToLowerInvariant();
        WhisperConfig fallbackConfig = variant switch
        {
            "base" => WhisperConfig.Base,
            "small" => WhisperConfig.Small,
            "medium" => WhisperConfig.Medium,
            "large" or "large-v3" => WhisperConfig.LargeV3,
            "turbo" or "large-v3-turbo" => WhisperConfig.LargeV3Turbo,
            "distil" or "distil-large-v3" or "distil-whisper" => WhisperConfig.DistilLargeV3,
            _ => WhisperConfig.Tiny
        };

        string? modelFile = s.ModelFile ?? FindDefaultGgmlFile(variant);

        WhisperPipeline pipeline;
        WhisperConfig config;
        if (modelFile is not null)
        {
            pipeline = WhisperPipeline.Load(modelFile);
            config = pipeline.Config;
        }
        else
        {
            Console.Error.WriteLine(
                $"Warning: no GGML weights file found for model '{s.Model}' (looked for ggml-{variant}.bin under ./models). " +
                "Running with untrained placeholder weights -- output will not be meaningful transcription. " +
                "Pass --model-file <path-to-ggml-*.bin> to use a real checkpoint.");
            config = fallbackConfig;
            pipeline = new WhisperPipeline(config);
        }

        using (pipeline)
        {
        Console.WriteLine($"OpenAI Whisper Native Speech-to-Text ({config.AudioState}d / {config.AudioLayer}L)");
        Console.WriteLine($"Input Audio: {Path.GetFullPath(s.InputPath)}");
        Console.WriteLine($"Task:        {task}");
        Console.WriteLine($"Language:    {(string.IsNullOrEmpty(s.Language) ? "auto (en)" : s.Language)}");
        Console.WriteLine($"Timestamps:  {!s.NoTimestamps}");
        Console.WriteLine($"Silero VAD:  {s.UseVad}");

        // Load WAV audio samples
        var (samples, sampleRate, channels) = WavReader.ReadWav(s.InputPath);
        Console.WriteLine($"Loaded {samples.Length:N0} audio samples ({channels} ch @ {sampleRate}Hz, {samples.Length / (double)sampleRate:F2}s)");

        var request = new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = sampleRate,
            Language = s.Language,
            Task = task,
            EnableTimestamps = !s.NoTimestamps,
            UseVad = s.UseVad,
            Temperature = s.Temperature,
            Progress = (done, total) =>
            {
                Console.Write($"\rProcessing audio chunk {done}/{total}...");
            }
        };

        var sw = Stopwatch.StartNew();
        var result = pipeline.Transcribe(request);
        sw.Stop();

        Console.WriteLine("\rProcessing complete!                ");
        Console.WriteLine();
        Console.WriteLine("--- Transcription Result ---");
        Console.WriteLine(result.Text);
        Console.WriteLine("----------------------------");

        if (result.Segments.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Timestamped Segments:");
            foreach (var seg in result.Segments)
            {
                Console.WriteLine($"  [{seg.Start:mm\\:ss\\.ff} --> {seg.End:mm\\:ss\\.ff}] {seg.Text}");
            }
        }

        double audioDuration = result.Duration.TotalSeconds;
        double rtf = sw.Elapsed.TotalSeconds / Math.Max(0.001, audioDuration);
        Console.WriteLine();
        Console.WriteLine($"Audio duration: {audioDuration:F2}s");
        Console.WriteLine($"Inference time: {sw.Elapsed.TotalSeconds:F2}s ({1.0 / Math.Max(0.001, rtf):F1}x real-time)");

        if (!string.IsNullOrEmpty(s.OutputPath))
        {
            File.WriteAllText(s.OutputPath, result.Text);
            Console.WriteLine($"Saved output to: {Path.GetFullPath(s.OutputPath)}");
        }

        return 0;
        }
    }

    private static string? FindDefaultGgmlFile(string variant)
    {
        string fileName = variant switch
        {
            "base" => "ggml-base.bin",
            "small" => "ggml-small.bin",
            "medium" => "ggml-medium.bin",
            "large" or "large-v3" => "ggml-large-v3.bin",
            "turbo" or "large-v3-turbo" => "ggml-large-v3-turbo.bin",
            "distil" or "distil-large-v3" or "distil-whisper" => "ggml-distil-large-v3.bin",
            _ => "ggml-tiny.bin"
        };

        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            string candidate = Path.Combine(dir.FullName, "models", fileName);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }

        return null;
    }
}
