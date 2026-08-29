using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Audio.FishSpeech;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Audio.MeloTTS;
using OpenTail.Stingray.Audio.Orpheus;
using OpenTail.Stingray.Audio.Parler;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.QwenTTS;

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

        [CommandOption("-g|--gpu|--backend <BACKEND>")]
        [Description("Compute backend: auto (default), vulkan, or cpu.")]
        public string Backend { get; init; } = "auto";
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
        bool allowGpu = s.Backend.ToLowerInvariant() is not ("cpu" or "0");

        try
        {
            pipeline = engine switch
            {
                "kokoro" => new KokoroPipeline(KokoroModel.Load(
                    ResolveKokoroModelPath(s.ModelPath),
                    ResolveKokoroVoiceFile(s.VoicesDir ?? ResolveKokoroVoicesDir(s.ModelPath), s.Voice))),
                "piper" => s.ModelPath is not null
                    ? PiperPipeline.FromConfigFile(s.ModelPath)
                    : throw new ArgumentException("--model (-m) is required for the piper engine (path to .onnx.json config)."),
                "f5" or "f5tts" or "f5-tts" => s.ModelPath is not null
                    ? F5TtsPipeline.Load(s.ModelPath)
                    : throw new ArgumentException("--model (-m) is required for the f5tts engine (path to .safetensors model file)."),
                "chatterbox" or "chatterbox-turbo" =>
                    ChatterboxPipeline.Load(s.ModelPath ?? ResolveChatterboxModelPath()),
                "melo" or "melotts" => s.ModelPath is not null
                    ? MeloPipeline.Load(s.ModelPath)
                    : throw new ArgumentException("--model (-m) is required for the melo engine (path to model file)."),
                "cosyvoice" or "cosyvoice3" or "cosy" =>
                    CosyVoice3Pipeline.Load(s.ModelPath ?? ResolveCosyVoiceModelPath()),
                "parler" or "parler-tts" or "parlertts" =>
                    ParlerFullPipeline.Load(s.ModelPath ?? ResolveParlerModelPath()),
                "qwen" or "qwentts" or "qwen-tts" or "qwen-talker" =>
                    QwenTtsPipeline.Load(s.ModelPath ?? ResolveQwenTtsModelPath()),
                "fish" or "fishspeech" or "fish-speech" or "s2" or "s2-pro" =>
                    FishSpeechFullPipeline.Load(s.ModelPath ?? ResolveFishSpeechModelPath()),
                "orpheus" or "orpheus-tts" or "orpheustts" =>
                    OrpheusPipeline.Load(s.ModelPath ?? ResolveOrpheusModelPath(), allowGpu: allowGpu),
                _ => throw new ArgumentException($"Unknown TTS engine: '{s.Engine}'. Supported: kokoro, piper, f5tts, chatterbox, melo, cosyvoice, parler, qwentts, fishspeech, orpheus.")
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

    /// <summary>
    /// Real GGUF weights only -- <c>KokoroModel</c>'s parameterless/weights-null constructor is a
    /// pure procedural placeholder synth (see its own doc comment: "Supports both pure simulated/
    /// procedural evaluation and real GGUF weights"), NOT real Kokoro-82M inference. Silently
    /// falling back to it from a bare `stingray tts` invocation with no `-m` produced audible
    /// garbage noise with no error -- always require a real model path, explicit or auto-resolved.
    /// </summary>
    private static string ResolveKokoroModelPath(string? given)
    {
        if (given is not null)
        {
            if (!File.Exists(given))
                throw new ArgumentException($"Kokoro model file not found: '{given}'.");
            return given;
        }

        foreach (var c in new[] { "models/kokoro-82m-q8_0.gguf", "models/kokoro-82m.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No Kokoro model found. Pass --model (-m) with a path to a Kokoro .gguf checkpoint " +
            "(e.g. models/kokoro-82m-q8_0.gguf), or place one at that default path.");
    }

    private static string? ResolveKokoroVoicesDir(string? modelPath)
    {
        string dir = modelPath is not null ? Path.GetDirectoryName(modelPath) ?? "models" : "models";
        return Directory.Exists(dir) ? dir : null;
    }

    /// <summary>
    /// <c>KokoroWeights</c> takes a specific voice FILE path, not a directory -- passing a
    /// directory silently fails its own `File.Exists` check, so no real trained style vector ever
    /// loads and every synthesis falls back to <c>KokoroVoices</c>' procedural placeholder presets
    /// (seeded-random "calibrated initial style vectors", never trained against the real model,
    /// which produces garbled/unintelligible speech). Builds the real `kokoro-voice-{voice}.gguf`
    /// filename and warns (does not fail the whole command) if it's missing, since synthesis still
    /// works with the placeholder preset -- just not with that voice's real trained identity.
    /// </summary>
    private static string? ResolveKokoroVoiceFile(string? voicesDir, string voice)
    {
        if (voicesDir is null) return null;
        string candidate = Path.Combine(voicesDir, $"kokoro-voice-{voice}.gguf");
        if (File.Exists(candidate)) return candidate;

        Console.Error.WriteLine(
            $"Warning: no real voice file for '{voice}' at '{candidate}' -- using a procedural " +
            "placeholder style vector instead of that voice's real trained identity.");
        return null;
    }

    private static string ResolveCosyVoiceModelPath()
    {
        foreach (var c in new[] { "models/cosyvoice3/CosyVoice3-2512_F16.gguf", "models/cosyvoice3/CosyVoice3-2512.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No CosyVoice3 model found. Pass --model (-m) with a path to a CosyVoice .gguf checkpoint " +
            "(e.g. models/cosyvoice3/CosyVoice3-2512_F16.gguf), or place one at that default path.");
    }

    private static string ResolveParlerModelPath()
    {
        foreach (var c in new[] { "models/parler-tts-mini-v1.safetensors", "models/parler-tts-mini-v1-Q8_0.gguf", "models/Parler_TTS_mini.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No Parler-TTS model found. Pass --model (-m) with a path to a Parler .safetensors or .gguf checkpoint " +
            "(e.g. models/parler-tts-mini-v1.safetensors), or place one at that default path.");
    }

    private static string ResolveQwenTtsModelPath()
    {
        foreach (var c in new[] { "models/qwen-talker-0.6b-base-Q8_0.gguf", "models/qwen-talker-0.6b.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No Qwen-Talker model found. Pass --model (-m) with a path to a QwenTalker .gguf checkpoint " +
            "(e.g. models/qwen-talker-0.6b-base-Q8_0.gguf), or place one at that default path.");
    }

    private static string ResolveFishSpeechModelPath()
    {
        foreach (var c in new[] { "models/s2-pro-q4_k_m.gguf", "models/s2-pro-q8_0.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No Fish-Speech (S2-Pro) model found. Pass --model (-m) with a path to an s2-pro .gguf checkpoint " +
            "(e.g. models/s2-pro-q4_k_m.gguf), or place one at that default path.");
    }

    private static string ResolveChatterboxModelPath()
    {
        foreach (var c in new[] { "models/chatterbox-turbo-t3-q4_k.gguf", "models/chatterbox-turbo-t3.gguf", "models/chatterbox_t3.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No Chatterbox model found. Pass --model (-m) with a path to a Chatterbox T3 .gguf checkpoint " +
            "(e.g. models/chatterbox-turbo-t3-q4_k.gguf), or place one at that default path.");
    }

    private static string ResolveOrpheusModelPath()
    {
        foreach (var c in new[] { "models/orpheus-3b-0.1-ft.Q4_K_M.gguf", "models/orpheus-3b.gguf" })
            if (File.Exists(c)) return c;

        throw new ArgumentException(
            "No Orpheus model found. Pass --model (-m) with a path to an Orpheus .gguf checkpoint " +
            "(e.g. models/orpheus-3b-0.1-ft.Q4_K_M.gguf), or place one at that default path.");
    }
}
