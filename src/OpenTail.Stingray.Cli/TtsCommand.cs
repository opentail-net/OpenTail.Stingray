using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Audio.FishSpeech;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Audio.MeloTTS;
using OpenTail.Stingray.Audio.MmsTts;
using OpenTail.Stingray.Audio.Orpheus;
using OpenTail.Stingray.Audio.Parler;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Cli;

public sealed class TtsCommand : Command<TtsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|-p|--text|--prompt <TEXT>")]
        [Description("Input text to synthesize into speech audio.")]
        public string? Text { get; init; }

        [CommandOption("-e|--engine <ENGINE>")]
        [Description("TTS engine: kokoro (default), piper, f5tts, chatterbox, melo, cosyvoice, parler, qwentts, fishspeech, orpheus, mms, xtts.")]
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

        [CommandOption("--seed <N>")]
        [Description("RNG seed for engines that sample (fishspeech, parler, qwentts, xtts, mms, cosyvoice) so two runs are comparable. Default: each engine's own default (fishspeech, xtts and mms are random). Other engines ignore it.")]
        public int? Seed { get; init; }

        [CommandOption("-g|--gpu|--backend <BACKEND>")]
        [Description("Compute backend: auto (default), vulkan, or cpu.")]
        public string Backend { get; init; } = "auto";
    }

    // Default checkpoint locations, searched when --model is not given. Preference order matters: for Fish Speech the
    // Q8_0 file comes first because its Fast-AR stage is essentially the original model while Q4_K_M is a materially
    // different distribution (docs/1-correctness/08, 2026-10-01).
    private static readonly ModelSearch KokoroSearch = new("Kokoro", "a Kokoro .gguf checkpoint", "models/kokoro-82m-q8_0.gguf",
        ["models/kokoro-82m-q8_0.gguf", "models/kokoro-82m.gguf"]);
    private static readonly ModelSearch ChatterboxSearch = new("Chatterbox", "a Chatterbox T3 .gguf checkpoint", "models/chatterbox-turbo-t3-q4_k.gguf",
        ["models/chatterbox-turbo-t3-q4_k.gguf", "models/chatterbox-turbo-t3.gguf", "models/chatterbox_t3.gguf"]);
    private static readonly ModelSearch CosyVoiceSearch = new("CosyVoice3", "a CosyVoice .gguf checkpoint", "models/cosyvoice3/CosyVoice3-2512_F16.gguf",
        ["models/cosyvoice3/CosyVoice3-2512_F16.gguf", "models/cosyvoice3/CosyVoice3-2512.gguf"]);
    private static readonly ModelSearch ParlerSearch = new("Parler-TTS", "a Parler .safetensors or .gguf checkpoint", "models/parler-tts-mini-v1.safetensors",
        ["models/parler-tts-mini-v1.safetensors", "models/parler-tts-mini-v1-Q8_0.gguf", "models/Parler_TTS_mini.gguf"]);
    private static readonly ModelSearch QwenTtsSearch = new("Qwen-Talker", "a QwenTalker .gguf checkpoint", "models/qwen-talker-0.6b-base-Q8_0.gguf",
        ["models/qwen-talker-0.6b-base-Q8_0.gguf", "models/qwen-talker-0.6b.gguf"]);
    private static readonly ModelSearch FishSpeechSearch = new("Fish-Speech (S2-Pro)", "an s2-pro .gguf checkpoint", "models/s2-pro-q8_0.gguf",
        ["models/s2-pro-q8_0.gguf", "models/s2-pro-q4_k_m.gguf"]);
    private static readonly ModelSearch OrpheusSearch = new("Orpheus", "an Orpheus .gguf checkpoint", "models/orpheus-3b-0.1-ft.Q4_K_M.gguf",
        ["models/orpheus-3b-0.1-ft.Q4_K_M.gguf", "models/orpheus-3b.gguf"]);
    private static readonly ModelSearch MmsTtsSearch = new("MMS-TTS", "an MMS-TTS checkpoint directory", "models/mms-tts-eng",
        ["models/mms-tts-eng"], DirMarker: "model.safetensors",
        Extra: "The directory must contain config.json/vocab.json/model.safetensors (from huggingface.co/facebook/mms-tts-<lang>).");
    private static readonly ModelSearch XttsSearch = new("XTTS-v2", "an XTTS-v2 checkpoint directory", "models/xtts-v2",
        ["models/xtts-v2"], DirMarker: "model.safetensors",
        Extra: "The directory must contain vocab.json/model.safetensors/mel_stats.safetensors (converted from huggingface.co/coqui/XTTS-v2's model.pth via scratch-llamacpp-ref/xtts_convert_to_safetensors.py). XTTS-v2 also requires --ref-audio (a real voice-cloning source clip).");

    internal int ExecuteInternal(Settings s, CancellationToken cancellation = default) => Execute(s, cancellation);

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(s.Text))
        {
            Console.Error.WriteLine("Error: --text (-t) is required for text-to-speech generation.");
            return ExitCodes.Usage;
        }

        string? engine = TtsEngines.Canonical(s.Engine);
        if (engine is null)
        {
            Console.Error.WriteLine($"Error: unknown TTS engine '{s.Engine}'. Supported: {TtsEngines.Names}.");
            return ExitCodes.Usage;
        }

        bool allowGpu = s.Backend.ToLowerInvariant() is not ("cpu" or "0");

        // Explicit opt-in only ("vulkan", not "auto") -- the CFM UNet's real, measured Vulkan
        // perf is a mixed bag (small attention-width matmuls lose to CPU, the wider FFN matmul
        // wins), so don't silently default existing users onto an unproven-net path the way
        // "auto" does for other engines' allowGpu. Left as an explicit --backend vulkan choice.
        Core.IComputeBackend? gpuBackend = null;
        if (s.Backend.ToLowerInvariant() == "vulkan")
        {
            try
            {
                gpuBackend = new Vulkan.VulkanBackend(0);
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[yellow]Note:[/] Vulkan GPU init failed ({ex.Message}); falling back to CPU.");
            }
        }

        ITextToSpeechPipeline pipeline;
        try
        {
            pipeline = engine switch
            {
                "kokoro" => new KokoroPipeline(KokoroModel.Load(
                    ResolveKokoroModelPath(s.ModelPath),
                    ResolveKokoroVoiceFile(s.VoicesDir ?? ResolveKokoroVoicesDir(s.ModelPath), s.Voice))),
                // Accept the voice's .onnx (what people naturally pass) as well as its .onnx.json
                // config, which piper-voices always ships beside it.
                "piper" => s.ModelPath is not null
                    ? PiperPipeline.FromConfigFile(
                        s.ModelPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) && File.Exists(s.ModelPath + ".json")
                            ? s.ModelPath + ".json" : s.ModelPath)
                    : throw new ArgumentException("--model (-m) is required for the piper engine (the voice's .onnx, with its .onnx.json beside it)."),
                "f5tts" => s.ModelPath is not null
                    ? F5TtsPipeline.Load(s.ModelPath, backend: gpuBackend)
                    : throw new ArgumentException("--model (-m) is required for the f5tts engine (path to .safetensors model file)."),
                "chatterbox" => ChatterboxPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(ChatterboxSearch), backend: gpuBackend),
                "melo" => s.ModelPath is not null
                    ? MeloPipeline.Load(s.ModelPath)
                    : throw new ArgumentException("--model (-m) is required for the melo engine (path to model file)."),
                "cosyvoice" => CosyVoice3Pipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(CosyVoiceSearch), backend: gpuBackend),
                "parler" => ParlerFullPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(ParlerSearch), backend: gpuBackend),
                "qwentts" => QwenTtsPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(QwenTtsSearch)),
                "fishspeech" => FishSpeechFullPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(FishSpeechSearch)),
                "orpheus" => OrpheusPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(OrpheusSearch), allowGpu: allowGpu),
                "mms" => MmsTtsPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(MmsTtsSearch)),
                "xtts" => XttsPipeline.Load(s.ModelPath ?? ModelPathResolver.Resolve(XttsSearch)),
                _ => throw new ArgumentException($"Unknown TTS engine: '{s.Engine}'. Supported: {TtsEngines.Names}.")
            };
        }
        catch (ArgumentException ex)
        {
            // A missing/invalid model path or option is a usage problem the user can fix by changing the invocation.
            Console.Error.WriteLine($"Error initializing TTS pipeline '{s.Engine}': {ex.Message}");
            return ExitCodes.Usage;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error initializing TTS pipeline '{s.Engine}': {ex.Message}");
            return ExitCodes.Failure;
        }

        if (s.Seed is not null && !TtsEngines.HonorsSeed(engine))
            Console.Error.WriteLine($"Note: --seed has no effect on the '{engine}' engine.");

        using (pipeline)
        {
            Console.WriteLine($"{pipeline.Architecture} Native Text-to-Speech");
            Console.WriteLine($"Voice:    {s.Voice}");
            Console.WriteLine($"Speed:    {s.Speed:F2}x");
            if (s.Seed is not null)
            {
                Console.WriteLine($"Seed:     {s.Seed}");
            }
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
                ReferenceText = s.ReferenceText,
                Seed = s.Seed,
            };

            var result = pipeline.Generate(req);
            sw.Stop();

            double audioDuration = result.Duration.TotalSeconds;
            double rtf = sw.Elapsed.TotalSeconds / Math.Max(0.001, audioDuration);

            Console.WriteLine($"Generated {audioDuration:F2}s audio in {sw.Elapsed.TotalSeconds:F2}s ({rtf:F2}x RTF) -> {s.OutputPath}");
            return ExitCodes.Success;
        }
    }

    /// <summary>
    /// Real GGUF weights only -- <c>KokoroModel</c>'s parameterless/weights-null constructor is a
    /// pure procedural placeholder synth (see its own doc comment: "Supports both pure simulated/
    /// procedural evaluation and real GGUF weights"), NOT real Kokoro-82M inference. Silently
    /// falling back to it from a bare `stingray tts` invocation with no `-m` produced audible
    /// garbage noise with no error -- always require a real model path, explicit or auto-resolved.
    /// </summary>
    private static string ResolveKokoroModelPath(string? given) =>
        given is not null ? ModelPathResolver.RequireExisting("Kokoro", given) : ModelPathResolver.Resolve(KokoroSearch);

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
}
