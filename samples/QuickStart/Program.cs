// Quick start: chat, speak and transcribe with OpenTail.Stingray, all in-process.
//
//   dotnet run -- chat  <model.gguf>  "Your question"
//   dotnet run -- speak <voice.onnx>  "Text to say"   out.wav
//   dotnet run -- hear  <ggml-base.bin> in.wav
//
// Models (public downloads):
//   chat:  stingray pull -r Qwen/Qwen2.5-0.5B-Instruct-GGUF             (469 MB)
//   speak: huggingface.co/rhasspy/piper-voices  en/en_US/lessac/medium   (.onnx + .onnx.json, 63 MB)
//   hear:  huggingface.co/ggerganov/whisper.cpp  ggml-base.bin            (148 MB)

using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

switch (args.ElementAtOrDefault(0))
{
    case "chat":
    {
        // Load a GGUF model and its tokenizer, and run it on the CPU.
        using var model = GgufModel.Open(args[1]);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var cpu = new CpuBackend();
        var forward = new ForwardPass(model, cpu, hp, maxContextLength: 4096);
        await using var engine = new InferenceEngine(forward, tokenizer, "quickstart", forward);

        // Format the question with the model's own chat template, then stream the answer.
        string prompt = tokenizer.ChatTemplate!.Render(new Dictionary<string, object?>
        {
            ["messages"] = JinjaChatTemplate.BuildMessages(args[2]),
            ["add_generation_prompt"] = true,
        });
        await foreach (string piece in engine.GenerateAsync(prompt, new SamplingParams { Temperature = 0.7f, MaxNewTokens = 200 }))
            Console.Write(piece);
        Console.WriteLine();
        break;
    }
    case "speak":
    {
        // Piper voice: pass the .onnx.json that sits beside the .onnx.
        using var tts = PiperPipeline.FromConfigFile(args[1] + ".json");
        var result = tts.Generate(new AudioGenerationRequest { Text = args[2], OutputPath = args[3] });
        Console.WriteLine($"Wrote {args[3]} ({result.Duration.TotalSeconds:F1} s of audio)");
        break;
    }
    case "hear":
    {
        using var stt = WhisperPipeline.Load(args[1]);
        var (samples, sampleRate, _) = WavReader.ReadWav(args[2]);
        var result = stt.Transcribe(new SpeechToTextRequest { AudioSamples = samples, SampleRate = sampleRate });
        Console.WriteLine(result.Text);
        break;
    }
    default:
        Console.WriteLine("usage: chat <model.gguf> <question> | speak <voice.onnx> <text> <out.wav> | hear <ggml-base.bin> <in.wav>");
        break;
}
