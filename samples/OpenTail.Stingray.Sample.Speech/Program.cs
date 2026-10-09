using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Core.Catalog;

// OpenTail.Stingray library sample: Text-to-Speech (TTS) and Speech-to-Text (ASR)
//
//   dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release
//   dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release -- --speak "Hello from .NET" -o out.wav
//   dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release -- --transcribe out.wav

string? speakText = null;
string? transcribePath = null;
string outputPath = "sample_speech.wav";
string? voiceConfigPath = null;
string? whisperModelPath = null;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--speak" when i + 1 < args.Length:
            speakText = args[++i]; break;
        case "-o" or "--output" when i + 1 < args.Length:
            outputPath = args[++i]; break;
        case "--transcribe" when i + 1 < args.Length:
            transcribePath = args[++i]; break;
        case "--voice" when i + 1 < args.Length:
            voiceConfigPath = args[++i]; break;
        case "--whisper" when i + 1 < args.Length:
            whisperModelPath = args[++i]; break;
        case "-h" or "--help":
            Console.WriteLine("usage: opentail-speech-sample [--speak \"text\"] [-o output.wav] [--transcribe input.wav]");
            return 0;
    }
}

// If neither --speak nor --transcribe was specified, perform a combined roundtrip
bool doCombinedRoundtrip = speakText is null && transcribePath is null;

// Resolve voice config (.onnx.json)
voiceConfigPath ??= ResolveVoiceConfig();
// Resolve whisper model (.bin)
whisperModelPath ??= ResolveWhisperModel();

if (doCombinedRoundtrip || speakText != null)
{
    string text = speakText ?? "OpenTail Stingray brings fast neural voice synthesis and transcription directly to managed .NET.";
    if (voiceConfigPath is null || !File.Exists(voiceConfigPath))
    {
        Console.Error.WriteLine("Error: Piper voice model not found.");
        Console.Error.WriteLine("Run 'stingray setup speak' or pass --voice <path-to-.onnx.json>");
        if (!doCombinedRoundtrip) return 1;
    }
    else
    {
        Console.WriteLine($"[TTS] Synthesizing speech with Piper...");
        Console.WriteLine($"      Text: \"{text}\"");
        using var tts = PiperPipeline.FromConfigFile(voiceConfigPath);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var ttsResult = tts.Generate(new AudioGenerationRequest
        {
            Text = text,
            OutputPath = outputPath
        });
        sw.Stop();
        Console.WriteLine($"      Generated {ttsResult.Duration.TotalSeconds:F2}s of audio in {sw.ElapsedMilliseconds}ms -> {outputPath}");
    }
}

if (doCombinedRoundtrip || transcribePath != null)
{
    string wavToTranscribe = transcribePath ?? outputPath;
    if (whisperModelPath is null || !File.Exists(whisperModelPath))
    {
        Console.Error.WriteLine("Error: Whisper model not found.");
        Console.Error.WriteLine("Run 'stingray setup transcribe' or pass --whisper <path-to-ggml-base.bin>");
        if (!doCombinedRoundtrip) return 1;
    }
    else if (!File.Exists(wavToTranscribe))
    {
        Console.Error.WriteLine($"Error: WAV file not found: {wavToTranscribe}");
        return 1;
    }
    else
    {
        Console.WriteLine($"[ASR] Transcribing audio with Whisper...");
        Console.WriteLine($"      File: {wavToTranscribe}");
        using var stt = WhisperPipeline.Load(whisperModelPath);
        var (samples, sampleRate, _) = WavReader.ReadWav(wavToTranscribe);
        var sttResult = stt.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = sampleRate
        });
        Console.WriteLine($"      Transcription: \"{sttResult.Text.Trim()}\"");
    }
}

return 0;

static string? ResolveVoiceConfig()
{
    try
    {
        var home = ModelHome.Default();
        var entry = ModelCatalog.Find("piper-lessac");
        if (entry != null && home.StateOf(entry) == InstallState.Installed)
        {
            var jsonFile = entry.Files.FirstOrDefault(f => f.FileName.EndsWith(".json", StringComparison.OrdinalIgnoreCase));
            if (jsonFile != null && File.Exists(home.PathOf(jsonFile)))
                return home.PathOf(jsonFile);
        }
    }
    catch { }
    return null;
}

static string? ResolveWhisperModel()
{
    try
    {
        var home = ModelHome.Default();
        var entry = ModelCatalog.Find("whisper-base");
        if (entry != null && home.StateOf(entry) == InstallState.Installed)
        {
            if (File.Exists(home.PathOf(entry.MainFile)))
                return home.PathOf(entry.MainFile);
        }
    }
    catch { }
    return null;
}
