
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: measures wall-clock generation time for QwenTTS and CosyVoice3
/// on the same prompt, to establish a performance baseline before any optimization work. Includes
/// a warmup run (excluded from timing, since first-call JIT/model-load effects would otherwise
/// dominate) followed by several timed runs so a single slow/fast outlier doesn't skew the result.</summary>
public sealed class TtsPerformanceBaselineDebugTest : HeavyTestBase
{
    private const string Prompt = "Hello, I will make some lunch, darling!";
    private const int Runs = 3;

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Baseline_QwenTts()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "QwenTTS talker model not found");

        using var pipeline = QwenTtsPipeline.Load(talkerPath!);

        // Warmup (not timed): first call pays for JIT/tensor-cache warmup.
        var warm = pipeline.Generate(Prompt, seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "qwen-tts-perf-turn5.wav");
                new AudioGenerationResult(lastWav, 24000).SaveWav(wavPath);
            }
        }

        double audioSec = sampleCount / 24000.0;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[QwenTTS] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[QwenTTS] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_CosyVoice3()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");

        using var pipeline = CosyVoice3Pipeline.Load(modelPath!);

        var warm = pipeline.Generate(Prompt, seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "cosyvoice3-perf-turn2.wav");
                new AudioGenerationResult(lastWav, 24000).SaveWav(wavPath);
            }
        }

        double audioSec = sampleCount / 24000.0;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[CosyVoice3] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[CosyVoice3] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_QwenTts()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "QwenTTS talker model not found");

        using var pipeline = QwenTtsPipeline.Load(talkerPath!);

        // Warmup (not timed): JIT and tensor allocations
        await foreach (var _ in pipeline.GenerateStreamAsync(Prompt, chunkFrames: 1, seed: 42)) { }

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(Prompt, chunkFrames: 1, seed: 42))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "qwen-tts-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, 24000).SaveWav(wavPath);
        }

        double audioSec = totalSamples / 24000.0;
        string msg = $"[QwenTTS-Stream-Frame1] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[QwenTTS-Stream-Frame1] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_FishSpeech()
    {
        string? modelPath = FindRepoFile("models/s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoFile("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "FishSpeech S2 Pro GGUF or tokenizer not found");

        using var pipeline = new FishSpeechFullPipeline(modelPath!, tokDir!, modelPath!);

        // Warmup (not timed): JIT and tensor allocations
        await foreach (var _ in pipeline.GenerateStreamAsync(Prompt, chunkFrames: 1, seed: 42)) { }

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(Prompt, chunkFrames: 1, seed: 42))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "fishspeech-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, 44100).SaveWav(wavPath);
        }

        double audioSec = totalSamples / 44100.0;
        string msg = $"[FishSpeech-Stream-Frame1] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[FishSpeech-Stream-Frame1] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_FishSpeech()
    {
        string? modelPath = FindRepoFile("models/s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoFile("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "FishSpeech S2 Pro GGUF or tokenizer not found");

        using var pipeline = new FishSpeechFullPipeline(modelPath!, tokDir!, modelPath!);

        // Warmup (not timed): JIT and tensor allocations
        var warm = pipeline.Synthesize(Prompt, seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Synthesize(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "fishspeech-perf-turn3.wav");
                new AudioGenerationResult(lastWav, 44100).SaveWav(wavPath);
            }
        }

        double audioSec = sampleCount / 44100.0;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[FishSpeech] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[FishSpeech] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_Xtts()
    {
        string? checkpointDir = FindRepoFile("models/xtts-v2/model.safetensors") is { } p ? Path.GetDirectoryName(p) : null;
        string? refWav = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        Assert.SkipUnless(checkpointDir != null && refWav != null, "XTTS checkpoint or reference audio not found");

        var pipeline = OpenTail.Stingray.Audio.Xtts.XttsPipeline.Load(checkpointDir!);

        // Warmup
        var warm = pipeline.Generate(Prompt, refWav!, "en", seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, refWav!, "en", seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "xtts-perf-turn2.wav");
                new AudioGenerationResult(lastWav, 24000).SaveWav(wavPath);
            }
        }

        double audioSec = sampleCount / 24000.0;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[XTTS-v2] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[XTTS-v2] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_Xtts()
    {
        string? checkpointDir = FindRepoFile("models/xtts-v2/model.safetensors") is { } p ? Path.GetDirectoryName(p) : null;
        string? refWav = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        Assert.SkipUnless(checkpointDir != null && refWav != null, "XTTS checkpoint or reference audio not found");

        var pipeline = OpenTail.Stingray.Audio.Xtts.XttsPipeline.Load(checkpointDir!);

        // Warmup
        await foreach (var _ in pipeline.GenerateStreamAsync(Prompt, refWav!, "en", chunkTokens: 6, seed: 42)) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(Prompt, refWav!, "en", chunkTokens: 6, seed: 42))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "xtts-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, 24000).SaveWav(wavPath);
        }

        double audioSec = totalSamples / 24000.0;
        string msg = $"[XTTS-Stream-Chunk6] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[XTTS-Stream-Chunk6] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_MmsTts()
    {
        string? checkpointDir = FindRepoFile("models/mms-tts-eng/model.safetensors") is { } p ? Path.GetDirectoryName(p) : null;
        Assert.SkipUnless(checkpointDir != null, "MMS-TTS checkpoint not found");

        var pipeline = OpenTail.Stingray.Audio.MmsTts.MmsTtsPipeline.Load(checkpointDir!);

        // Warmup
        var warm = pipeline.Generate(Prompt, seed: 42);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Generate(Prompt, seed: 42);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "mms-tts-perf-turn2.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[MMS-TTS-eng] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[MMS-TTS-eng] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_MmsTts()
    {
        string? checkpointDir = FindRepoFile("models/mms-tts-eng/model.safetensors") is { } p ? Path.GetDirectoryName(p) : null;
        Assert.SkipUnless(checkpointDir != null, "MMS-TTS checkpoint not found");

        var pipeline = OpenTail.Stingray.Audio.MmsTts.MmsTtsPipeline.Load(checkpointDir!);

        // Warmup
        await foreach (var _ in pipeline.GenerateStreamAsync(Prompt, chunkFrames: 16, seed: 42)) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(Prompt, chunkFrames: 16, seed: 42))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "mms-tts-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[MMS-TTS-Stream-Frame16] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[MMS-TTS-Stream-Frame16] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_Piper()
    {
        string? onnxPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? jsonPath = FindRepoFile("models/en_US-lessac-medium.onnx.json");
        Assert.SkipUnless(onnxPath != null && jsonPath != null, "Piper ONNX model not found");

        var pipeline = OpenTail.Stingray.Audio.Piper.PiperPipeline.FromConfigFile(jsonPath!);

        // Warmup
        var warm = pipeline.Generate(new AudioGenerationRequest { Text = Prompt });
        Assert.NotEmpty(warm.Samples);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var res = pipeline.Generate(new AudioGenerationRequest { Text = Prompt });
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = res.Samples.Length;
            lastWav = res.Samples;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "piper-perf-turn1.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[Piper-lessac-medium] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[Piper-lessac-medium] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_Piper()
    {
        string? onnxPath = FindRepoFile("models/en_US-lessac-medium.onnx");
        string? jsonPath = FindRepoFile("models/en_US-lessac-medium.onnx.json");
        Assert.SkipUnless(onnxPath != null && jsonPath != null, "Piper ONNX model not found");

        var pipeline = OpenTail.Stingray.Audio.Piper.PiperPipeline.FromConfigFile(jsonPath!);

        // Warmup
        await foreach (var _ in pipeline.GenerateStreamAsync(new AudioGenerationRequest { Text = Prompt })) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(new AudioGenerationRequest { Text = Prompt }))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "piper-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[Piper-Stream-Frame16] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[Piper-Stream-Frame16] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_Kokoro()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? voicePath = FindRepoFile("models/kokoro-voice-af_heart.gguf");
        Assert.SkipUnless(modelPath != null && voicePath != null, "Kokoro GGUF model or voice not found");

        using var model = OpenTail.Stingray.Audio.Kokoro.KokoroModel.Load(modelPath!, voicePath!);
        using var pipeline = new OpenTail.Stingray.Audio.Kokoro.KokoroPipeline(model);

        var req = new AudioGenerationRequest { Text = Prompt, Voice = "af_heart" };
        var warm = pipeline.Generate(req);
        Assert.NotEmpty(warm.Samples);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var res = pipeline.Generate(req);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = res.Samples.Length;
            lastWav = res.Samples;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "kokoro-perf-turn2.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[Kokoro-82M] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[Kokoro-82M] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_MeloTts()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        Assert.SkipUnless(modelPath != null, "MeloTTS ONNX model not found");

        using var pipeline = OpenTail.Stingray.Audio.MeloTTS.MeloPipeline.Load(modelPath!);

        var req = new AudioGenerationRequest { Text = Prompt, Voice = "EN-US" };
        var warm = pipeline.Generate(req);
        Assert.NotEmpty(warm.Samples);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var res = pipeline.Generate(req);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = res.Samples.Length;
            lastWav = res.Samples;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "melotts-perf-turn2.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[MeloTTS] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[MeloTTS] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_Kokoro()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? voicePath = FindRepoFile("models/kokoro-voice-af_heart.gguf");
        Assert.SkipUnless(modelPath != null && voicePath != null, "Kokoro GGUF model or voice not found");

        using var model = OpenTail.Stingray.Audio.Kokoro.KokoroModel.Load(modelPath!, voicePath!);
        using var pipeline = new OpenTail.Stingray.Audio.Kokoro.KokoroPipeline(model);

        var req = new AudioGenerationRequest { Text = Prompt, Voice = "af_heart" };

        // Warmup
        await foreach (var _ in pipeline.GenerateStreamAsync(req)) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(req))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "kokoro-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[Kokoro-82M-Stream] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[Kokoro-82M-Stream] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_MeloTts()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        Assert.SkipUnless(modelPath != null, "MeloTTS ONNX model not found");

        using var pipeline = OpenTail.Stingray.Audio.MeloTTS.MeloPipeline.Load(modelPath!);

        var req = new AudioGenerationRequest { Text = Prompt, Voice = "EN-US" };

        // Warmup
        await foreach (var _ in pipeline.GenerateStreamAsync(req)) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(req))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "melotts-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[MeloTTS-Stream] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[MeloTTS-Stream] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_F5Tts()
    {
        string? modelPath = FindRepoFile("models/f5tts_base.safetensors");
        Assert.SkipUnless(modelPath != null, "F5TTS safetensors model not found");

        using var pipeline = OpenTail.Stingray.Audio.F5TTS.F5TtsPipeline.Load(modelPath!);

        var req = new AudioGenerationRequest { Text = Prompt };
        var warm = pipeline.Generate(req);
        Assert.NotEmpty(warm.Samples);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var res = pipeline.Generate(req);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = res.Samples.Length;
            lastWav = res.Samples;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "f5tts-perf-baseline.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
                string turn1Path = Path.Combine(outDir, "f5tts-perf-turn1.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(turn1Path);
                string turn2Path = Path.Combine(outDir, "f5tts-perf-turn2.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(turn2Path);
                string turn3Path = Path.Combine(outDir, "f5tts-perf-turn3.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(turn3Path);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[F5-TTS] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[F5-TTS] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_F5Tts()
    {
        string? modelPath = FindRepoFile("models/f5tts_base.safetensors");
        Assert.SkipUnless(modelPath != null, "F5TTS safetensors model not found");

        using var pipeline = OpenTail.Stingray.Audio.F5TTS.F5TtsPipeline.Load(modelPath!);

        var req = new AudioGenerationRequest { Text = Prompt };

        // Warmup (not timed)
        await foreach (var _ in pipeline.GenerateStreamAsync(req)) { }

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(req))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "f5tts-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[F5-TTS-Stream] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[F5-TTS-Stream] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_Chatterbox()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        string? s3GenPath = FindRepoFile("models/chatterbox-turbo-s3gen-q4_k.gguf");
        Assert.SkipUnless(t3Path != null && s3GenPath != null, "Chatterbox GGUF models not found");

        using var pipeline = OpenTail.Stingray.Audio.Chatterbox.ChatterboxPipeline.Load(t3Path!, s3GenPath!);

        var req = new AudioGenerationRequest { Text = Prompt, Voice = "nova" };
        var warm = pipeline.Generate(req);
        Assert.NotEmpty(warm.Samples);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var res = pipeline.Generate(req);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = res.Samples.Length;
            lastWav = res.Samples;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "chatterbox-perf-turn1.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[Chatterbox] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[Chatterbox] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_Chatterbox()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        string? s3GenPath = FindRepoFile("models/chatterbox-turbo-s3gen-q4_k.gguf");
        Assert.SkipUnless(t3Path != null && s3GenPath != null, "Chatterbox GGUF models not found");

        using var pipeline = OpenTail.Stingray.Audio.Chatterbox.ChatterboxPipeline.Load(t3Path!, s3GenPath!);

        var req = new AudioGenerationRequest { Text = Prompt, Voice = "nova" };

        // Warmup
        await foreach (var _ in pipeline.GenerateStreamAsync(req)) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.GenerateStreamAsync(req))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "chatterbox-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[Chatterbox-Stream] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[Chatterbox-Stream] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public void Baseline_Parler()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        Assert.SkipUnless(modelPath != null && tokenizerPath != null, "Parler safetensors model or tokenizer not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(modelPath!);
        using var pipeline = new OpenTail.Stingray.Audio.Parler.ParlerFullPipeline(tokenizerPath!, loader);

        const string femaleDesc = "Jenny's voice speaks with a warm and friendly tone at a relaxed, natural pace with clear articulation and gentle pauses in a quiet studio environment.";
        var warm = pipeline.Synthesize(Prompt, description: femaleDesc, maxNewTokens: 250);
        Assert.NotEmpty(warm);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        float[]? lastWav = null;
        for (int i = 0; i < Runs; i++)
        {
            var sw = Stopwatch.StartNew();
            var wav = pipeline.Synthesize(Prompt, description: femaleDesc, maxNewTokens: 250);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = wav.Length;
            lastWav = wav;
        }

        if (lastWav != null)
        {
            string? outDir = FindRepoFile("docs/audio-samples");
            if (outDir != null)
            {
                string wavPath = Path.Combine(outDir, "parler-perf-turn2.wav");
                new AudioGenerationResult(lastWav, pipeline.DefaultSampleRate).SaveWav(wavPath);
            }
        }

        double audioSec = (double)sampleCount / pipeline.DefaultSampleRate;
        double meanSec = Average(elapsedSec);
        double rtf = meanSec / audioSec;
        string msg = $"[Parler-TTS] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[Parler-TTS] runs(s)=[{string.Join(", ", Array.ConvertAll(elapsedSec, x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    [Fact]
    public async Task Streaming_Parler()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        Assert.SkipUnless(modelPath != null && tokenizerPath != null, "Parler safetensors model or tokenizer not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(modelPath!);
        using var pipeline = new OpenTail.Stingray.Audio.Parler.ParlerFullPipeline(tokenizerPath!, loader);

        const string femaleDesc = "Jenny's voice speaks with a warm and friendly tone at a relaxed, natural pace with clear articulation and gentle pauses in a quiet studio environment.";

        // Warmup
        await foreach (var _ in pipeline.SynthesizeStreamAsync(Prompt, description: femaleDesc, maxNewTokens: 250, chunkFrames: 16)) break;

        var sw = Stopwatch.StartNew();
        double ttfaSec = 0;
        var chunks = new List<float[]>();
        int totalSamples = 0;

        await foreach (var chunk in pipeline.SynthesizeStreamAsync(Prompt, description: femaleDesc, maxNewTokens: 250, chunkFrames: 16))
        {
            if (chunks.Count == 0)
            {
                ttfaSec = sw.Elapsed.TotalSeconds;
            }
            chunks.Add(chunk);
            totalSamples += chunk.Length;
        }
        sw.Stop();
        double totalSec = sw.Elapsed.TotalSeconds;

        var fullPcm = new float[totalSamples];
        int offset = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, fullPcm, offset, c.Length);
            offset += c.Length;
        }

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            string wavPath = Path.Combine(outDir, "parler-streaming-streamed.wav");
            new AudioGenerationResult(fullPcm, pipeline.DefaultSampleRate).SaveWav(wavPath);
        }

        double audioSec = (double)totalSamples / pipeline.DefaultSampleRate;
        string msg = $"[Parler-TTS-Stream-Frame16] prompt=\"{Prompt}\" audio={audioSec:F2}s samples={totalSamples} chunks={chunks.Count}\n" +
                     $"[Parler-TTS-Stream-Frame16] Time-To-First-Audio (TTFA)={ttfaSec:F3}s TotalTime={totalSec:F3}s";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    private static double Average(double[] values)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        return sum / values.Length;
    }
}
