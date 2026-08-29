
namespace OpenTail.Stingray.Tests.Audio;

public sealed class CosyVoice3ClipGenDebugTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_CosyVoice3_Clips()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");

        string? outDir = Path.GetDirectoryName(FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav"));
        Assert.SkipUnless(outDir != null, "docs/audio-samples directory not found");

        using var pipeline = CosyVoice3Pipeline.Load(modelPath!);
        const string prompt = "Hello, I will make some lunch, darling!";

        // 1. Smooth F0 with odeSteps = 10
        var pcmSmooth10 = pipeline.Generate(prompt, maxNewSpeechTokens: 150, odeSteps: 10, seed: 42, cfgRate: 0.7f, temperature: 0.8f, pitchScale: 1.25f);
        if (pcmSmooth10.Length > 0)
        {
            string path = Path.Combine(outDir!, "cosyvoice3-smooth-ode10.wav");
            new AudioGenerationResult(pcmSmooth10, 24000).SaveWav(path);
            Console.WriteLine($"Saved {path}: {pcmSmooth10.Length} samples ({pcmSmooth10.Length / 24000.0:F2}s)");
        }

        // 2. Smooth F0 with odeSteps = 15 (smoother mel trajectory)
        var pcmSmooth15 = pipeline.Generate(prompt, maxNewSpeechTokens: 150, odeSteps: 15, seed: 42, cfgRate: 0.7f, temperature: 0.8f, pitchScale: 1.25f);
        if (pcmSmooth15.Length > 0)
        {
            string path = Path.Combine(outDir!, "cosyvoice3-smooth-ode15.wav");
            new AudioGenerationResult(pcmSmooth15, 24000).SaveWav(path);
            Console.WriteLine($"Saved {path}: {pcmSmooth15.Length} samples ({pcmSmooth15.Length / 24000.0:F2}s)");
        }

        // 3. Smooth F0 with odeSteps = 20 (highest quality mel flow integration)
        var pcmSmooth20 = pipeline.Generate(prompt, maxNewSpeechTokens: 150, odeSteps: 20, seed: 42, cfgRate: 0.7f, temperature: 0.8f, pitchScale: 1.25f);
        if (pcmSmooth20.Length > 0)
        {
            string path = Path.Combine(outDir!, "cosyvoice3-smooth-ode20.wav");
            new AudioGenerationResult(pcmSmooth20, 24000).SaveWav(path);
            Console.WriteLine($"Saved {path}: {pcmSmooth20.Length} samples ({pcmSmooth20.Length / 24000.0:F2}s)");
        }

        // 4. Zero-shot cloning with real reference audio -- dump alignment/speaker debug numbers
        string? refAudio = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        if (refAudio != null)
        {
            Environment.SetEnvironmentVariable("STINGRAY_DEBUG_COSYVOICE3", "1");
            var pcmCloned = pipeline.Generate(prompt, maxNewSpeechTokens: 150, odeSteps: 20, seed: 42, referenceAudioPath: refAudio, cfgRate: 0.7f, referenceText: prompt, temperature: 0.8f, pitchScale: 1.25f);
            Environment.SetEnvironmentVariable("STINGRAY_DEBUG_COSYVOICE3", null);
            if (pcmCloned.Length > 0)
            {
                string path = Path.Combine(outDir!, "cosyvoice3-cloned-align-debug.wav");
                new AudioGenerationResult(pcmCloned, 24000).SaveWav(path);
                Console.WriteLine($"Saved {path}: {pcmCloned.Length} samples ({pcmCloned.Length / 24000.0:F2}s)");
            }
        }
    }
}
