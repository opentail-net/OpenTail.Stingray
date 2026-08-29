
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

        // 1. Text-only synthesis with CFG rate 0.7
        var sw = Stopwatch.StartNew();
        var pcmTextOnly = pipeline.Generate("Hello there, this is a test.", maxNewSpeechTokens: 100, odeSteps: 10, seed: 42, cfgRate: 0.7f);
        sw.Stop();
        Assert.NotEmpty(pcmTextOnly);

        var resTextOnly = new AudioGenerationResult(pcmTextOnly, 24000);
        string outTextOnly = Path.Combine(outDir!, "cosyvoice3-cfg07-seed42.wav");
        resTextOnly.SaveWav(outTextOnly);
        Console.WriteLine($"[CosyVoice3 TextOnly] saved {outTextOnly} samples={pcmTextOnly.Length} durationSec={pcmTextOnly.Length / 24000.0:F2} elapsedSec={sw.Elapsed.TotalSeconds:F2}s");

        // 2. Zero-shot cloning with reference audio
        string? refAudio = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        if (refAudio != null)
        {
            sw.Restart();
            var pcmCloned = pipeline.Generate("Hello there, this is a test.", maxNewSpeechTokens: 100, odeSteps: 10, seed: 42, referenceAudioPath: refAudio, cfgRate: 0.7f);
            sw.Stop();
            Assert.NotEmpty(pcmCloned);

            var resCloned = new AudioGenerationResult(pcmCloned, 24000);
            string outCloned = Path.Combine(outDir!, "cosyvoice3-cloned-cfg07-seed42.wav");
            resCloned.SaveWav(outCloned);
            Console.WriteLine($"[CosyVoice3 Cloned] saved {outCloned} samples={pcmCloned.Length} durationSec={pcmCloned.Length / 24000.0:F2} elapsedSec={sw.Elapsed.TotalSeconds:F2}s");
        }
    }
}
