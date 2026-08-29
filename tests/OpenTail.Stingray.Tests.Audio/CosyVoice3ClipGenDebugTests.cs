
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

        // Zero-shot cloning with real reference audio, WITHOUT the pitchScale=1.25 fudge factor --
        // isolate whether that hand-tuned hack (calibrated against the no-reference case) is what
        // makes cloned output sound worse than the unconditioned "smooth" runs.
        string? refAudio = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        if (refAudio != null)
        {
            var pcmClonedNoPitch = pipeline.Generate(prompt, maxNewSpeechTokens: 150, odeSteps: 20, seed: 42, referenceAudioPath: refAudio, cfgRate: 0.7f, referenceText: prompt, temperature: 0.8f, pitchScale: 1.0f);
            if (pcmClonedNoPitch.Length > 0)
            {
                string path = Path.Combine(outDir!, "cosyvoice3-cloned-nopitchscale.wav");
                new AudioGenerationResult(pcmClonedNoPitch, 24000).SaveWav(path);
                Console.WriteLine($"Saved {path}: {pcmClonedNoPitch.Length} samples ({pcmClonedNoPitch.Length / 24000.0:F2}s)");
            }
        }
    }
}
