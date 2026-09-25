
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

        // Reference clip for zero-shot cloning: CV3_REF / CV3_REF_TEXT, defaulting to the always-present
        // audio.cpp `a.wav` (the old default, docs/audio-samples/fishspeech-lunch-REFERENCE.wav, was a
        // local-only sample that no longer exists, which made this test skip).
        string? refAudio = Environment.GetEnvironmentVariable("CV3_REF") ?? FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(refAudio != null, "no CosyVoice3 reference audio (set CV3_REF)");
        string refText = Environment.GetEnvironmentVariable("CV3_REF_TEXT")
            ?? "This little work was finished in the year 1803, and intended for immediate publication.";
        string outPath = Environment.GetEnvironmentVariable("CV3_OUT")
            ?? Path.Combine(Path.GetTempPath(), "cosyvoice3-cloned-nopitchscale.wav");

        using var pipeline = CosyVoice3Pipeline.Load(modelPath!);
        const string prompt = "Hello, I will make some lunch, darling!";

        // Zero-shot cloning with real reference audio, WITHOUT the pitchScale=1.25 fudge factor.
        var pcmClonedNoPitch = pipeline.Generate(prompt, maxNewSpeechTokens: 150, odeSteps: 20, seed: 42, referenceAudioPath: refAudio, cfgRate: 0.7f, referenceText: refText, temperature: 0.8f, pitchScale: 1.0f);
        Assert.True(pcmClonedNoPitch.Length > 0);
        new AudioGenerationResult(pcmClonedNoPitch, 24000).SaveWav(outPath);
        Console.WriteLine($"Saved {outPath}: {pcmClonedNoPitch.Length} samples ({pcmClonedNoPitch.Length / 24000.0:F2}s)");
    }
}
