
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug helper: converts the real reference's raw float32 PCM prompt clip
/// (`examples/cosyvoice.cpp/prompt24k.pcm`) into a real .wav file, so it can be passed to our own
/// CLI's --ref-audio flag for a true apples-to-apples full-pipeline run (the earlier
/// "(gibberish)" run mistakenly used the reference's SYNTHESIZED OUTPUT wav as --ref-audio, not
/// this original prompt clip).</summary>
public sealed class CosyVoice3MakePromptWavDebugTest : HeavyTestBase
{
    private static string? FindModelPath(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Convert_Prompt24k_ToWav()
    {
        string? pcmPath = FindModelPath("examples/cosyvoice.cpp/prompt24k.pcm");
        Assert.SkipUnless(pcmPath != null, "prompt24k.pcm not found");

        var bytes = File.ReadAllBytes(pcmPath!);
        var samples = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);

        string repoRoot = Directory.GetParent(Directory.GetParent(Directory.GetParent(pcmPath!)!.FullName)!.FullName)!.FullName;
        string outPath = Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice3-real-prompt-24k.wav");
        var result = new AudioGenerationResult(samples, 24000);
        result.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {samples.Length} samples, {samples.Length / 24000.0:F2}s");
    }
}
