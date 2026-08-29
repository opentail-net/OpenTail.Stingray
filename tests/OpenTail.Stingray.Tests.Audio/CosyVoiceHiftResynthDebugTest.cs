
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: isolates HiFT vocoder correctness by resynthesizing a REAL
/// reference wav's own real mel (bypassing LLM/DiT entirely) -- if this sounds clean, the
/// "mosquito"/"dentist drill" buzz reported for full CosyVoice3 generation lives upstream in the
/// DiT/CFM stage, not in HiFT.</summary>
public sealed class CosyVoiceHiftResynthDebugTest : HeavyTestBase
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
    public void Resynthesize_RealMel_Through_Hift()
    {
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        string? refPath = FindModelPath("docs/audio-samples/fishspeech-lunch.wav");
        Assert.SkipUnless(ggufPath != null && refPath != null, "CosyVoice3 GGUF or reference wav not found");

        var (samples, sr, _) = WavReader.ReadWav(refPath!);
        if (sr != CosyVoiceMelExtractor.SampleRate)
            samples = AudioResampler.Resample(samples, sr, CosyVoiceMelExtractor.SampleRate);

        var mel = CosyVoiceMelExtractor.Shared.ExtractMel(samples); // channel-last [T, 80]
        int numFrames = mel.Length / CosyVoiceMelExtractor.NumMels;
        Assert.True(numFrames > 0);

        var melChannelFirst = new float[mel.Length];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < CosyVoiceMelExtractor.NumMels; c++)
                melChannelFirst[c * numFrames + f] = mel[f * CosyVoiceMelExtractor.NumMels + c];

        using var hiftWeights = new CosyVoice3HiftWeights(ggufPath!);
        var rng = new Random(0);
        var wav = CosyVoiceHiftVocoder.Generate(hiftWeights, melChannelFirst, numFrames, rng);

        float peak = 0f;
        for (int i = 0; i < wav.Length; i++) peak = MathF.Max(peak, MathF.Abs(wav[i]));
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < wav.Length; i++) wav[i] *= gain;
        }

        var result = new AudioGenerationResult(wav, CosyVoiceMelExtractor.SampleRate);
        // ggufPath = <repoRoot>/models/cosyvoice3/CosyVoice3-2512_F16.gguf
        string repoRoot = Directory.GetParent(Directory.GetParent(Directory.GetParent(ggufPath!)!.FullName)!.FullName)!.FullName;
        result.SaveWav(Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice3-hift-resynth-realmel.wav"));
        Console.WriteLine($"Wrote resynth wav, {wav.Length} samples, {numFrames} mel frames");
    }
}
