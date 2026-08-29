
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: feeds the REAL reference's own dumped `speech_feat` mel
/// (`examples/cosyvoice.cpp/src/cosyvoice-token2wav.cpp`'s `COSY_DUMP_SPEECH_FEAT_PATH`,
/// frame-major [T,80] float32, known-good since the reference CLI itself produced clean speech
/// from it) through OUR C# HiFT vocoder. If this is still buzzy, the bug is confirmed to be in
/// our HiFT port itself, not in our own (possibly mismatched) mel extraction.</summary>
public sealed class CosyVoiceHiftReferenceMelDebugTest : HeavyTestBase
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
    public void Resynthesize_ReferenceDumpedMel_Through_OurHift()
    {
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        string? dumpPath = FindModelPath("examples/cosyvoice.cpp/speech_feat_dump.bin");
        Assert.SkipUnless(ggufPath != null && dumpPath != null, "CosyVoice3 GGUF or reference speech_feat dump not found");

        var bytes = File.ReadAllBytes(dumpPath!);
        int totalFloats = bytes.Length / sizeof(float);
        const int melDim = 80;
        int numFrames = totalFloats / melDim;
        Assert.True(numFrames > 0);

        var melFrameMajor = new float[totalFloats]; // [T, 80], as dumped
        Buffer.BlockCopy(bytes, 0, melFrameMajor, 0, bytes.Length);

        var melChannelFirst = new float[totalFloats];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < melDim; c++)
                melChannelFirst[c * numFrames + f] = melFrameMajor[f * melDim + c];

        string repoRoot = Directory.GetParent(Directory.GetParent(Directory.GetParent(ggufPath!)!.FullName)!.FullName)!.FullName;

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

        var result = new AudioGenerationResult(wav, 24000);
        result.SaveWav(Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice3-hift-resynth-referencemel.wav"));
        Console.WriteLine($"Wrote resynth wav from reference-dumped mel, {wav.Length} samples, {numFrames} mel frames");
    }
}
