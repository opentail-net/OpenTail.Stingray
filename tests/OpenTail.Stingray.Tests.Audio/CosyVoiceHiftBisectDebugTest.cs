
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: bisects the CosyVoice3 HiFT vocoder's "mosquito"/buzz bug by
/// feeding the real reference's own dumped excitation signal (`examples/cosyvoice.cpp`'s
/// COSY_DUMP_EXCITATION_PATH, known-good since the reference CLI produces clean speech) directly
/// into our C# HiFTVocoderKernels.DecodeForTest, bypassing our own SineGen entirely. If this is
/// clean, the bug is in our SineGen/source generation; if still buzzy, the bug is in Decode
/// (conv_pre/upsample/resblock/ISTFT).</summary>
public sealed class CosyVoiceHiftBisectDebugTest : HeavyTestBase
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
    public void Decode_With_Reference_Excitation()
    {
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        string? melDumpPath = FindModelPath("examples/cosyvoice.cpp/speech_feat_dump.bin");
        string? excDumpPath = FindModelPath("examples/cosyvoice.cpp/excitation_dump.bin");
        Assert.SkipUnless(ggufPath != null && melDumpPath != null && excDumpPath != null,
            "CosyVoice3 GGUF or reference speech_feat/excitation dumps not found");

        const int melDim = 80;
        var melBytes = File.ReadAllBytes(melDumpPath!);
        int numFrames = (melBytes.Length / sizeof(float)) / melDim;
        var melFrameMajor = new float[melBytes.Length / sizeof(float)];
        Buffer.BlockCopy(melBytes, 0, melFrameMajor, 0, melBytes.Length);
        var melChannelFirst = new float[melFrameMajor.Length];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < melDim; c++)
                melChannelFirst[c * numFrames + f] = melFrameMajor[f * melDim + c];

        var excBytes = File.ReadAllBytes(excDumpPath!);
        var excitation = new float[excBytes.Length / sizeof(float)];
        Buffer.BlockCopy(excBytes, 0, excitation, 0, excBytes.Length);

        using var hiftWeights = new CosyVoice3HiftWeights(ggufPath!);
        var wav = HiFTVocoderKernels.DecodeForTest(hiftWeights, melChannelFirst, numFrames, excitation, excitation.Length, melDim);

        float peak = 0f;
        for (int i = 0; i < wav.Length; i++) peak = MathF.Max(peak, MathF.Abs(wav[i]));
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < wav.Length; i++) wav[i] *= gain;
        }

        string repoRoot = Directory.GetParent(Directory.GetParent(Directory.GetParent(ggufPath!)!.FullName)!.FullName)!.FullName;
        var result = new AudioGenerationResult(wav, 24000);
        result.SaveWav(Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice3-hift-bisect-referenceexcitation.wav"));
        Console.WriteLine($"Wrote bisect wav, {wav.Length} samples, {numFrames} mel frames");
    }
}
