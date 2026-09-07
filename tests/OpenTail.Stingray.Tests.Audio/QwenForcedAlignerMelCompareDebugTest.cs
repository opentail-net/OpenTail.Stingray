namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Compares this port's own mel-spectrogram output against the reference's real dumped
/// mel features, frame-by-frame -- continuation of docs/audio-review-progress.md's Qwen3 Forced
/// Aligner bisection (2026-09-07). Note: this port's mel array is MEL-MAJOR (`mel[m*frames+f]`),
/// the reference's real dumped array is presumably frame-major (row-per-frame, matching its own
/// internal `AudioTensor` convention) -- transpose as needed when reading.</summary>
public sealed class QwenForcedAlignerMelCompareDebugTest : HeavyTestBase
{
    private const string DumpDir = "C:/Users/Dmitri/AppData/Local/Temp/claude/C--Git-Public/e20d4458-3906-46e3-a5d1-af57aa8bcc0d/scratchpad/fa_dump";
    private const int MelBins = 128;

    private static float[] ReadRawF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    [Fact]
    public void CompareMelFeatures_OursVsReference()
    {
        string ourPath = Path.Combine(DumpDir, "our_mel_features.bin");
        string refPath = Path.Combine(DumpDir, "mel_features.bin");
        Assert.SkipUnless(File.Exists(ourPath) && File.Exists(refPath), "dumps not found");

        var ours = ReadRawF32(ourPath); // mel-major: ours[m*ourFrames+f]
        var refRaw = ReadRawF32(refPath);
        int ourFrames = ours.Length / MelBins;
        int refFrames = refRaw.Length / MelBins;
        Console.Error.WriteLine($"[FA-MEL-CMP] our_frames={ourFrames} ref_frames={refFrames}");

        // Try BOTH possible reference layouts (mel-major vs frame-major) and report whichever
        // correlates better, to avoid guessing which convention the reference dump uses.
        int frames = Math.Min(ourFrames, refFrames);

        double CosineMelMajor(int f)
        {
            double dot = 0, na = 0, nb = 0;
            for (int m = 0; m < MelBins; m++)
            {
                float a = ours[m * ourFrames + f];
                float b = refRaw[m * refFrames + f];
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        }

        double CosineFrameMajor(int f)
        {
            double dot = 0, na = 0, nb = 0;
            for (int m = 0; m < MelBins; m++)
            {
                float a = ours[m * ourFrames + f];
                float b = refRaw[f * MelBins + m];
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        }

        for (int f = 0; f < Math.Min(frames, 10); f++)
        {
            Console.Error.WriteLine($"[FA-MEL-CMP] frame={f} cosine_melMajor={CosineMelMajor(f):F6} cosine_frameMajor={CosineFrameMajor(f):F6}");
        }

        Assert.True(frames > 0);
    }
}
