namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Compares this port's own audio-encoder internal stages against the reference's real
/// dumped stages, frame-by-frame -- continuation of docs/audio-review-progress.md's audio-encoder
/// bisection (2026-09-07).</summary>
public sealed class QwenForcedAlignerEncoderStageCompareDebugTest : HeavyTestBase
{
    private const string DumpDir = "C:/Users/Dmitri/AppData/Local/Temp/claude/C--Git-Public/e20d4458-3906-46e3-a5d1-af57aa8bcc0d/scratchpad/fa_dump";
    private const int HiddenDim = 1024;

    private static float[] ReadRawF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    private static void CompareFrameByFrame(string label, float[] ours, float[] reference, int frames)
    {
        for (int f = 0; f < frames; f++)
        {
            double d2 = 0, r2 = 0, dot = 0, na = 0, nb = 0;
            for (int i = 0; i < HiddenDim; i++)
            {
                float a = ours[f * HiddenDim + i], b = reference[f * HiddenDim + i];
                double d = a - b;
                d2 += d * d; r2 += (double)b * b;
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            }
            double rowErr = Math.Sqrt(d2 / Math.Max(1e-12, r2));
            double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
            Console.Error.WriteLine($"[FA-ENC-CMP] {label} frame={f} relError={rowErr:F6} cosine={cosine:F6}");
        }
    }

    [Fact]
    public void CompareEncoderStages_OursVsReference()
    {
        string ourConvPos = Path.Combine(DumpDir, "our_enc_after_conv_pos.bin");
        string refConvPos = Path.Combine(DumpDir, "enc_after_conv_pos.bin");
        Assert.SkipUnless(File.Exists(ourConvPos) && File.Exists(refConvPos), "dumps not found");

        var ours = ReadRawF32(ourConvPos);
        var refAll = ReadRawF32(refConvPos);
        int ourFrames = ours.Length / HiddenDim;
        int refFrames = refAll.Length / HiddenDim;
        Console.Error.WriteLine($"[FA-ENC-CMP] our_frames={ourFrames} ref_frames={refFrames}");

        // Reference has 1 extra (padding) frame -- compare the first min(ourFrames, refFrames).
        int frames = Math.Min(ourFrames, refFrames);
        CompareFrameByFrame("after_conv_pos", ours, refAll, frames);

        Assert.True(frames > 0);
    }
}
