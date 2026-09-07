namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Direct frame-by-frame comparison of this port's real audio encoder output against the
/// reference's real dumped `audio_embeddings.bin` (see docs/audio-review-progress.md's "ROOT
/// CAUSE LOCALIZED" entry, 2026-09-07) -- the precise next step that entry scopes: determine
/// whether the previously-found audio-splice-position divergence originates in
/// <see cref="Audio.QwenASR.QwenAsrAudioEncoder"/> itself or in how its output is spliced into
/// the combined embedding table afterward.
/// </summary>
public sealed class QwenForcedAlignerAudioEncoderFrameCompareDebugTest : HeavyTestBase
{
    private const string DumpDir = "C:/Users/Dmitri/AppData/Local/Temp/claude/C--Git-Public/e20d4458-3906-46e3-a5d1-af57aa8bcc0d/scratchpad/fa_dump";

    private static float[] ReadRawF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    [Fact]
    public void OurAudioEmbeddings_ComparedFrameByFrame_AgainstRealReferenceDump()
    {
        string refPath = Path.Combine(DumpDir, "audio_embeddings.bin");
        string ourPath = Path.Combine(DumpDir, "our_audio_embeddings.bin");
        Assert.SkipUnless(File.Exists(refPath) && File.Exists(ourPath), "real dumps not found -- see class doc comment for how to regenerate");

        const int hiddenDim = 1024;
        var refEmb = ReadRawF32(refPath);
        var ourEmb = ReadRawF32(ourPath);
        Assert.Equal(refEmb.Length, ourEmb.Length);
        int frames = refEmb.Length / hiddenDim;

        double sumSqDiff = 0, sumSqReal = 0;
        for (int i = 0; i < refEmb.Length; i++)
        {
            double d = ourEmb[i] - refEmb[i];
            sumSqDiff += d * d;
            sumSqReal += (double)refEmb[i] * refEmb[i];
        }
        double overallRelError = Math.Sqrt(sumSqDiff / Math.Max(1e-12, sumSqReal));
        Console.Error.WriteLine($"[FA-AudioEnc-Compare] overall relative L2 error = {overallRelError:F6}");

        for (int f = 0; f < frames; f++)
        {
            double d2 = 0, r2 = 0, dot = 0, na = 0, nb = 0;
            for (int i = 0; i < hiddenDim; i++)
            {
                float a = ourEmb[f * hiddenDim + i], b = refEmb[f * hiddenDim + i];
                double d = a - b;
                d2 += d * d;
                r2 += (double)b * b;
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            }
            double rowErr = Math.Sqrt(d2 / Math.Max(1e-12, r2));
            double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
            Console.Error.WriteLine($"[FA-AudioEnc-ROW] frame={f} relError={rowErr:F6} cosine={cosine:F6}");
        }

        Assert.True(double.IsFinite(overallRelError));
    }
}
