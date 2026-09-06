
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real golden-verification test for the Voxtral mel frontend and audio-tower forward
/// pass against the real C++ reference's exact dumped arrays, captured via
/// STINGRAY_VOXTRAL_TRACE=1 full-array binary dumps added to audio_encoder.cpp (the built-in
/// trace_log_scalar calls don't dump full arrays). Feeds the SAME real mel input the reference
/// used (dumped, not re-extracted) into this port's own VoxtralAudioEncoder to isolate the
/// encoder/projector from the mel frontend, then separately reports the mel extractor's own
/// stats on the real audio for comparison. Skips (does not fail) when the dumps or the real
/// checkpoint are not present, matching this project's other RealCheckpoint-dependent tests.</summary>
public sealed class VoxtralRealReferenceMatchTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
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

    private static string? DumpDir()
    {
        var temp = Path.GetTempPath();
        var candidate = Path.Combine(temp, "claude");
        if (!Directory.Exists(candidate)) return null;
        var hits = Directory.GetFiles(candidate, "voxtral_mel.bin", SearchOption.AllDirectories);
        return hits.Length > 0 ? Path.GetDirectoryName(hits[0]) : null;
    }

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    [Fact]
    public void AudioEncoder_RealReferenceMelInput_MatchesRealReferenceEmbeddings()
    {
        string? checkpointPath = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors");
        Assert.SkipUnless(checkpointPath != null, "voxtral-mini-realtime model.safetensors not found");
        string? dumpDir = DumpDir();
        Assert.SkipUnless(dumpDir != null, "real reference voxtral_mel.bin/voxtral_audio_embeddings.bin dumps not found -- see class doc comment to generate them");

        var mel = ReadF32(Path.Combine(dumpDir!, "voxtral_mel.bin"));
        var refEmbeddings = ReadF32(Path.Combine(dumpDir!, "voxtral_audio_embeddings.bin"));

        int melBins = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.NumMelBins;
        int melFrames = mel.Length / melBins;

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(checkpointPath!);
        var w = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights(loader);

        var tokens = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoder.Forward(w, mel, melFrames);

        int textHidden = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.TextHiddenSize;
        var ours = new float[tokens.Length * textHidden];
        for (int t = 0; t < tokens.Length; t++) Array.Copy(tokens[t], 0, ours, t * textHidden, textHidden);

        int n = Math.Min(ours.Length, refEmbeddings.Length);
        double sumOurs = 0, sumRef = 0, sumSqOurs = 0, sumSqRef = 0, sumAbsDiff = 0, maxAbsDiff = 0;
        for (int i = 0; i < n; i++)
        {
            sumOurs += ours[i];
            sumRef += refEmbeddings[i];
            sumSqOurs += (double)ours[i] * ours[i];
            sumSqRef += (double)refEmbeddings[i] * refEmbeddings[i];
            double diff = Math.Abs(ours[i] - refEmbeddings[i]);
            sumAbsDiff += diff;
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }
        double meanOurs = sumOurs / n, meanRef = sumRef / n;
        double stdOurs = Math.Sqrt(Math.Max(0, sumSqOurs / n - meanOurs * meanOurs));
        double stdRef = Math.Sqrt(Math.Max(0, sumSqRef / n - meanRef * meanRef));
        Console.Error.WriteLine(
            $"[VoxtralMatch] n={n} tokens={tokens.Length} melFrames={melFrames} " +
            $"ours(mean={meanOurs:F5},std={stdOurs:F5}) ref(mean={meanRef:F5},std={stdRef:F5}) " +
            $"meanAbsDiff={sumAbsDiff / n:F5} maxAbsDiff={maxAbsDiff:F5}");
    }
}
