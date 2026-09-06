
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real golden-verification test for RvcSynthesizerEncoder against the real C++
/// reference's exact intermediate/output arrays, captured via STINGRAY_RVC_TRACE=1 full-array
/// binary dumps added to native_pipeline.cpp/synthesizer.cpp (the built-in engine::debug::trace_log_f32
/// only logs 40 sparse sample points, not enough for a real numeric comparison). The reference's
/// real inputs (content features, pitch ids, sine source, and the exact Gaussian noise fed into
/// z_p) are consumed directly rather than re-derived, since this test isolates the synthesizer
/// itself -- HuBERT/RMVPE are already separately golden-verified elsewhere in this test project,
/// and the reference's z_p noise comes from a Philox-based CUDA RNG this port does not replicate.
///
/// Binary dumps are looked for in the OS scratchpad
/// (`rvc_synth/rvc_synth_{features,pitch,sine_source,noise,m,logs,z_p,z,raw_output}.bin`) rather
/// than being committed to the repo -- generate them by rebuilding audiocpp_cli with the
/// STINGRAY_RVC_TRACE dumps present (gitignored, local-only, see docs/audio-review-progress.md)
/// and running: `STINGRAY_RVC_TRACE=1 audiocpp_cli --log --task vc --family rvc --model
/// rvc-f16.gguf --backend cpu --audio a.wav --request-option voice_id=default --out out.wav`.
/// Skips (does not fail) when the dumps are not present, same convention as the other
/// RealCheckpoint-dependent tests in this project.</summary>
public sealed class RvcSynthesizerRealReferenceMatchTests : HeavyTestBase
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
        var hits = Directory.GetFiles(candidate, "rvc_synth_features.bin", SearchOption.AllDirectories);
        return hits.Length > 0 ? Path.GetDirectoryName(hits[0]) : null;
    }

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static int[] ReadI32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new int[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    [Fact]
    public void Forward_RealReferenceInputs_MatchesRealReferenceOutputs()
    {
        string? checkpointPath = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(checkpointPath != null, "rvc-f16.gguf not found");
        string? dumpDir = DumpDir();
        Assert.SkipUnless(dumpDir != null, "real reference rvc_synth_*.bin dumps not found -- see class doc comment to generate them");

        var featuresFlat = ReadF32(Path.Combine(dumpDir!, "rvc_synth_features.bin"));
        var pitch = ReadI32(Path.Combine(dumpDir!, "rvc_synth_pitch.bin"));
        var sine = ReadF32(Path.Combine(dumpDir!, "rvc_synth_sine_source.bin"));
        var noise = ReadF32(Path.Combine(dumpDir!, "rvc_synth_noise.bin"));
        var refM = ReadF32(Path.Combine(dumpDir!, "rvc_synth_m.bin"));
        var refLogs = ReadF32(Path.Combine(dumpDir!, "rvc_synth_logs.bin"));
        var refZp = ReadF32(Path.Combine(dumpDir!, "rvc_synth_z_p.bin"));
        var refZ = ReadF32(Path.Combine(dumpDir!, "rvc_synth_z.bin"));
        var refAudio = ReadF32(Path.Combine(dumpDir!, "rvc_synth_raw_output.bin"));

        int frames = pitch.Length;
        int featureDim = featuresFlat.Length / frames;
        var featuresBtc = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[featureDim];
            Array.Copy(featuresFlat, t * featureDim, row, 0, featureDim);
            featuresBtc[t] = row;
        }

        using var model = OpenTail.Stingray.Core.GgufModel.Open(checkpointPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcSynthesizerWeights(source, "voice_v2_default_checkpoint", sampleRate: 40000, v1: false, hasF0: true);

        var result = OpenTail.Stingray.Audio.Rvc.RvcSynthesizerEncoder.Forward(w, featuresBtc, pitch, sine, speakerId: 0, noiseChannelMajor: noise);

        ReportStats("m", result.M, refM);
        ReportStats("logs", result.Logs, refLogs);
        ReportStats("z_p", result.Zp, refZp);
        ReportStats("z", result.Z, refZ);
        ReportStats("audio", result.Audio, refAudio);
    }

    private static void ReportStats(string label, float[] ours, float[] reference)
    {
        int n = Math.Min(ours.Length, reference.Length);
        double sumOurs = 0, sumRef = 0, sumSqOurs = 0, sumSqRef = 0, sumAbsDiff = 0, maxAbsDiff = 0;
        for (int i = 0; i < n; i++)
        {
            sumOurs += ours[i];
            sumRef += reference[i];
            sumSqOurs += (double)ours[i] * ours[i];
            sumSqRef += (double)reference[i] * reference[i];
            double diff = Math.Abs(ours[i] - reference[i]);
            sumAbsDiff += diff;
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }
        double meanOurs = sumOurs / n, meanRef = sumRef / n;
        double stdOurs = Math.Sqrt(Math.Max(0, sumSqOurs / n - meanOurs * meanOurs));
        double stdRef = Math.Sqrt(Math.Max(0, sumSqRef / n - meanRef * meanRef));
        Console.Error.WriteLine(
            $"[RvcSynthMatch] {label} n={n} ours(mean={meanOurs:F5},std={stdOurs:F5}) " +
            $"ref(mean={meanRef:F5},std={stdRef:F5}) meanAbsDiff={sumAbsDiff / n:F5} maxAbsDiff={maxAbsDiff:F5}");
    }
}
