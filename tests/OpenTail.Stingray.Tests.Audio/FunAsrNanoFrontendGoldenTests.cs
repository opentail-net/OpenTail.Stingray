
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real golden-verification test for the Fun-ASR-Nano-2512 mel+LFR frontend against
/// `examples/audio.cpp/tests/fun_asr_nano/frontend_reference.{json,bin}` -- a real fixture
/// generated directly from the real `transformers.FunAsrNanoFeatureExtractor`, with real synthetic
/// waveforms (silence/impulse/sine) AND their real expected mel+LFR output embedded together, so
/// no external audio file or C++ build is needed.</summary>
public sealed class FunAsrNanoFrontendGoldenTests : HeavyTestBase
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

    [Theory]
    [InlineData("silence")]
    [InlineData("impulse")]
    [InlineData("sine_440hz")]
    public void ExtractLogMelAndLfr_MatchesRealTransformersReference(string fixtureName)
    {
        string? jsonPath = FindRepoFile("examples/audio.cpp/tests/fun_asr_nano/frontend_reference.json");
        string? binPath = FindRepoFile("examples/audio.cpp/tests/fun_asr_nano/frontend_reference.bin");
        Assert.SkipUnless(jsonPath != null && binPath != null, "frontend_reference.{json,bin} not found");

        var data = ReadAllF32(binPath!);
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(jsonPath!));
        var fixture = doc.RootElement.GetProperty("fixtures").GetProperty(fixtureName);

        int frames = fixture.GetProperty("frames").GetInt32();
        int width = fixture.GetProperty("width").GetInt32();
        var waveform = ReadFlat(data, fixture.GetProperty("waveform"));
        var expected = ReadFlat(data, fixture.GetProperty("features"));

        var extractor = new OpenTail.Stingray.Audio.FunASR.FunAsrRealMelExtractor();
        var logMel = extractor.ExtractLogMel(waveform, waveformScale: 1f);
        var lfr = OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoder.ApplyRealLfr(logMel, lfrM: 7, lfrN: 6);

        Assert.Equal(frames, lfr.Length);
        var actual = new float[frames * width];
        for (int i = 0; i < frames; i++) Array.Copy(lfr[i], 0, actual, i * width, width);

        double maxAbsDiff = 0;
        for (int i = 0; i < actual.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(actual[i] - expected[i]));
        Console.Error.WriteLine($"[FunAsrNanoFrontendGolden] {fixtureName} frames={frames} width={width} maxAbsDiff={maxAbsDiff:F6}");
        Assert.True(maxAbsDiff < 0.05, $"{fixtureName} maxAbsDiff {maxAbsDiff} too high vs real transformers reference");
    }

    private static float[] ReadAllF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static float[] ReadFlat(float[] data, System.Text.Json.JsonElement entry)
    {
        int offset = entry.GetProperty("offset_f32").GetInt32();
        int count = entry.GetProperty("count").GetInt32();
        var result = new float[count];
        Array.Copy(data, offset, result, 0, count);
        return result;
    }
}
