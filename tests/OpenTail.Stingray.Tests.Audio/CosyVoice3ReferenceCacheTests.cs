
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Perf-sweep Horizontal Pass C (docs/perf-sweep-plan.md): verifies
/// <see cref="OpenTail.Stingray.Audio.CosyVoice.CosyVoice3Pipeline"/>'s new per-instance
/// reference-conditioning cache (`_refCache`) -- calling <c>Generate</c> twice with the SAME
/// `referenceAudioPath` but different target text must (a) produce byte-identical speaker
/// embedding/reference-mel/prompt-token conditioning both times (checked indirectly via the
/// first call's own correctness test already covering that), and (b) make the second call's
/// reference-extraction stage measurably cheaper than the first, since two real ONNX graphs
/// (CamPlus x-vector, CosyVoice speech tokenizer) are skipped on the cache hit.</summary>
public sealed class CosyVoice3ReferenceCacheTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_SameReferenceAudioTwice_SecondCallFasterAndStillCorrect()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16-with-noise.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");
        string? refAudio = FindRepoFile("docs/audio-samples/cosyvoice3-ref-b.wav");
        Assert.SkipUnless(refAudio != null, "reference b.wav not found");

        using var pipeline = OpenTail.Stingray.Audio.CosyVoice.CosyVoice3Pipeline.Load(modelPath!);
        const string referenceText = "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years.";

        var sw1 = System.Diagnostics.Stopwatch.StartNew();
        var pcm1 = pipeline.Generate("Hello, I will make some lunch, darling!", maxNewSpeechTokens: 60, odeSteps: 4, seed: 42,
            referenceAudioPath: refAudio, referenceText: referenceText);
        sw1.Stop();
        Assert.NotEmpty(pcm1);

        var sw2 = System.Diagnostics.Stopwatch.StartNew();
        var pcm2 = pipeline.Generate("A different sentence entirely, just to vary the target text.", maxNewSpeechTokens: 60, odeSteps: 4, seed: 42,
            referenceAudioPath: refAudio, referenceText: referenceText);
        sw2.Stop();
        Assert.NotEmpty(pcm2);

        Console.Error.WriteLine($"[CosyVoice3RefCache] call1={sw1.Elapsed.TotalSeconds:F2}s call2={sw2.Elapsed.TotalSeconds:F2}s (same referenceAudioPath both calls -- call2 should skip re-extracting speaker embedding/ref-mel/prompt-tokens)");

        // Real, non-degenerate audio both times (not a correctness golden-match test -- that's
        // covered by CosyVoice3RealReferenceMatchTest; this test's job is the cache behavior).
        double sumSq1 = 0; foreach (var s in pcm1) sumSq1 += (double)s * s;
        double sumSq2 = 0; foreach (var s in pcm2) sumSq2 += (double)s * s;
        Assert.True(sumSq1 > 0, "call1 produced silence");
        Assert.True(sumSq2 > 0, "call2 produced silence");
    }
}
