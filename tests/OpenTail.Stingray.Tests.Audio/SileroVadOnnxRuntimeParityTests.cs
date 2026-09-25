namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Golden parity: our native <see cref="OpenTail.Stingray.Audio.Vad.SileroVad"/> vs onnxruntime
/// running the same `models/silero_vad.onnx` (onnxruntime used only as the oracle), per 512-sample
/// frame of a real 16 kHz clip padded with 1s of silence each side. Feeds onnxruntime the official
/// v5 input (64-sample context + 512-sample frame, carried `state`), the convention whose absence in
/// the native port made probabilities differ by up to 0.75 before 2026-09-25.
/// </summary>
public sealed class SileroVadOnnxRuntimeParityTests : HeavyTestBase
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

    [Fact]
    public void NativeSilero_MatchesOnnxRuntime_PerFrame_OnRealSpeech()
    {
        string? onnx = FindRepoFile("models/silero_vad.onnx");
        Assert.SkipUnless(onnx != null, "models/silero_vad.onnx not found");
        string? wav = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(wav != null, "reference a.wav not found");
        using var ort = OpenTail.Stingray.Core.OnnxModelSession.TryLoad(onnx);
        Assert.SkipUnless(ort is { IsAvailable: true }, "onnxruntime not available");

        var (samples, sr, _) = WavReader.ReadWav(wav!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
        var audio = new float[samples.Length + 32000];
        samples.CopyTo(audio, 16000);

        using var ours = OpenTail.Stingray.Audio.Vad.SileroVad.Load(onnx!);
        var state = new float[2 * 128];
        var context = new float[64];
        double maxDiff = 0;
        int frames = 0, agree = 0, speech = 0;
        for (int off = 0; off + 512 <= audio.Length; off += 512, frames++)
        {
            var frame = audio.AsSpan(off, 512);
            float p = ours.ProcessFrame(frame);

            var x = new float[576];
            context.CopyTo(x, 0);
            frame.CopyTo(x.AsSpan(64));
            var outs = ort!.Run(("input", x, new[] { 1, 576 }), ("state", state, new[] { 2, 1, 128 }), ("sr", new long[] { 16000 }, Array.Empty<int>()));
            float q = outs["output"][0];
            state = outs["stateN"];
            frame.Slice(512 - 64).CopyTo(context);

            maxDiff = Math.Max(maxDiff, Math.Abs(p - q));
            if ((p > 0.5f) == (q > 0.5f)) agree++;
            if (q > 0.5f) speech++;
        }
        Console.WriteLine($"[SileroParity] frames {frames}, speech {speech}, maxAbsDiff {maxDiff:E2}, agreement {agree}/{frames}");

        Assert.True(speech > frames / 4 && speech < frames, $"clip should contain both speech and silence (speech frames {speech}/{frames})");
        Assert.Equal(frames, agree);
        Assert.True(maxDiff < 1e-4, $"per-frame probability differs from onnxruntime by {maxDiff}");
    }
}
