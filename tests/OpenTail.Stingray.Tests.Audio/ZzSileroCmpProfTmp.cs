namespace OpenTail.Stingray.Tests.Audio;

// Scratch harness (untracked): our native SileroVad vs onnxruntime on the same silero_vad.onnx, per
// 512-sample frame of a real 16 kHz clip. ZZ_SILERO=1, ZZ_WAV optional.
public sealed class ZzSileroCmpProfTmp
{
    [Fact]
    public void Silero_NativeVsOnnxRuntime_PerFrame()
    {
        if (Environment.GetEnvironmentVariable("ZZ_SILERO") != "1") return;
        string onnx = @"C:\Git-Public\OpenTail.Stingray\models\silero_vad.onnx";
        string wav = Environment.GetEnvironmentVariable("ZZ_WAV") ?? @"C:\Git-Public\OpenTail.Stingray\examples\audio.cpp\assets\resources\a.wav";
        var (samples, sr, _) = WavReader.ReadWav(wav);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
        // 1s of silence on each side so both speech and non-speech frames are exercised.
        var audio = new float[samples.Length + 32000];
        samples.CopyTo(audio, 16000);

        using var ort = new OpenTail.Stingray.Core.OnnxModelSession(onnx);
        Console.WriteLine($"[ZZ silero] onnx inputs: {string.Join(",", ort.InputNames)}  outputs: {string.Join(",", ort.OutputNames)}");
        using var ours = OpenTail.Stingray.Audio.Vad.SileroVad.Load(onnx);

        var state = new float[2 * 1 * 128];
        var context = new float[64];
        double maxDiff = 0; int n = 0, oursSpeech = 0, refSpeech = 0, agree = 0;
        for (int off = 0; off + 512 <= audio.Length; off += 512, n++)
        {
            var frame = audio.AsSpan(off, 512);
            float p = ours.ProcessFrame(frame);

            var x = new float[576];
            context.CopyTo(x, 0);
            frame.CopyTo(x.AsSpan(64));
            var outs = ort.Run(("input", x, new[] { 1, 576 }), ("state", state, new[] { 2, 1, 128 }), ("sr", new long[] { 16000 }, Array.Empty<int>()));
            float q = outs["output"][0];
            state = outs["stateN"];
            frame.Slice(512 - 64).CopyTo(context);

            maxDiff = Math.Max(maxDiff, Math.Abs(p - q));
            if (p > 0.5f) oursSpeech++;
            if (q > 0.5f) refSpeech++;
            if ((p > 0.5f) == (q > 0.5f)) agree++;
            if (n % 20 == 0) Console.WriteLine($"[ZZ silero] frame {n}: ours {p:F4} ort {q:F4}");
        }
        Console.WriteLine($"[ZZ silero] frames {n}: maxAbsDiff {maxDiff:E2}, speech frames ours {oursSpeech} / ort {refSpeech}, decision agreement {agree}/{n}");
    }
}
