namespace OpenTail.Stingray.Tests.Audio;

// Scratch (untracked): GGUF Paraformer (FunAsrPipeline) on a real Mandarin clip. ZZ_FUN=1.
public sealed class ZzFunAsrZhTmp
{
    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ZZ_FUN") != "1") return;
        using var p = OpenTail.Stingray.Audio.FunASR.FunAsrPipeline.Load(@"C:\Git-Public\OpenTail.Stingray\models\_models\paraformer-q8.gguf");
        var (s, r, _) = WavReader.ReadWav(@"C:\Git-Public\OpenTail.Stingray\docs\audio-samples\paraformer-zh-test-0.wav");
        var res = p.Transcribe(new SpeechToTextRequest { AudioSamples = s, SampleRate = r, Language = "zh" });
        Console.WriteLine($"[funasr-gguf] '{res.Text}'");
    }
}
