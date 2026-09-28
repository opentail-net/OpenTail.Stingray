using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch harness (untracked): ASR engines on the audio.cpp LibriSpeech validation clips (ground
// truth in the .txt next to each wav). ZZ_ASR=qwen|parakeet.
public sealed class ZzAsrSweepProfTmp
{
    [Fact]
    public void Transcribe_LibriSpeechClips()
    {
        string? which = Environment.GetEnvironmentVariable("ZZ_ASR");
        if (which is null) return;
        string root = @"C:\Git-Public\OpenTail.Stingray";
        string clipDir = Path.Combine(root, "examples", "audio.cpp", "assets", "asr_validation", "librispeech");

        var sw = Stopwatch.StartNew();
        using ISpeechToTextPipeline pipeline = which switch
        {
            "qwen" => OpenTail.Stingray.Audio.QwenASR.QwenAsrPipeline.LoadFromSafetensors(Path.Combine(root, "models", "qwen3-asr-0.6b-hf")),
            "parakeet" => OpenTail.Stingray.Audio.Parakeet.ParakeetPipeline.Load(Path.Combine(root, "models", "parakeet-ctc-0.6b-q4_k.gguf")),
            _ => throw new ArgumentException(which),
        };
        Console.WriteLine($"[ZZ asr] {which} load {sw.Elapsed.TotalSeconds:F1}s");

        foreach (var wav in Directory.GetFiles(clipDir, "*.wav").Order())
        {
            var (samples, sr, _) = WavReader.ReadWav(wav);
            if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
            var request = new SpeechToTextRequest { AudioSamples = samples, SampleRate = 16000, Language = "en", Task = SpeechTask.Transcribe };
            sw.Restart();
            var result = pipeline.Transcribe(request);
            string truth = File.ReadAllText(Path.ChangeExtension(wav, ".txt")).Trim();
            Console.WriteLine($"[ZZ asr] {which} {Path.GetFileNameWithoutExtension(wav)} ({samples.Length / 16000.0:F1}s, {sw.Elapsed.TotalSeconds:F1}s)\n  truth: {truth}\n  ours:  {result.Text}");
        }
    }
}
