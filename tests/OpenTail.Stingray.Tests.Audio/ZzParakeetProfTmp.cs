using System.Diagnostics;
using OpenTail.Stingray.Audio.Parakeet;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch (untracked): Parakeet TDT stage timings. ZZ_PK=1 [ZZ_PK_MODEL=path] [ZZ_PK_RUNS=n].
public sealed class ZzParakeetProfTmp
{
    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ZZ_PK") != "1") return;
        string root = "C:/Git-Public/OpenTail.Stingray/";
        string model = Environment.GetEnvironmentVariable("ZZ_PK_MODEL") ?? root + "models/_models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-q4_k.gguf";
        string wav = root + "examples/audio.cpp/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0001.wav";
        int runs = int.Parse(Environment.GetEnvironmentVariable("ZZ_PK_RUNS") ?? "5");
        var sw = Stopwatch.StartNew();
        using var w = new ParakeetWeights(model);
        Console.WriteLine($"load {sw.ElapsedMilliseconds} ms  managed {GC.GetTotalMemory(true) / 1e9:F2} GB");
        var tok = ParakeetTokenizer.FromGguf(w.Model);
        var melx = ParakeetMelExtractor.FromWeights(w);
        var (samples, sr, _) = WavReader.ReadWav(wav);
        for (int r = 0; r < runs; r++)
        {
            sw.Restart();
            var mel = melx.ExtractMel(samples);
            long tMel = sw.ElapsedMilliseconds;
            var (hidden, _, tEnc) = ParakeetConformerEncoder.Forward(w, mel, mel.Length / melx.MelCount);
            long tEncMs = sw.ElapsedMilliseconds - tMel;
            string text = w.Tdt is null ? "(ctc model, no decode)                   " : new ParakeetTdtDecoder(tok, w).DecodeGreedy(hidden, tEnc, TimeSpan.Zero).FullText;
            long tDec = sw.ElapsedMilliseconds - tMel - tEncMs;
            Console.WriteLine($"run {r}: mel {tMel} ms, encoder {tEncMs} ms, decode {tDec} ms, total {sw.ElapsedMilliseconds} ms | {text[..40]}");
        }
    }
}
