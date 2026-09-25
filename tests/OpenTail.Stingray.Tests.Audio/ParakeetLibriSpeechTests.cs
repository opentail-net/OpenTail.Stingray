namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Parakeet CTC 0.6B on the 4 audio.cpp LibriSpeech validation clips (ground truth next to each wav), per checkpoint
/// quantization (cstr/parakeet-ctc-0.6b-GGUF: q4_k and the unquantized file), to separate model-code errors from
/// quantization errors. Word error rate by edit distance.
/// </summary>
public sealed class ParakeetLibriSpeechTests
{
    private static string? Find(string rel)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, rel);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int WordEdits(string[] a, string[] b)
    {
        var d = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
        for (int j = 0; j <= b.Length; j++) d[0, j] = j;
        for (int i = 1; i <= a.Length; i++)
            for (int j = 1; j <= b.Length; j++)
                d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
        return d[a.Length, b.Length];
    }

    private static string[] Words(string s) =>
        new string(s.ToUpperInvariant().Where(c => char.IsLetterOrDigit(c) || c == '\'' || char.IsWhiteSpace(c)).ToArray())
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    // CrispASR (the reference implementation for these GGUFs, examples/CrispASR, built 2026-09-25) on the unquantized
    // checkpoint, casing and punctuation stripped. Before the tokenizer fix we dropped "a" (id 3) in 0001 twice.
    private static readonly Dictionary<string, string> s_crispAsr = new()
    {
        ["librispeech_test_clean_6930-75918-0000"] = "concord returned to its place amidst the tents",
        ["librispeech_test_clean_6930-75918-0001"] = "the english forwarded to the french baskets of flowers of which they had made a plentiful provision to greet the arrival of the young princess the french in return invited the english to a supper which was to be given the next day",
        ["librispeech_test_other_7902-96591-0000"] = "i'm from the cutter lying off the coast",
        ["librispeech_test_other_7902-96591-0001"] = "don't cry he said i was obliged to come",
    };

    [Theory]
    [InlineData("parakeet-ctc-0.6b-q4_k.gguf", 0.06)]
    [InlineData("parakeet-ctc-0.6b-f16.gguf", 0.06)]
    public void LibriSpeech_WordErrorRate(string file, double maxWer)
    {
        string? model = Find($"models/_models/{file}") ?? Find($"models/{file}");
        string? clipDir = Find("examples/audio.cpp/assets/asr_validation/librispeech");
        Assert.SkipUnless(model != null && clipDir != null, $"{file} or the LibriSpeech clips not found");

        using var pipeline = OpenTail.Stingray.Audio.Parakeet.ParakeetPipeline.Load(model!);
        int edits = 0, words = 0;
        foreach (var wav in Directory.GetFiles(clipDir!, "*.wav").Order())
        {
            var (samples, sr, _) = WavReader.ReadWav(wav);
            if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
            var result = pipeline.Transcribe(new SpeechToTextRequest { AudioSamples = samples, SampleRate = 16000, Language = "en", Task = SpeechTask.Transcribe });
            var truth = Words(File.ReadAllText(Path.ChangeExtension(wav, ".txt")));
            int e = WordEdits(truth, Words(result.Text));
            edits += e;
            words += truth.Length;
            Console.WriteLine($"[Parakeet {file}] {Path.GetFileNameWithoutExtension(wav)}: {e}/{truth.Length} edits\n  ours: {result.Text}");
            Assert.Equal(string.Join(' ', Words(s_crispAsr[Path.GetFileNameWithoutExtension(wav)])), string.Join(' ', Words(result.Text)));
        }
        double wer = (double)edits / words;
        Console.WriteLine($"[Parakeet {file}] WER {wer:P1} ({edits}/{words})");
        Assert.True(wer <= maxWer, $"WER {wer:P1}");
    }

}
