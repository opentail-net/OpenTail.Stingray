using System.Numerics.Tensors;
using OpenTail.Stingray.Audio.Wav2Vec2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// <see cref="Wav2Vec2CtcModel"/> on facebook/wav2vec2-base-960h against (a) Xenova/wav2vec2-base-960h's ONNX export of
/// the same weights (CTC logits over the whole clip; the ONNX takes the already-normalized <c>input_values</c>) and
/// (b) LibriSpeech ground truth for the 4 audio.cpp validation clips (word error rate, greedy CTC).
/// </summary>
public sealed class Wav2Vec2CtcRealWeightsTests
{
    private static string? FindRepoPath(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static readonly string[] Clips =
    [
        "librispeech_test_clean_6930-75918-0000", "librispeech_test_clean_6930-75918-0001",
        "librispeech_test_other_7902-96591-0000", "librispeech_test_other_7902-96591-0001",
    ];

    private static float[] Load16k(string path)
    {
        var (samples, sr, ch) = WavReader.ReadWav(path);
        Assert.Equal(16000, sr);
        Assert.Equal(1, ch);
        return samples;
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

    [Fact]
    public void LibriSpeech_LogitsMatchOnnx_AndTranscribe()
    {
        string? dir = FindRepoPath("models/_models/hf/facebook__wav2vec2-base-960h");
        string? onnxPath = FindRepoPath("models/_models/hf/Xenova__wav2vec2-base-960h/onnx/model.onnx");
        string? clipDir = FindRepoPath("examples/audio.cpp/assets/asr_validation/librispeech");
        Assert.SkipUnless(dir != null && onnxPath != null && clipDir != null, "wav2vec2-base-960h, its ONNX, or the LibriSpeech clips not found");

        using var model = Wav2Vec2CtcModel.Load(dir!);
        using var onnx = new OnnxModelSession(onnxPath!);
        int totalWords = 0, totalEdits = 0;
        foreach (string clip in Clips)
        {
            var audio = Load16k(Path.Combine(clipDir!, clip + ".wav"));
            string truth = File.ReadAllText(Path.Combine(clipDir!, clip + ".txt")).Trim();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var logits = model.Logits(audio, out int frames);
            long ms = sw.ElapsedMilliseconds;

            double mean = audio.Average(x => (double)x);
            double var = audio.Average(x => (x - mean) * (x - mean));
            var normalized = audio.Select(x => (float)((x - mean) / Math.Sqrt(var + 1e-7))).ToArray();
            var reference = onnx.Run(("input_values", normalized, [1, normalized.Length]))["logits"];
            Assert.Equal(reference.Length, logits.Length);
            int v = model.VocabSize, agree = 0;
            float maxAbs = 0, minCos = 1;
            for (int f = 0; f < frames; f++)
            {
                var a = logits.AsSpan(f * v, v);
                var b = reference.AsSpan(f * v, v);
                if (TensorPrimitives.IndexOfMax(a) == TensorPrimitives.IndexOfMax(b)) agree++;
                minCos = Math.Min(minCos, TensorPrimitives.CosineSimilarity(a, b));
                for (int i = 0; i < v; i++) maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
            }

            string ours = model.DecodeGreedy(logits, frames);
            string onnxText = model.DecodeGreedy(reference, frames);
            var tw = truth.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int edits = WordEdits(tw, ours.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            totalWords += tw.Length;
            totalEdits += edits;
            Console.WriteLine($"[Wav2Vec2] {clip} ({audio.Length / 16000.0:F1}s, {frames} frames, {ms} ms): logits maxAbs {maxAbs:E2} min cos {minCos:F6} argmax {agree}/{frames}\n" +
                $"  ours : {ours}\n  truth: {truth}\n  word edits {edits}/{tw.Length}");
            Assert.Equal(onnxText, ours);
            Assert.True(minCos > 0.9999f, $"{clip}: logits cosine {minCos}");
        }
        double wer = (double)totalEdits / totalWords;
        Console.WriteLine($"[Wav2Vec2] WER over 4 clips: {wer:P1} ({totalEdits}/{totalWords})");
        Assert.True(wer < 0.15, $"WER {wer}");
    }
}
