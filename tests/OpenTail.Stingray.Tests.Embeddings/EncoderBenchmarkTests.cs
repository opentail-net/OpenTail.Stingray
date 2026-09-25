using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Throughput of <see cref="TransformerEncoder"/> vs ONNX Runtime (each checkpoint's own <c>onnx/model.onnx</c>,
/// default CPU session) on the same machine: 32 passages of ~128 tokens, as one batch (ONNX: padded
/// [32, L] with an attention mask; ours: packed without padding) and one at a time. Median of 5 timed runs
/// after a warm-up. Heavy: runs only with <c>STINGRAY_RUN_HEAVY_TESTS=1</c>.
/// </summary>
public sealed class EncoderBenchmarkTests
{
    private static readonly string[] s_passages = Enumerable.Range(0, 32).Select(i =>
        $"Passage {i}: " + string.Join(" ", Enumerable.Range(0, 9).Select(j =>
            $"The {((i + j) % 7) switch { 0 => "river", 1 => "library", 2 => "engine", 3 => "forest", 4 => "market", 5 => "theory", _ => "harbor" }} " +
            $"number {i * 9 + j} changes slowly over time."))).ToArray();

    private static double Median(List<double> xs) { xs.Sort(); return xs[xs.Count / 2]; }

    [Theory]
    [InlineData("sentence-transformers__all-MiniLM-L6-v2")]
    [InlineData("BAAI__bge-small-en-v1.5")]
    [InlineData("BAAI__bge-large-en-v1.5")]
    [InlineData("nomic-ai__nomic-embed-text-v1.5")]
    public void Throughput_VsOnnxRuntime(string repoDir)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1", "benchmark: set STINGRAY_RUN_HEAVY_TESTS=1");
        string? dir = BertEncoderOnnxParityTests.FindRepoDir($"models/_models/hf/{repoDir}");
        string? onnxPath = dir is null ? null : BertEncoderOnnxParityTests.FindOnnx(dir);
        Assert.SkipUnless(onnxPath != null, $"{repoDir} checkpoint/onnx not found");

        using var pipeline = HfEncoderEmbeddingPipeline.Load(dir!);
        using var encoder = TransformerEncoder.Load(dir!);
        using var onnx = new OnnxModelSession(onnxPath!);
        var inputs = s_passages.Select(p => pipeline.Tokenizer.Encode(p, 512)).ToArray();
        int tokens = inputs.Sum(x => x.Ids.Length), maxLen = inputs.Max(x => x.Ids.Length);

        double Time(Action a)
        {
            a();
            var runs = new List<double>();
            for (int r = 0; r < 5; r++) { var sw = Stopwatch.StartNew(); a(); runs.Add(sw.Elapsed.TotalMilliseconds); }
            return Median(runs);
        }

        double oursBatch = Time(() => encoder.EncodeBatch(inputs));
        double oursSingle = Time(() => { foreach (var x in inputs) encoder.Encode(x); });

        int b = inputs.Length;
        var ids = new long[b * maxLen];
        var mask = new long[b * maxLen];
        var types = new long[b * maxLen];
        for (int s = 0; s < b; s++)
            for (int t = 0; t < inputs[s].Ids.Length; t++)
            {
                ids[s * maxLen + t] = inputs[s].Ids[t];
                mask[s * maxLen + t] = 1;
            }
        double onnxBatch = Time(() => onnx.Run(("input_ids", ids, [b, maxLen]), ("attention_mask", mask, [b, maxLen]), ("token_type_ids", types, [b, maxLen])));
        double onnxSingle = Time(() =>
        {
            foreach (var x in inputs)
            {
                int n = x.Ids.Length;
                onnx.Run(("input_ids", x.Ids.Select(v => (long)v).ToArray(), [1, n]), ("attention_mask", Enumerable.Repeat(1L, n).ToArray(), [1, n]),
                    ("token_type_ids", new long[n], [1, n]));
            }
        });

        Console.WriteLine($"[EncBench] {repoDir}: {b} passages, {tokens} tokens (max {maxLen}) | " +
            $"batch: ours {oursBatch:F0} ms ({b * 1000 / oursBatch:F1} emb/s) vs ORT {onnxBatch:F0} ms ({b * 1000 / onnxBatch:F1} emb/s) | " +
            $"one-by-one: ours {oursSingle:F0} ms vs ORT {onnxSingle:F0} ms");
    }
}
