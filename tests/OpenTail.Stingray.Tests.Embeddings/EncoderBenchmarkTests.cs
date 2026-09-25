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

    /// <summary>
    /// The same 32 passages through the vendored <c>llama-server --embedding</c> (C++/ggml, CPU) on F32 and Q8_0
    /// GGUFs of the same checkpoint: one 32-input request (all sequences in one ubatch, like our packed batch) and
    /// 32 single-input requests. Includes localhost HTTP + JSON overhead, which the one-by-one numbers feel most.
    /// Threads: llama.cpp's default (physical cores) unless <c>STINGRAY_BENCH_LLAMA_THREADS</c> is set.
    /// </summary>
    [Theory]
    [InlineData("sentence-transformers__all-MiniLM-L6-v2", "all-MiniLM-L6-v2.F32.gguf", "all-MiniLM-L6-v2.Q8_0.gguf")]
    [InlineData("BAAI__bge-small-en-v1.5", "bge-small-en-v1.5-f32.gguf", "bge-small-en-v1.5-q8_0.gguf")]
    [InlineData("BAAI__bge-large-en-v1.5", "bge-large-en-v1.5-f32.gguf", "bge-large-en-v1.5-q8_0.gguf")]
    [InlineData("nomic-ai__nomic-embed-text-v1.5", "nomic-embed-text-v1.5.f32.gguf", "nomic-embed-text-v1.5.Q8_0.gguf")]
    public void Throughput_VsLlamaCpp(string repoDir, string f32Gguf, string q8Gguf)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1", "benchmark: set STINGRAY_RUN_HEAVY_TESTS=1");
        string? dir = BertEncoderOnnxParityTests.FindRepoDir($"models/_models/hf/{repoDir}");
        string? ggufDir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/encoder-gguf");
        string? exe = LlamaServerOracle.FindServerExe();
        Assert.SkipUnless(dir != null && ggufDir != null && exe != null, $"{repoDir} checkpoint, models/_models/encoder-gguf or llama-server not found");

        using var pipeline = HfEncoderEmbeddingPipeline.Load(dir!);
        using var encoder = TransformerEncoder.Load(dir!);
        var inputs = s_passages.Select(p => pipeline.Tokenizer.Encode(p, 512)).ToArray();
        int tokens = inputs.Sum(x => x.Ids.Length);

        double Time(Action a)
        {
            a();
            var runs = new List<double>();
            for (int r = 0; r < 5; r++) { var sw = Stopwatch.StartNew(); a(); runs.Add(sw.Elapsed.TotalMilliseconds); }
            return Median(runs);
        }

        double oursBatch = Time(() => encoder.EncodeBatch(inputs));
        double oursSingle = Time(() => { foreach (var x in inputs) encoder.Encode(x); });
        Console.WriteLine($"[LlamaBench] {repoDir}: {inputs.Length} passages, {tokens} tokens | ours F32: batch {oursBatch:F0} ms, one-by-one {oursSingle:F0} ms");

        string? threads = Environment.GetEnvironmentVariable("STINGRAY_BENCH_LLAMA_THREADS");
        foreach (string gguf in new[] { f32Gguf, q8Gguf })
        {
            string path = Path.Combine(ggufDir!, gguf);
            Assert.SkipUnless(File.Exists(path), $"{path} not found");
            var extra = new List<string> { "-np", "32" };
            if (threads is not null) extra.AddRange(["-t", threads, "-tb", threads]);
            using var server = LlamaServerOracle.Start(exe!, path, "--embedding", extraArgs: extra);
            int llamaTokens = server.EmbedAndCountTokens(s_passages);
            double batch = Time(() => server.EmbedAndCountTokens(s_passages));
            double single = Time(() => { foreach (var p in s_passages) server.EmbedAndCountTokens([p]); });
            Console.WriteLine($"[LlamaBench] {repoDir}: llama.cpp {gguf} ({llamaTokens} tokens): batch {batch:F0} ms, one-by-one {single:F0} ms");
        }
    }
}
