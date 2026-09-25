using System.Diagnostics;
using System.Numerics.Tensors;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Real-weight parity of <see cref="TransformerEncoder"/> (BERT family, F32 safetensors) against each
/// checkpoint's own <c>onnx/model.onnx</c> run through onnxruntime: the same token ids go into both,
/// and <c>last_hidden_state</c> is compared per token (max abs diff and cosine). Also checks that a
/// batch of different-length inputs gives exactly the per-input results. Checkpoints come from
/// <c>models/_models/hf</c>; a missing one skips visibly. Timings are printed so a real run is
/// distinguishable from a no-op.
/// </summary>
public sealed class BertEncoderOnnxParityTests
{
    internal static readonly string[] Texts =
    [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France? Paris is the capital and most populous city of France.",
        "Crème brûlée à São Paulo — naïve façade, Ångström, 東京タワー, 🦙 emoji and ﬁle ligatures.",
        string.Join(" ", Enumerable.Repeat("A fairly long sentence that repeats to make a realistic passage length for retrieval.", 8)),
    ];

    internal static string? FindRepoDir(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    internal static string? FindOnnx(string dir) =>
        new[] { Path.Combine(dir, "onnx", "model.onnx"), Path.Combine(dir, "model.onnx") }.FirstOrDefault(File.Exists);

    internal static float[] RunOnnxHidden(OnnxModelSession onnx, EncodedInput input)
    {
        int n = input.Ids.Length;
        var ids = input.Ids.Select(i => (long)i).ToArray();
        var mask = Enumerable.Repeat(1L, n).ToArray();
        var types = input.TypeIds.Select(i => (long)i).ToArray();
        var outputs = onnx.Run(("input_ids", ids, [1, n]), ("attention_mask", mask, [1, n]), ("token_type_ids", types, [1, n]));
        return outputs.TryGetValue("last_hidden_state", out var h) ? h : outputs.Values.First();
    }

    [Theory]
    [InlineData("BAAI__bge-small-en-v1.5")]
    [InlineData("sentence-transformers__all-MiniLM-L6-v2")]
    [InlineData("intfloat__multilingual-e5-small")]
    [InlineData("sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2")]
    [InlineData("BAAI__bge-large-en-v1.5")]
    public void LastHiddenState_MatchesOnnx(string repoDir)
    {
        string? dir = FindRepoDir($"models/_models/hf/{repoDir}");
        string? onnxPath = dir is null ? null : FindOnnx(dir);
        Assert.SkipUnless(onnxPath != null, $"{repoDir} checkpoint/onnx not found");

        var sw = Stopwatch.StartNew();
        using var pipeline = HfEncoderEmbeddingPipeline.Load(dir!);
        using var onnx = new OnnxModelSession(onnxPath!);
        Console.WriteLine($"[BertParity] {repoDir}: onnx outputs {string.Join(",", onnx.OutputNames)}");
        Console.WriteLine($"[BertParity] {repoDir}: loaded in {sw.ElapsedMilliseconds} ms");

        sw.Restart();
        var ours = pipeline.EncodeHidden(Texts, out var inputs);
        long oursMs = sw.ElapsedMilliseconds;
        int h = pipeline.EmbeddingDimensions;

        float worstAbs = 0f, worstCos = 1f;
        long onnxMs = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            sw.Restart();
            var reference = RunOnnxHidden(onnx, inputs[i]);
            onnxMs += sw.ElapsedMilliseconds;
            Assert.Equal(reference.Length, ours[i].Length);
            for (int t = 0; t < inputs[i].Ids.Length; t++)
            {
                var a = ours[i].AsSpan(t * h, h);
                var b = reference.AsSpan(t * h, h);
                worstCos = MathF.Min(worstCos, TensorPrimitives.CosineSimilarity(a, b));
                for (int k = 0; k < h; k++) worstAbs = MathF.Max(worstAbs, MathF.Abs(a[k] - b[k]));
            }
        }
        Console.WriteLine($"[BertParity] {repoDir}: {inputs.Sum(x => x.Ids.Length)} tokens, ours {oursMs} ms, onnx {onnxMs} ms, maxAbs={worstAbs:E3} minCos={worstCos:F7}");
        Assert.True(worstCos >= 0.9999f, $"min token cosine {worstCos}");
        Assert.True(worstAbs <= 1e-3f, $"max abs diff {worstAbs}");

        // Batched (packed) == one at a time, bit for bit.
        for (int i = 0; i < inputs.Length; i++)
        {
            var single = pipeline.EncodeHidden([Texts[i]], out _)[0];
            Assert.Equal(single, ours[i]);
        }
    }
}
