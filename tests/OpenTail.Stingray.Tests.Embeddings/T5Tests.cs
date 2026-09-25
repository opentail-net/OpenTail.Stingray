using System.Numerics.Tensors;
using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// T5 encoder-decoder (<see cref="T5Model"/>) against google-t5/t5-small's own ONNX exports
/// (<c>onnx/encoder_model.onnx</c>: last_hidden_state; <c>onnx/decoder_model.onnx</c>: logits for a full decoder
/// sequence, no cache), plus greedy generation receipts.
/// </summary>
public sealed class T5Tests
{
    internal static string? FindDir(string repo = "google-t5__t5-small") => BertEncoderOnnxParityTests.FindRepoDir($"models/_models/hf/{repo}");

    internal static string Detokenize(UnigramTokenizer u, IEnumerable<int> ids, int[] skip) =>
        string.Concat(ids.Where(i => !skip.Contains(i)).Select(i => u.Pieces[i])).Replace('▁', ' ').Trim();

    private static readonly string[] Prompts =
    [
        "translate English to German: The house is wonderful.",
        "summarize: The tower is 324 metres tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres on each side.",
        "translate English to French: Machine learning models run on many kinds of hardware.",
    ];

    [Theory]
    [InlineData("google-t5__t5-small")]   // ReLU FFN, tied embeddings (lm_head = shared, output scaled by d_model^-0.5)
    [InlineData("google__flan-t5-small")] // gated-GELU FFN (wi_0/wi_1), untied lm_head; ONNX from Xenova/flan-t5-small
    public void Encoder_And_Decoder_MatchOnnx(string repo)
    {
        string? dir = FindDir(repo);
        Assert.SkipUnless(dir != null && File.Exists(Path.Combine(dir, "onnx", "encoder_model.onnx")), $"{repo} checkpoint/onnx not found");

        using var model = T5Model.Load(dir!);
        var tok = EncoderTokenizer.FromTokenizerJson(Path.Combine(dir!, "tokenizer.json"));
        using var encOnnx = new OnnxModelSession(Path.Combine(dir!, "onnx", "encoder_model.onnx"));
        using var decOnnx = new OnnxModelSession(Path.Combine(dir!, "onnx", "decoder_model.onnx"));
        Console.WriteLine($"[T5] encoder onnx inputs: {string.Join(",", encOnnx.InputNames)}; decoder: {string.Join(",", decOnnx.InputNames)}");
        int d = model.Config.DModel, vocab = model.Config.VocabSize;

        foreach (string prompt in Prompts)
        {
            int[] ids = tok.Encode(prompt).Ids;
            int t = ids.Length;
            var ours = model.Encode(ids);
            long[] ids64 = ids.Select(x => (long)x).ToArray(), mask = Enumerable.Repeat(1L, t).ToArray();
            var refEnc = encOnnx.Run(("input_ids", ids64, [1, t]), ("attention_mask", mask, [1, t]))["last_hidden_state"];
            float encMax = MaxAbs(ours, refEnc);
            Assert.True(encMax < 1e-3f, $"encoder maxAbs {encMax}");

            // Decoder input: start token + our greedy continuation (teacher-forced through both).
            var gen = model.GenerateGreedy(ids, 24);
            int[] decIds = [model.Config.DecoderStartTokenId, .. gen];
            int n = decIds.Length;
            var full = model.StartDecoder(ours, t, n).Step(decIds);
            var logits = model.Logits(full, n);
            var refLogits = decOnnx.Run(("input_ids", decIds.Select(x => (long)x).ToArray(), [1, n]),
                ("encoder_attention_mask", mask, [1, t]), ("encoder_hidden_states", refEnc, [1, t, d]))["logits"];
            float logitMax = MaxAbs(logits, refLogits);
            float minCos = 1f;
            int argmaxAgree = 0;
            for (int i = 0; i < n; i++)
            {
                var a = logits.AsSpan(i * vocab, vocab);
                var b = refLogits.AsSpan(i * vocab, vocab);
                minCos = Math.Min(minCos, TensorPrimitives.CosineSimilarity(a, b));
                if (TensorPrimitives.IndexOfMax(a) == TensorPrimitives.IndexOfMax(b)) argmaxAgree++;
            }

            // Incremental (cached) decoding must equal the one-shot pass.
            var inc = model.StartDecoder(ours, t, n);
            float incMax = 0;
            for (int i = 0; i < n; i++)
                incMax = Math.Max(incMax, MaxAbs(inc.Step([decIds[i]]), full.AsSpan(i * d, d).ToArray()));

            Console.WriteLine($"[T5] {repo} t={t} n={n}: encoder maxAbs {encMax:E2}; logits maxAbs {logitMax:E2}, min cos {minCos:F7}, argmax {argmaxAgree}/{n}; cached-vs-full maxAbs {incMax:E2}");
            Assert.True(minCos > 0.99999f, $"logits cosine {minCos}");
            Assert.Equal(n, argmaxAgree);
            Assert.True(incMax < 1e-4f, $"cached decode differs by {incMax}");
        }
    }

    [Theory]
    [InlineData("google-t5__t5-small")]
    [InlineData("google__flan-t5-small")]
    public void GreedyGeneration_Receipts(string repo)
    {
        string? dir = FindDir(repo);
        Assert.SkipUnless(dir != null, $"{repo} checkpoint not found");
        using var model = T5Model.Load(dir!);
        var tok = EncoderTokenizer.FromTokenizerJson(Path.Combine(dir!, "tokenizer.json"));
        var u = UnigramTokenizer.FromTokenizerJson(Path.Combine(dir!, "tokenizer.json"));
        var outputs = Prompts.Select(p => Detokenize(u, model.GenerateGreedy(tok.Encode(p).Ids, 48), [0, 1, 2])).ToArray();
        for (int i = 0; i < Prompts.Length; i++) Console.WriteLine($"[T5] {Prompts[i]}\n     -> {outputs[i]}");
        // HF transformers docs' T5 example: "translate English to German: The house is wonderful." -> "Das Haus ist wunderbar."
        if (repo == "google-t5__t5-small") Assert.Equal("Das Haus ist wunderbar.", outputs[0]);
    }

    internal static float MaxAbs(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }
}
