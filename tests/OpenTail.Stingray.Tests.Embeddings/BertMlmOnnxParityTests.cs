using System.Diagnostics;
using System.Numerics.Tensors;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// google-bert/bert-base-uncased: its ONNX export (<c>model.onnx</c>) is <c>BertForMaskedLM</c> and outputs
/// only MLM <c>logits</c>, so the encoder is checked through the checkpoint's own MLM head applied to our
/// last hidden state: <c>logits = LN(gelu(W·h + b)) · word_embeddings^T + bias</c> (HF
/// <c>BertLMPredictionHead</c>, decoder tied to the word embeddings). This checkpoint also exercises the
/// <c>bert.</c> prefix and the TF-era <c>LayerNorm.gamma/beta</c> tensor names. Includes the model card's
/// fill-mask prompt, whose documented top prediction is "fashion".
/// </summary>
public sealed class BertMlmOnnxParityTests
{
    [Fact]
    public void BertBaseUncased_MlmLogits_MatchOnnx()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/google-bert__bert-base-uncased");
        string? onnxPath = dir is null ? null : BertEncoderOnnxParityTests.FindOnnx(dir);
        Assert.SkipUnless(onnxPath != null, "bert-base-uncased checkpoint/onnx not found");

        var sw = Stopwatch.StartNew();
        using var encoder = TransformerEncoder.Load(dir!);
        Assert.Equal("bert.", encoder.Prefix);
        var tok = EncoderTokenizer.FromTokenizerJson(Path.Combine(dir!, "tokenizer.json"));
        using var st = SafetensorsLoader.OpenDirectory(dir!);
        int h = encoder.Config.HiddenSize, vocab = encoder.Config.VocabSize;
        var dense = st.ReadF32("cls.predictions.transform.dense.weight");
        var denseB = st.ReadF32("cls.predictions.transform.dense.bias");
        var (lnW, lnB) = TransformerEncoder.ReadLayerNorm(st, "cls.predictions.transform.LayerNorm");
        var wordEmb = st.ReadF32("bert.embeddings.word_embeddings.weight");
        var decBias = st.ReadF32("cls.predictions.bias");
        using var onnx = new OnnxModelSession(onnxPath!);
        Console.WriteLine($"[BertMlm] loaded in {sw.ElapsedMilliseconds} ms");

        // Model card prompt "Hello I'm a [MASK] model." with [MASK] = 103 spliced in by id.
        var left = tok.Encode("Hello I'm a");
        var right = tok.Encode("model.");
        int[] maskedIds = [.. left.Ids[..^1], 103, .. right.Ids[1..]];
        var inputs = BertEncoderOnnxParityTests.Texts.Select(t => tok.Encode(t))
            .Append(new EncodedInput(maskedIds, new int[maskedIds.Length])).ToArray();

        float worstAbs = 0f, worstCos = 1f;
        foreach (var input in inputs)
        {
            var hidden = encoder.Encode(input);
            int n = input.Ids.Length;
            var ours = new float[n * vocab];
            var tmp = new float[h];
            for (int t = 0; t < n; t++)
            {
                for (int o = 0; o < h; o++) tmp[o] = TensorPrimitives.Dot(dense.AsSpan(o * h, h), hidden.AsSpan(t * h, h)) + denseB[o];
                ErfGelu.InPlace(tmp);
                float mean = TensorPrimitives.Average(tmp);
                TensorPrimitives.Subtract(tmp, mean, tmp);
                float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(tmp) / h + encoder.Config.LayerNormEps);
                TensorPrimitives.Multiply(tmp, inv, tmp);
                TensorPrimitives.FusedMultiplyAdd(tmp, lnW, lnB, tmp);
                for (int v = 0; v < vocab; v++) ours[t * vocab + v] = TensorPrimitives.Dot(wordEmb.AsSpan(v * h, h), tmp) + decBias[v];
            }
            var ids = input.Ids.Select(i => (long)i).ToArray();
            var reference = onnx.Run(("input_ids", ids, [1, n]), ("attention_mask", Enumerable.Repeat(1L, n).ToArray(), [1, n]),
                ("token_type_ids", new long[n], [1, n]))["logits"];
            Assert.Equal(reference.Length, ours.Length);
            for (int t = 0; t < n; t++)
            {
                var a = ours.AsSpan(t * vocab, vocab);
                var b = reference.AsSpan(t * vocab, vocab);
                worstCos = MathF.Min(worstCos, TensorPrimitives.CosineSimilarity(a, b));
                for (int k = 0; k < vocab; k++) worstAbs = MathF.Max(worstAbs, MathF.Abs(a[k] - b[k]));
            }
            if (input.Ids == maskedIds)
            {
                int maskPos = Array.IndexOf(maskedIds, 103);
                int best = TensorPrimitives.IndexOfMax(ours.AsSpan(maskPos * vocab, vocab));
                Console.WriteLine($"[BertMlm] fill-mask top id = {best} (model card: 4827 'fashion')");
                Assert.Equal(4827, best);
            }
        }
        Console.WriteLine($"[BertMlm] {inputs.Length} inputs: logits maxAbs={worstAbs:E3} minCos={worstCos:F7}, total {sw.ElapsedMilliseconds} ms");
        Assert.True(worstCos >= 0.9999f, $"min cosine {worstCos}");
        Assert.True(worstAbs <= 2e-3f, $"max abs diff {worstAbs}");
    }
}
