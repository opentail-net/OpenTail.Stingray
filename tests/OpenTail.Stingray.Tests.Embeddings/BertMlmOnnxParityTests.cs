using System.Diagnostics;
using System.Numerics.Tensors;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// google-bert/bert-base-uncased: its ONNX export (<c>model.onnx</c>) is <c>BertForMaskedLM</c> and outputs
/// only MLM <c>logits</c>, so the encoder is checked through the checkpoint's own MLM head applied to our
/// last hidden state: <c>logits = LN(gelu(W·h + b)) · word_embeddings^T + bias</c> (HF
/// <c>BertLMPredictionHead</c>, decoder tied to the word embeddings). This checkpoint also exercises the
/// <c>bert.</c> prefix and the TF-era <c>LayerNorm.gamma/beta</c> tensor names. FacebookAI/xlm-roberta-base
/// (<c>XLMRobertaForMaskedLM</c>, <c>lm_head.*</c>, same head shape) is checked the same way and covers the
/// RoBERTa position offset (pad + 1). Includes the model cards' fill-mask prompt, whose documented top
/// prediction is "fashion" for both.
/// </summary>
public sealed class BertMlmOnnxParityTests
{
    [Theory]
    [InlineData("google-bert__bert-base-uncased", "bert.", "cls.predictions.transform.dense", "cls.predictions.transform.LayerNorm", "cls.predictions.bias", 103, "[MASK]")]
    [InlineData("FacebookAI__xlm-roberta-base", "roberta.", "lm_head.dense", "lm_head.layer_norm", "lm_head.bias", 250001, "<mask>")]
    public void MlmLogits_MatchOnnx(string repoDir, string prefix, string headDense, string headLn, string decoderBias, int maskId, string maskText)
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir($"models/_models/hf/{repoDir}");
        string? onnxPath = dir is null ? null : BertEncoderOnnxParityTests.FindOnnx(dir);
        Assert.SkipUnless(onnxPath != null, $"{repoDir} checkpoint/onnx not found");

        var sw = Stopwatch.StartNew();
        using var encoder = TransformerEncoder.Load(dir!);
        Assert.Equal(prefix, encoder.Prefix);
        var tok = EncoderTokenizer.FromTokenizerJson(Path.Combine(dir!, "tokenizer.json"));
        using var st = SafetensorsLoader.OpenDirectory(dir!);
        int h = encoder.Config.HiddenSize, vocab = encoder.Config.VocabSize;
        var dense = st.ReadF32(headDense + ".weight");
        var denseB = st.ReadF32(headDense + ".bias");
        var (lnW, lnB) = TransformerEncoder.ReadLayerNorm(st, headLn);
        var wordEmb = st.ReadF32(prefix + "embeddings.word_embeddings.weight");
        var decBias = st.ReadF32(decoderBias);
        var pieces = JsonDocumentPieces(Path.Combine(dir!, "tokenizer.json"));
        using var onnx = new OnnxModelSession(onnxPath!);
        Console.WriteLine($"[Mlm] {repoDir}: loaded in {sw.ElapsedMilliseconds} ms; onnx outputs {string.Join(",", onnx.OutputNames)}");

        // Model card prompt "Hello I'm a {maskText} model." with the mask id spliced in.
        var left = tok.Encode("Hello I'm a");
        var right = tok.Encode("model.");
        int[] maskedIds = [.. left.Ids[..^1], maskId, .. right.Ids[1..]];
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
                int maskPos = Array.IndexOf(maskedIds, maskId);
                int best = TensorPrimitives.IndexOfMax(ours.AsSpan(maskPos * vocab, vocab));
                Console.WriteLine($"[Mlm] {repoDir}: 'Hello I'm a {maskText} model.' top = {best} '{pieces[best]}' (model cards: 'fashion')");
                Assert.Equal("fashion", pieces[best].TrimStart('▁'));
            }
        }
        Console.WriteLine($"[Mlm] {repoDir}: {inputs.Length} inputs: logits maxAbs={worstAbs:E3} minCos={worstCos:F7}, total {sw.ElapsedMilliseconds} ms");
        Assert.True(worstCos >= 0.9999f, $"min cosine {worstCos}");
        Assert.True(worstAbs <= 2e-3f, $"max abs diff {worstAbs}");
    }

    /// <summary>Id → piece from a tokenizer.json vocab (WordPiece dict or Unigram [piece, score] list).</summary>
    private static string[] JsonDocumentPieces(string tokenizerJson)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        var vocab = doc.RootElement.GetProperty("model").GetProperty("vocab");
        if (vocab.ValueKind == System.Text.Json.JsonValueKind.Array)
            return vocab.EnumerateArray().Select(e => e[0].GetString() ?? "").ToArray();
        var pieces = new string[vocab.EnumerateObject().Max(p => p.Value.GetInt32()) + 1];
        foreach (var p in vocab.EnumerateObject()) pieces[p.Value.GetInt32()] = p.Name;
        return pieces;
    }
}
