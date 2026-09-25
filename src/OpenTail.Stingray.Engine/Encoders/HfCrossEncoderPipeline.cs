using System.Numerics.Tensors;
using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Cross-encoder reranker from a HF <c>*ForSequenceClassification</c> checkpoint: each (query, document)
/// pair is tokenized as one pair input (BERT <c>[CLS] q [SEP] d [SEP]</c>, XLM-R <c>&lt;s&gt; q &lt;/s&gt;&lt;/s&gt; d &lt;/s&gt;</c>),
/// encoded, and scored by the checkpoint's classification head on the first token:
/// <list type="bullet">
/// <item>RoBERTa/XLM-R (<c>RobertaClassificationHead</c>): <c>out_proj(tanh(dense(h[0])))</c>.</item>
/// <item>BERT (<c>BertForSequenceClassification</c>): <c>classifier(tanh(pooler(h[0])))</c>.</item>
/// </list>
/// The score is the raw logit of label 0 (what both ms-marco-MiniLM and bge-reranker-v2-m3 document);
/// <see cref="ApplySigmoid"/> maps it to (0, 1) like bge's <c>normalize=True</c>.
/// </summary>
public sealed class HfCrossEncoderPipeline : IRerankerPipeline
{
    private readonly TransformerEncoder _encoder;
    private readonly float[] _denseW, _denseB, _outW, _outB;
    private readonly int _labels;

    public EncoderTokenizer Tokenizer { get; }
    public string ModelName { get; }
    /// <summary>Pairs are truncated (longest-first) to this many tokens, special tokens included.</summary>
    public int MaxSequenceLength { get; init; }

    /// <summary>When true, scores are sigmoid(logit) instead of the raw logit.</summary>
    public bool ApplySigmoid { get; init; }

    private HfCrossEncoderPipeline(string name, EncoderTokenizer tokenizer, TransformerEncoder encoder,
        float[] denseW, float[] denseB, float[] outW, float[] outB, int labels)
    {
        ModelName = name;
        Tokenizer = tokenizer;
        _encoder = encoder;
        (_denseW, _denseB, _outW, _outB, _labels) = (denseW, denseB, outW, outB, labels);
        MaxSequenceLength = encoder.Config.MaxSequenceLength;
    }

    public static HfCrossEncoderPipeline Load(string modelDir, int? maxSequenceLength = null)
    {
        var config = EncoderConfig.FromFile(Path.Combine(modelDir, "config.json"));
        using var st = SafetensorsLoader.OpenDirectory(modelDir);
        var encoder = TransformerEncoder.Load(config, st);
        string p = encoder.Prefix;
        float[] denseW, denseB, outW, outB;
        if (st.Contains("classifier.out_proj.weight"))
        {
            (denseW, denseB) = (st.ReadF32("classifier.dense.weight"), st.ReadF32("classifier.dense.bias"));
            (outW, outB) = (st.ReadF32("classifier.out_proj.weight"), st.ReadF32("classifier.out_proj.bias"));
        }
        else if (st.Contains("classifier.weight") && st.Contains(p + "pooler.dense.weight"))
        {
            (denseW, denseB) = (st.ReadF32(p + "pooler.dense.weight"), st.ReadF32(p + "pooler.dense.bias"));
            (outW, outB) = (st.ReadF32("classifier.weight"), st.ReadF32("classifier.bias"));
        }
        else
        {
            encoder.Dispose();
            throw new InvalidDataException($"'{modelDir}' has no sequence-classification head (classifier.* tensors).");
        }
        int labels = outB.Length;
        var tokenizer = EncoderTokenizer.FromTokenizerJson(Path.Combine(modelDir, "tokenizer.json"));
        string name = Path.GetFileName(Path.TrimEndingDirectorySeparator(modelDir));
        return new HfCrossEncoderPipeline(name, tokenizer, encoder, denseW, denseB, outW, outB, labels)
        {
            MaxSequenceLength = Math.Min(maxSequenceLength ?? int.MaxValue, encoder.Config.MaxSequenceLength),
        };
    }

    /// <summary>All label logits per (query, document) pair, [documents][labels].</summary>
    public float[][] Logits(string query, IReadOnlyList<string> documents)
    {
        var inputs = documents.Select(d => Tokenizer.EncodePair(query, d, MaxSequenceLength)).ToArray();
        var hidden = _encoder.EncodeBatch(inputs);
        int h = _encoder.Config.HiddenSize;
        var result = new float[documents.Count][];
        var pooled = new float[h];
        for (int i = 0; i < hidden.Length; i++)
        {
            var first = hidden[i].AsSpan(0, h);
            for (int o = 0; o < h; o++) pooled[o] = MathF.Tanh(TensorPrimitives.Dot(_denseW.AsSpan(o * h, h), first) + _denseB[o]);
            var logits = new float[_labels];
            for (int l = 0; l < _labels; l++) logits[l] = TensorPrimitives.Dot(_outW.AsSpan(l * h, h), pooled) + _outB[l];
            result[i] = logits;
        }
        return result;
    }

    /// <summary>Relevance score per document, in input order.</summary>
    public float[] Score(string query, IReadOnlyList<string> documents) =>
        Logits(query, documents).Select(l => ApplySigmoid ? 1f / (1f + MathF.Exp(-l[0])) : l[0]).ToArray();

    public RerankResult Rerank(RerankRequest request)
    {
        var scores = Score(request.Query, request.Documents);
        int tokens = request.Documents.Sum(d => Tokenizer.EncodePair(request.Query, d, MaxSequenceLength).Ids.Length);
        var ranked = scores.Select((s, i) => new RerankDocumentResult
        {
            Index = i,
            RelevanceScore = s,
            Document = request.ReturnDocuments ? request.Documents[i] : null,
        }).OrderByDescending(r => r.RelevanceScore).ToList();
        if (request.TopN is int n && n >= 0 && n < ranked.Count) ranked = ranked.Take(n).ToList();
        return new RerankResult(ModelName, ranked, tokens);
    }

    public void Dispose() => _encoder.Dispose();
}
