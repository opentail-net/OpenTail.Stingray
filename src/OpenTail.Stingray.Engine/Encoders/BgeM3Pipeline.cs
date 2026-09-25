using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>One text's BGE-M3 outputs: dense (CLS, L2-normalized), sparse lexical weights (token id → weight) and
/// ColBERT multi-vectors (one L2-normalized row per non-CLS token, [count, dim]).</summary>
public sealed record BgeM3Output(float[] Dense, Dictionary<int, float> LexicalWeights, float[] ColbertVecs, int ColbertCount);

/// <summary>
/// BAAI/bge-m3's three retrieval heads over the shared XLM-R encoder, as FlagEmbedding's <c>BGEM3FlagModel</c> computes
/// them: dense = normalized CLS; sparse = <c>relu(sparse_linear(h))</c> per token, max per token id, special tokens
/// (cls/eos/pad/unk) dropped; ColBERT = <c>normalize(colbert_linear(h[1:]))</c> over every non-CLS token. The two head
/// weights ship as PyTorch zip checkpoints (<c>sparse_linear.pt</c>, <c>colbert_linear.pt</c>), read with
/// <see cref="TorchCheckpointReader"/>.
/// </summary>
public sealed class BgeM3Pipeline : IDisposable
{
    private readonly HfEncoderEmbeddingPipeline _encoder;
    private readonly float[] _sparseW;
    private readonly float _sparseB;
    private readonly PackedLinearF32 _colbert;
    private readonly HashSet<int> _unusedTokens;

    public HfEncoderEmbeddingPipeline Encoder => _encoder;

    private BgeM3Pipeline(HfEncoderEmbeddingPipeline encoder, float[] sparseW, float sparseB, PackedLinearF32 colbert, HashSet<int> unused)
    {
        (_encoder, _sparseW, _sparseB, _colbert, _unusedTokens) = (encoder, sparseW, sparseB, colbert, unused);
    }

    public static bool IsBgeM3Directory(string dir) =>
        File.Exists(Path.Combine(dir, "sparse_linear.pt")) && File.Exists(Path.Combine(dir, "colbert_linear.pt"));

    public static BgeM3Pipeline Load(string dir)
    {
        var encoder = HfEncoderEmbeddingPipeline.Load(dir);
        try
        {
            int h = encoder.EmbeddingDimensions;
            var sparse = TorchCheckpointReader.Read(Path.Combine(dir, "sparse_linear.pt"));
            var colbert = TorchCheckpointReader.Read(Path.Combine(dir, "colbert_linear.pt"));
            var sw = Expect(sparse, "weight", 1, h);
            var sb = Expect(sparse, "bias", 1);
            var cw = colbert["weight"];
            if (cw.Shape is not [var colbertDim, var cin] || cin != h)
                throw new InvalidDataException($"colbert_linear.weight: shape [{string.Join(",", cw.Shape)}], expected [dim, {h}].");
            var cb = Expect(colbert, "bias", colbertDim);
            return new BgeM3Pipeline(encoder, sw.Data, sb.Data[0], new PackedLinearF32(cw.Data, cb.Data, colbertDim, h), ReadSpecialTokenIds(dir));
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    private static TorchTensor Expect(Dictionary<string, TorchTensor> d, string name, params int[] shape)
    {
        if (!d.TryGetValue(name, out var t)) throw new InvalidDataException($"missing tensor '{name}'.");
        if (!t.Shape.SequenceEqual(shape))
            throw new InvalidDataException($"'{name}': shape [{string.Join(",", t.Shape)}], expected [{string.Join(",", shape)}].");
        return t;
    }

    /// <summary>FlagEmbedding's <c>unused_tokens</c>: cls, eos, pad and unk, looked up in <c>tokenizer.json</c>'s added tokens.</summary>
    private static HashSet<int> ReadSpecialTokenIds(string dir)
    {
        var ids = new HashSet<int>();
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "tokenizer.json")));
        foreach (var t in doc.RootElement.GetProperty("added_tokens").EnumerateArray())
            if (t.GetProperty("content").GetString() is "<s>" or "</s>" or "<pad>" or "<unk>")
                ids.Add(t.GetProperty("id").GetInt32());
        if (ids.Count != 4) throw new InvalidDataException("tokenizer.json: expected <s>, </s>, <pad>, <unk> among added_tokens.");
        return ids;
    }

    public BgeM3Output[] Encode(IReadOnlyList<string> texts)
    {
        var hidden = _encoder.EncodeHidden(texts, out var inputs);
        int h = _encoder.EmbeddingDimensions, dim = _colbert.OutDim;
        var result = new BgeM3Output[texts.Count];
        for (int i = 0; i < texts.Count; i++)
        {
            int n = inputs[i].Ids.Length;
            var hs = hidden[i];

            var dense = hs.AsSpan(0, h).ToArray();
            HfEncoderEmbeddingPipeline.L2NormalizeInPlace(dense);

            var lexical = new Dictionary<int, float>();
            for (int t = 0; t < n; t++)
            {
                int id = inputs[i].Ids[t];
                float w = Math.Max(0f, TensorPrimitives.Dot(hs.AsSpan(t * h, h), _sparseW) + _sparseB);
                if (_unusedTokens.Contains(id) || w <= 0f) continue;
                if (!lexical.TryGetValue(id, out float prev) || w > prev) lexical[id] = w;
            }

            int m = n - 1;
            var colbert = new float[Math.Max(m, 0) * dim];
            if (m > 0)
            {
                _colbert.Forward(hs.AsSpan(h, m * h), colbert, m);
                for (int t = 0; t < m; t++) HfEncoderEmbeddingPipeline.L2NormalizeInPlace(colbert.AsSpan(t * dim, dim));
            }
            result[i] = new BgeM3Output(dense, lexical, colbert, m);
        }
        return result;
    }

    public static float DenseScore(BgeM3Output q, BgeM3Output p) => TensorPrimitives.Dot(q.Dense, p.Dense);

    /// <summary><c>compute_lexical_matching_score</c>: sum of w_q · w_p over token ids present in both.</summary>
    public static float SparseScore(BgeM3Output q, BgeM3Output p)
    {
        float s = 0;
        foreach (var (id, w) in q.LexicalWeights)
            if (p.LexicalWeights.TryGetValue(id, out float wp)) s += w * wp;
        return s;
    }

    /// <summary><c>colbert_score</c>: for each query vector the best dot product over the passage vectors, averaged.</summary>
    public static float ColbertScore(BgeM3Output q, BgeM3Output p)
    {
        if (q.ColbertCount == 0 || p.ColbertCount == 0) return 0f;
        int dim = q.ColbertVecs.Length / q.ColbertCount;
        float sum = 0;
        for (int a = 0; a < q.ColbertCount; a++)
        {
            var qa = q.ColbertVecs.AsSpan(a * dim, dim);
            float best = float.NegativeInfinity;
            for (int b = 0; b < p.ColbertCount; b++) best = Math.Max(best, TensorPrimitives.Dot(qa, p.ColbertVecs.AsSpan(b * dim, dim)));
            sum += best;
        }
        return sum / q.ColbertCount;
    }

    /// <summary><c>compute_score</c>'s weighted combination (weights for dense, sparse, colbert; FlagEmbedding's example uses
    /// 0.4/0.2/0.4): <c>sparse+dense</c> and <c>colbert+sparse+dense</c> are weighted means over the modes they include.</summary>
    public static (float Dense, float Sparse, float Colbert, float SparseDense, float All) Score(BgeM3Output q, BgeM3Output p,
        float wDense = 0.4f, float wSparse = 0.2f, float wColbert = 0.4f)
    {
        float d = DenseScore(q, p), s = SparseScore(q, p), c = ColbertScore(q, p);
        return (d, s, c, (wDense * d + wSparse * s) / (wDense + wSparse),
            (wDense * d + wSparse * s + wColbert * c) / (wDense + wSparse + wColbert));
    }

    public void Dispose()
    {
        _colbert.Dispose();
        _encoder.Dispose();
    }
}
