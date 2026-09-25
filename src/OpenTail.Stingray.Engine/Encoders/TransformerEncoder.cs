using System.Numerics.Tensors;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Bidirectional post-LN transformer encoder (HF <c>BertModel</c> and the RoBERTa/XLM-R family), CPU F32,
/// loaded from a HF checkpoint directory (<c>config.json</c> + <c>model.safetensors</c>).
///
/// <para>Math, per HF <c>modeling_bert.py</c> and llama.cpp <c>src/models/bert.cpp</c>:
/// embeddings = LayerNorm(word[id] + position[offset + i] + token_type[t]); each layer:
/// <c>h = LN(x + Wo·MHA(x))</c>, <c>x' = LN(h + W2·gelu(W1·h))</c>, attention scaled by 1/√headDim.</para>
///
/// <para>Batches are packed without padding: every Linear runs once over all tokens of all sequences
/// (one GEMM per weight), and attention runs per sequence over its own tokens only. That makes a batched
/// result identical to encoding each input alone, with no attention mask to get wrong.</para>
/// </summary>
public sealed class TransformerEncoder : IDisposable
{
    private readonly float[] _wordEmb, _posEmb;
    private readonly float[]? _typeEmb;
    private readonly float[] _embLnW, _embLnB;
    private readonly Layer[] _layers;

    public EncoderConfig Config { get; }

    /// <summary>Tensor-name prefix of the encoder inside the checkpoint ("" or e.g. "bert.").</summary>
    public string Prefix { get; }

    private sealed class Layer : IDisposable
    {
        public required PackedLinearF32 Qkv, AttnOut, Ffn1, Ffn2;
        public required float[] AttnLnW, AttnLnB, OutLnW, OutLnB;

        public void Dispose()
        {
            Qkv.Dispose(); AttnOut.Dispose(); Ffn1.Dispose(); Ffn2.Dispose();
        }
    }

    private TransformerEncoder(EncoderConfig config, string prefix, float[] word, float[] pos, float[]? type,
        float[] lnW, float[] lnB, Layer[] layers)
    {
        Config = config;
        Prefix = prefix;
        _wordEmb = word;
        _posEmb = pos;
        _typeEmb = type;
        _embLnW = lnW;
        _embLnB = lnB;
        _layers = layers;
    }

    private static readonly string[] s_prefixes = ["", "bert.", "roberta.", "electra.", "model."];

    /// <summary>Finds the encoder prefix in a checkpoint (e.g. "bert." for <c>BertForSequenceClassification</c>).</summary>
    public static string DetectPrefix(SafetensorsLoader st) =>
        s_prefixes.FirstOrDefault(p => st.Contains(p + "embeddings.word_embeddings.weight"))
        ?? throw new InvalidDataException("No BERT-style 'embeddings.word_embeddings.weight' tensor found.");

    /// <summary>Reads a LayerNorm's (weight, bias), accepting the old TF-era gamma/beta names.</summary>
    public static (float[] W, float[] B) ReadLayerNorm(SafetensorsLoader st, string name) =>
        st.Contains(name + ".weight")
            ? (st.ReadF32(name + ".weight"), st.ReadF32(name + ".bias"))
            : (st.ReadF32(name + ".gamma"), st.ReadF32(name + ".beta"));

    public static TransformerEncoder Load(string modelDir)
    {
        var config = EncoderConfig.FromFile(Path.Combine(modelDir, "config.json"));
        using var st = SafetensorsLoader.OpenDirectory(modelDir);
        return Load(config, st);
    }

    public static TransformerEncoder Load(EncoderConfig c, SafetensorsLoader st)
    {
        if (c.HiddenAct != "gelu")
            throw new NotSupportedException($"hidden_act '{c.HiddenAct}' is not supported yet (only exact-erf 'gelu').");
        string p = DetectPrefix(st);
        int h = c.HiddenSize;
        var word = st.ReadF32(p + "embeddings.word_embeddings.weight");
        var pos = st.ReadF32(p + "embeddings.position_embeddings.weight");
        float[]? type = c.TypeVocabSize > 0 && st.Contains(p + "embeddings.token_type_embeddings.weight")
            ? st.ReadF32(p + "embeddings.token_type_embeddings.weight")
            : null;
        var (lnW, lnB) = ReadLayerNorm(st, p + "embeddings.LayerNorm");

        var layers = new Layer[c.NumLayers];
        for (int i = 0; i < c.NumLayers; i++)
        {
            string l = $"{p}encoder.layer.{i}.";
            float[] R(string n) => st.ReadF32(l + n);
            // Fuse Q, K, V into one [3H, H] weight so each layer runs a single GEMM for all three.
            var qkvW = new float[3 * h * h];
            var qkvB = new float[3 * h];
            string[] names = ["query", "key", "value"];
            for (int j = 0; j < 3; j++)
            {
                R($"attention.self.{names[j]}.weight").CopyTo(qkvW, j * h * h);
                R($"attention.self.{names[j]}.bias").CopyTo(qkvB, j * h);
            }
            var (aW, aB) = ReadLayerNorm(st, l + "attention.output.LayerNorm");
            var (oW, oB) = ReadLayerNorm(st, l + "output.LayerNorm");
            layers[i] = new Layer
            {
                Qkv = new PackedLinearF32(qkvW, qkvB, 3 * h, h),
                AttnOut = new PackedLinearF32(R("attention.output.dense.weight"), R("attention.output.dense.bias"), h, h),
                Ffn1 = new PackedLinearF32(R("intermediate.dense.weight"), R("intermediate.dense.bias"), c.IntermediateSize, h),
                Ffn2 = new PackedLinearF32(R("output.dense.weight"), R("output.dense.bias"), h, c.IntermediateSize),
                AttnLnW = aW, AttnLnB = aB, OutLnW = oW, OutLnB = oB,
            };
        }
        return new TransformerEncoder(c, p, word, pos, type, lnW, lnB, layers);
    }

    /// <summary>Last hidden state of one input, row-major [tokens, hidden].</summary>
    public float[] Encode(EncodedInput input) => EncodeBatch([input])[0];

    /// <summary>Last hidden state per input, each row-major [tokens_i, hidden].</summary>
    public float[][] EncodeBatch(IReadOnlyList<EncodedInput> batch)
    {
        int h = Config.HiddenSize;
        var offsets = new int[batch.Count + 1];
        for (int s = 0; s < batch.Count; s++)
        {
            int len = batch[s].Ids.Length;
            if (len > Config.MaxSequenceLength)
                throw new ArgumentException($"Input {s} has {len} tokens; this model takes at most {Config.MaxSequenceLength}.");
            offsets[s + 1] = offsets[s] + len;
        }
        int t = offsets[^1];

        var x = new float[t * h];
        Embed(batch, offsets, x);

        var qkv = new float[t * 3 * h];
        var ctx = new float[t * h];
        var tmp = new float[t * h];
        var ffn = new float[t * Config.IntermediateSize];
        foreach (var layer in _layers)
        {
            layer.Qkv.Forward(x, qkv, t);
            Attention(qkv, ctx, offsets);
            layer.AttnOut.Forward(ctx, tmp, t);
            AddLayerNorm(tmp, x, layer.AttnLnW, layer.AttnLnB, t);   // x = LN(x + attn)
            layer.Ffn1.Forward(x, ffn, t);
            Parallel.For(0, t, r => ErfGelu.InPlace(ffn.AsSpan(r * Config.IntermediateSize, Config.IntermediateSize)));
            layer.Ffn2.Forward(ffn, tmp, t);
            AddLayerNorm(tmp, x, layer.OutLnW, layer.OutLnB, t);    // x = LN(x + ffn)
        }

        var result = new float[batch.Count][];
        for (int s = 0; s < batch.Count; s++)
            result[s] = x.AsSpan(offsets[s] * h, (offsets[s + 1] - offsets[s]) * h).ToArray();
        return result;
    }

    private void Embed(IReadOnlyList<EncodedInput> batch, int[] offsets, float[] x)
    {
        int h = Config.HiddenSize;
        Parallel.For(0, batch.Count, s =>
        {
            var ids = batch[s].Ids;
            var types = batch[s].TypeIds;
            for (int i = 0; i < ids.Length; i++)
            {
                if ((uint)ids[i] >= (uint)Config.VocabSize)
                    throw new ArgumentException($"Token id {ids[i]} is outside the vocab ({Config.VocabSize}).");
                var row = x.AsSpan((offsets[s] + i) * h, h);
                TensorPrimitives.Add(_wordEmb.AsSpan(ids[i] * h, h), _posEmb.AsSpan((Config.PositionOffset + i) * h, h), row);
                if (_typeEmb is not null)
                {
                    int type = types.Length > i ? types[i] : 0;
                    TensorPrimitives.Add(row, _typeEmb.AsSpan(type * h, h), row);
                }
                LayerNormInPlace(row, _embLnW, _embLnB, Config.LayerNormEps);
            }
        });
    }

    /// <summary>x[r] = LN(x[r] + delta[r]) for every row.</summary>
    private void AddLayerNorm(float[] delta, float[] x, float[] w, float[] b, int rows)
    {
        int h = Config.HiddenSize;
        float eps = Config.LayerNormEps;
        Parallel.For(0, rows, r =>
        {
            var row = x.AsSpan(r * h, h);
            TensorPrimitives.Add(row, delta.AsSpan(r * h, h), row);
            LayerNormInPlace(row, w, b, eps);
        });
    }

    private static void LayerNormInPlace(Span<float> row, float[] w, float[] b, float eps)
    {
        float mean = TensorPrimitives.Sum(row) / row.Length;
        TensorPrimitives.Subtract(row, mean, row);
        float variance = TensorPrimitives.SumOfSquares(row) / row.Length;
        TensorPrimitives.Multiply(row, 1f / MathF.Sqrt(variance + eps), row);
        TensorPrimitives.FusedMultiplyAdd(row, w, b, row);
    }

    /// <summary>Softmax(QK^T/√d)·V per sequence and head; qkv rows are [Q | K | V] (each hidden wide).</summary>
    private void Attention(float[] qkv, float[] ctx, int[] offsets)
    {
        int h = Config.HiddenSize, d = Config.HeadDim, heads = Config.NumHeads, stride = 3 * h;
        float scale = 1f / MathF.Sqrt(d);
        int seqs = offsets.Length - 1;
        const int QueryBlock = 16;
        var work = new List<(int S, int Head, int Q0)>();
        for (int s = 0; s < seqs; s++)
            for (int hd = 0; hd < heads; hd++)
                for (int q0 = 0; q0 < offsets[s + 1] - offsets[s]; q0 += QueryBlock)
                    work.Add((s, hd, q0));

        Parallel.For(0, work.Count, () => new float[offsets.Length > 1 ? MaxLen(offsets) : 0], (wi, _, scores) =>
        {
            var (s, hd, q0) = work[wi];
            int o = offsets[s], len = offsets[s + 1] - o;
            int q1 = Math.Min(len, q0 + QueryBlock);
            for (int i = q0; i < q1; i++)
            {
                var q = qkv.AsSpan((o + i) * stride + hd * d, d);
                var sc = scores.AsSpan(0, len);
                for (int j = 0; j < len; j++)
                    sc[j] = TensorPrimitives.Dot(q, qkv.AsSpan((o + j) * stride + h + hd * d, d)) * scale;
                float max = TensorPrimitives.Max(sc);
                TensorPrimitives.Subtract(sc, max, sc);
                TensorPrimitives.Exp(sc, sc);
                TensorPrimitives.Divide(sc, TensorPrimitives.Sum(sc), sc);
                var outRow = ctx.AsSpan((o + i) * h + hd * d, d);
                outRow.Clear();
                for (int j = 0; j < len; j++)
                    TensorPrimitives.FusedMultiplyAdd(qkv.AsSpan((o + j) * stride + 2 * h + hd * d, d), sc[j], outRow, outRow);
            }
            return scores;
        }, _ => { });
    }

    private static int MaxLen(int[] offsets)
    {
        int m = 0;
        for (int s = 0; s + 1 < offsets.Length; s++) m = Math.Max(m, offsets[s + 1] - offsets[s]);
        return m;
    }

    public void Dispose()
    {
        foreach (var l in _layers) l.Dispose();
    }
}
