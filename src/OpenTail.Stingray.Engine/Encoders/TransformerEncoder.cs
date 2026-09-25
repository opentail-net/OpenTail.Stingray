using System.Numerics.Tensors;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Bidirectional post-LN transformer encoder, CPU F32, loaded from a HF checkpoint directory
/// (<c>config.json</c> + <c>model.safetensors</c>). Families (<see cref="EncoderFamily"/>):
/// <list type="bullet">
/// <item>BERT / ELECTRA / RoBERTa / XLM-R (HF <c>modeling_bert.py</c>, <c>modeling_roberta.py</c>; llama.cpp
/// <c>src/models/bert.cpp</c>): embeddings = LN(word + position[offset + i] + token_type); layer
/// <c>h = LN(x + Wo·MHA(x))</c>, <c>x' = LN(h + W2·gelu(W1·h))</c>.</item>
/// <item>MPNet (HF <c>modeling_mpnet.py</c>): as RoBERTa, no token types, plus a bucketed relative position bias
/// (<c>encoder.relative_attention_bias</c>, T5 bidirectional buckets, max distance 128) added to every layer's
/// scaled attention scores.</item>
/// <item>NomicBERT (vLLM <c>bert_with_rope.py</c>, llama.cpp <c>bert.cpp</c> NOMIC_BERT branch): no position table;
/// RoPE (NeoX halves, <c>rotary_emb_base</c>) on q/k; fused bias-free <c>Wqkv</c>; FFN
/// <c>fc2(silu(fc12·h) ⊙ fc11·h)</c>; post-LN <c>norm1</c>/<c>norm2</c>.</item>
/// </list>
///
/// <para>Batches are packed without padding: every Linear runs once over all tokens of all sequences
/// (one GEMM per weight), and attention runs per sequence over its own tokens only. That makes a batched
/// result identical to encoding each input alone, with no attention mask to get wrong.</para>
/// </summary>
public sealed class TransformerEncoder : IDisposable
{
    private readonly float[] _wordEmb;
    private readonly float[]? _posEmb, _typeEmb;
    private readonly float[] _embLnW, _embLnB;
    private readonly float[]? _relBias;   // MPNet: [buckets, heads]
    private readonly float[]? _ropeInvFreq; // NomicBERT: [headDim / 2]
    private readonly Layer[] _layers;

    public EncoderConfig Config { get; }

    /// <summary>Tensor-name prefix of the encoder inside the checkpoint ("" or e.g. "bert.").</summary>
    public string Prefix { get; }

    private sealed class Layer : IDisposable
    {
        public required PackedLinearF32 Qkv, AttnOut, Ffn1, Ffn2;
        public required float[] AttnLnW, AttnLnB, OutLnW, OutLnB;

        /// <summary>Ffn1 outputs [gate | up] (2 × intermediate) for SwiGLU instead of one GELU input.</summary>
        public bool SwiGlu;

        public void Dispose()
        {
            Qkv.Dispose(); AttnOut.Dispose(); Ffn1.Dispose(); Ffn2.Dispose();
        }
    }

    private TransformerEncoder(EncoderConfig config, string prefix, float[] word, float[]? pos, float[]? type,
        float[] lnW, float[] lnB, float[]? relBias, Layer[] layers)
    {
        Config = config;
        Prefix = prefix;
        _wordEmb = word;
        _posEmb = pos;
        _typeEmb = type;
        _embLnW = lnW;
        _embLnB = lnB;
        _relBias = relBias;
        _layers = layers;
        if (config.Family == EncoderFamily.NomicBert)
        {
            int half = config.HeadDim / 2;
            _ropeInvFreq = new float[half];
            for (int i = 0; i < half; i++) _ropeInvFreq[i] = 1f / MathF.Pow(config.RopeTheta, 2f * i / config.HeadDim);
        }
    }

    private static readonly string[] s_prefixes = ["", "bert.", "roberta.", "electra.", "mpnet.", "model."];

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
        if (c.Family != EncoderFamily.NomicBert && c.HiddenAct != "gelu")
            throw new NotSupportedException($"hidden_act '{c.HiddenAct}' is not supported yet (only exact-erf 'gelu').");
        string p = DetectPrefix(st);
        int h = c.HiddenSize;
        var word = st.ReadF32(p + "embeddings.word_embeddings.weight");
        float[]? pos = c.Family == EncoderFamily.NomicBert ? null : st.ReadF32(p + "embeddings.position_embeddings.weight");
        float[]? type = c.TypeVocabSize > 0 && st.Contains(p + "embeddings.token_type_embeddings.weight")
            ? st.ReadF32(p + "embeddings.token_type_embeddings.weight")
            : null;
        var (lnW, lnB) = c.Family == EncoderFamily.NomicBert ? ReadLayerNorm(st, p + "emb_ln") : ReadLayerNorm(st, p + "embeddings.LayerNorm");
        float[]? relBias = c.Family == EncoderFamily.MpNet ? st.ReadF32(p + "encoder.relative_attention_bias.weight") : null;
        if (relBias is not null && relBias.Length != c.RelativeAttentionBuckets * c.NumHeads)
            throw new InvalidDataException($"relative_attention_bias has {relBias.Length} floats, expected {c.RelativeAttentionBuckets}x{c.NumHeads}.");

        var layers = new Layer[c.NumLayers];
        for (int i = 0; i < c.NumLayers; i++)
            layers[i] = c.Family switch
            {
                EncoderFamily.NomicBert => LoadNomicLayer(c, st, $"{p}encoder.layers.{i}."),
                EncoderFamily.MpNet => LoadBertLayer(c, st, $"{p}encoder.layer.{i}.", "attention.attn.", ["q", "k", "v"], "attention.attn.o", "attention.LayerNorm"),
                _ => LoadBertLayer(c, st, $"{p}encoder.layer.{i}.", "attention.self.", ["query", "key", "value"], "attention.output.dense", "attention.output.LayerNorm"),
            };
        return new TransformerEncoder(c, p, word, pos, type, lnW, lnB, relBias, layers);
    }

    private static Layer LoadBertLayer(EncoderConfig c, SafetensorsLoader st, string l, string qkvPrefix, string[] qkvNames, string attnOut, string attnLn)
    {
        int h = c.HiddenSize;
        float[] R(string n) => st.ReadF32(l + n);
        // Fuse Q, K, V into one [3H, H] weight so each layer runs a single GEMM for all three.
        var qkvW = new float[3 * h * h];
        var qkvB = new float[3 * h];
        for (int j = 0; j < 3; j++)
        {
            R($"{qkvPrefix}{qkvNames[j]}.weight").CopyTo(qkvW, j * h * h);
            R($"{qkvPrefix}{qkvNames[j]}.bias").CopyTo(qkvB, j * h);
        }
        var (aW, aB) = ReadLayerNorm(st, l + attnLn);
        var (oW, oB) = ReadLayerNorm(st, l + "output.LayerNorm");
        return new Layer
        {
            Qkv = new PackedLinearF32(qkvW, qkvB, 3 * h, h),
            AttnOut = new PackedLinearF32(R(attnOut + ".weight"), R(attnOut + ".bias"), h, h),
            Ffn1 = new PackedLinearF32(R("intermediate.dense.weight"), R("intermediate.dense.bias"), c.IntermediateSize, h),
            Ffn2 = new PackedLinearF32(R("output.dense.weight"), R("output.dense.bias"), h, c.IntermediateSize),
            AttnLnW = aW, AttnLnB = aB, OutLnW = oW, OutLnB = oB,
        };
    }

    private static Layer LoadNomicLayer(EncoderConfig c, SafetensorsLoader st, string l)
    {
        int h = c.HiddenSize, inter = c.IntermediateSize;
        // [gate | up] = [fc12 | fc11] so one GEMM feeds SiluAndMul.
        var gateUp = new float[2 * inter * h];
        st.ReadF32(l + "mlp.fc12.weight").CopyTo(gateUp, 0);
        st.ReadF32(l + "mlp.fc11.weight").CopyTo(gateUp, inter * h);
        var (aW, aB) = ReadLayerNorm(st, l + "norm1");
        var (oW, oB) = ReadLayerNorm(st, l + "norm2");
        return new Layer
        {
            Qkv = new PackedLinearF32(st.ReadF32(l + "attn.Wqkv.weight"), null, 3 * h, h),
            AttnOut = new PackedLinearF32(st.ReadF32(l + "attn.out_proj.weight"), null, h, h),
            Ffn1 = new PackedLinearF32(gateUp, null, 2 * inter, h),
            Ffn2 = new PackedLinearF32(st.ReadF32(l + "mlp.fc2.weight"), null, h, inter),
            AttnLnW = aW, AttnLnB = aB, OutLnW = oW, OutLnB = oB,
            SwiGlu = true,
        };
    }

    /// <summary>Last hidden state of one input, row-major [tokens, hidden].</summary>
    public float[] Encode(EncodedInput input) => EncodeBatch([input])[0];

    /// <summary>Last hidden state per input, each row-major [tokens_i, hidden].</summary>
    public float[][] EncodeBatch(IReadOnlyList<EncodedInput> batch)
    {
        int h = Config.HiddenSize, inter = Config.IntermediateSize;
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
        bool swiGlu = _layers.Length > 0 && _layers[0].SwiGlu;
        var ffn = new float[t * inter * (swiGlu ? 2 : 1)];
        var act = swiGlu ? new float[t * inter] : ffn;
        foreach (var layer in _layers)
        {
            layer.Qkv.Forward(x, qkv, t);
            if (_ropeInvFreq is not null) ApplyRope(qkv, offsets);
            Attention(qkv, ctx, offsets);
            layer.AttnOut.Forward(ctx, tmp, t);
            AddLayerNorm(tmp, x, layer.AttnLnW, layer.AttnLnB, t);   // x = LN(x + attn)
            layer.Ffn1.Forward(x, ffn, t);
            if (layer.SwiGlu)
                Parallel.For(0, t, r =>
                {
                    var gate = ffn.AsSpan(r * 2 * inter, inter);
                    var up = ffn.AsSpan(r * 2 * inter + inter, inter);
                    var o = act.AsSpan(r * inter, inter);
                    for (int k = 0; k < inter; k++) o[k] = gate[k] / (1f + MathF.Exp(-gate[k])) * up[k];
                });
            else
                Parallel.For(0, t, r => ErfGelu.InPlace(ffn.AsSpan(r * inter, inter)));
            layer.Ffn2.Forward(act, tmp, t);
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
                _wordEmb.AsSpan(ids[i] * h, h).CopyTo(row);
                if (_posEmb is not null)
                    TensorPrimitives.Add(row, _posEmb.AsSpan((Config.PositionOffset + i) * h, h), row);
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

    /// <summary>NeoX-style rotary on the Q and K parts of each qkv row: pairs (i, i + d/2), angle pos·invFreq[i].</summary>
    private void ApplyRope(float[] qkv, int[] offsets)
    {
        int h = Config.HiddenSize, d = Config.HeadDim, half = d / 2, heads = Config.NumHeads, stride = 3 * h;
        var invFreq = _ropeInvFreq!;
        Parallel.For(0, offsets.Length - 1, s =>
        {
            Span<float> cos = stackalloc float[half], sin = stackalloc float[half];
            for (int i = 0; i < offsets[s + 1] - offsets[s]; i++)
            {
                for (int k = 0; k < half; k++)
                {
                    float angle = i * invFreq[k];
                    cos[k] = MathF.Cos(angle);
                    sin[k] = MathF.Sin(angle);
                }
                var row = qkv.AsSpan((offsets[s] + i) * stride, 2 * h); // Q then K
                for (int part = 0; part < 2 * heads; part++)
                {
                    var v = row.Slice(part * d, d);
                    for (int k = 0; k < half; k++)
                    {
                        float a = v[k], b = v[k + half];
                        v[k] = a * cos[k] - b * sin[k];
                        v[k + half] = b * cos[k] + a * sin[k];
                    }
                }
            }
        });
    }

    /// <summary>MPNet/T5 bidirectional relative-position bucket for <c>relative = key - query</c>
    /// (HF <c>MPNetEncoder.relative_position_bucket</c>, max distance 128, float32 log like torch).</summary>
    internal static int RelativePositionBucket(int relative, int numBuckets, int maxDistance = 128)
    {
        int ret = 0, n = -relative;
        numBuckets /= 2;
        if (n < 0) ret += numBuckets;
        n = Math.Abs(n);
        int maxExact = numBuckets / 2;
        if (n < maxExact) return ret + n;
        float scaled = MathF.Log(n / (float)maxExact) / (float)Math.Log(maxDistance / (double)maxExact) * (numBuckets - maxExact);
        return ret + Math.Min(maxExact + (int)scaled, numBuckets - 1);
    }

    /// <summary>Softmax(QK^T/√d [+ relative bias])·V per sequence and head; qkv rows are [Q | K | V].</summary>
    private void Attention(float[] qkv, float[] ctx, int[] offsets)
    {
        int h = Config.HiddenSize, d = Config.HeadDim, heads = Config.NumHeads, stride = 3 * h;
        float scale = 1f / MathF.Sqrt(d);
        int seqs = offsets.Length - 1, maxLen = 0;
        for (int s = 0; s < seqs; s++) maxLen = Math.Max(maxLen, offsets[s + 1] - offsets[s]);

        // Bucket per relative distance (key - query) in [-(maxLen-1), maxLen-1].
        int[]? bucketOf = null;
        if (_relBias is not null)
        {
            bucketOf = new int[2 * maxLen + 1];
            for (int rel = -maxLen; rel <= maxLen; rel++)
                bucketOf[rel + maxLen] = RelativePositionBucket(rel, Config.RelativeAttentionBuckets);
        }

        const int QueryBlock = 16;
        var work = new List<(int S, int Head, int Q0)>();
        for (int s = 0; s < seqs; s++)
            for (int hd = 0; hd < heads; hd++)
                for (int q0 = 0; q0 < offsets[s + 1] - offsets[s]; q0 += QueryBlock)
                    work.Add((s, hd, q0));

        Parallel.For(0, work.Count, () => new float[maxLen], (wi, _, scores) =>
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
                if (bucketOf is not null)
                    for (int j = 0; j < len; j++)
                        sc[j] += _relBias![bucketOf[j - i + maxLen] * heads + hd];
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

    public void Dispose()
    {
        foreach (var l in _layers) l.Dispose();
    }
}
