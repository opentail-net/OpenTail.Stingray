using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>T5 hyper-parameters from a HF <c>config.json</c> (<c>T5ForConditionalGeneration</c>, also the backbone of
/// Chronos / Chronos-Bolt, which declare <c>model_type: t5</c>).</summary>
public sealed record T5Config(
    int DModel, int DFf, int DKv, int NumHeads, int NumLayers, int NumDecoderLayers,
    int RelativeBuckets, int RelativeMaxDistance, bool GatedGelu, float LayerNormEps,
    bool TieWordEmbeddings, int VocabSize, int DecoderStartTokenId, int EosTokenId, int PadTokenId)
{
    public static T5Config FromJson(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var r = doc.RootElement;
        int I(string n, int d) => r.TryGetProperty(n, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : d;
        string ffn = r.TryGetProperty("feed_forward_proj", out var f) ? f.GetString() ?? "relu" : "relu";
        bool gated = ffn switch
        {
            "relu" => false,
            "gated-gelu" => true,
            _ => throw new NotSupportedException($"T5 feed_forward_proj '{ffn}' not supported (relu, gated-gelu)."),
        };
        int layers = I("num_layers", 6);
        return new T5Config(
            I("d_model", 512), I("d_ff", 2048), I("d_kv", 64), I("num_heads", 8), layers, I("num_decoder_layers", layers),
            I("relative_attention_num_buckets", 32), I("relative_attention_max_distance", 128), gated,
            r.TryGetProperty("layer_norm_epsilon", out var e) ? e.GetSingle() : 1e-6f,
            !r.TryGetProperty("tie_word_embeddings", out var t) || t.ValueKind != JsonValueKind.False,
            I("vocab_size", 32128), I("decoder_start_token_id", 0), I("eos_token_id", 1), I("pad_token_id", 0));
    }
}

/// <summary>
/// T5 encoder-decoder (HF <c>modeling_t5.py</c>), CPU F32 from safetensors. T5 specifics: RMS "LayerNorm" without bias
/// or mean, no 1/√d attention scaling, one relative-position bias table per stack (block 0; encoder bidirectional,
/// decoder unidirectional) added in every layer, no position bias in cross-attention, FFN <c>wo(relu(wi x))</c> or
/// gated <c>wo(gelu_tanh(wi_0 x) ⊙ wi_1 x)</c>, and tied embeddings scale the decoder output by <c>d_model^-0.5</c>
/// before the LM head. Decoding keeps a self-attention K/V cache and precomputes cross-attention K/V once.
/// </summary>
public sealed class T5Model : IDisposable
{
    private sealed class Attn : IDisposable
    {
        public required PackedLinearF32 Q, K, V, O;
        public void Dispose() { Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose(); }
    }

    private sealed class Block : IDisposable
    {
        public required Attn Self;
        public required float[] SelfNorm;
        public Attn? Cross;
        public float[]? CrossNorm;
        public required PackedLinearF32 Wi;          // [DFf or 2*DFf (gated: wi_0 | wi_1), DModel]
        public required PackedLinearF32 Wo;
        public required float[] FfnNorm;
        public void Dispose() { Self.Dispose(); Cross?.Dispose(); Wi.Dispose(); Wo.Dispose(); }
    }

    private readonly float[] _shared;              // [vocab, DModel]
    private readonly Block[] _enc, _dec;
    private readonly float[] _encFinalNorm, _decFinalNorm;
    private readonly float[] _encRelBias, _decRelBias; // [buckets, heads]
    private readonly PackedLinearF32 _lmHead;

    public T5Config Config { get; }

    private T5Model(T5Config c, float[] shared, Block[] enc, Block[] dec, float[] encNorm, float[] decNorm,
        float[] encBias, float[] decBias, PackedLinearF32 lmHead)
    {
        Config = c;
        (_shared, _enc, _dec, _encFinalNorm, _decFinalNorm, _encRelBias, _decRelBias, _lmHead) =
            (shared, enc, dec, encNorm, decNorm, encBias, decBias, lmHead);
    }

    public static T5Model Load(string dir)
    {
        using var st = SafetensorsLoader.OpenDirectory(dir);
        return Load(T5Config.FromJson(Path.Combine(dir, "config.json")), st);
    }

    /// <summary>Loads the encoder and decoder stacks. <paramref name="loadLmHead"/> false skips the vocab projection (models
    /// such as Chronos-Bolt that put their own head on the decoder output).</summary>
    public static T5Model Load(T5Config c, SafetensorsLoader st, string prefix = "", bool loadLmHead = true)
    {
        int inner = c.NumHeads * c.DKv;
        Attn ReadAttn(string p) => new()
        {
            Q = new PackedLinearF32(st.ReadF32(p + ".q.weight"), null, inner, c.DModel),
            K = new PackedLinearF32(st.ReadF32(p + ".k.weight"), null, inner, c.DModel),
            V = new PackedLinearF32(st.ReadF32(p + ".v.weight"), null, inner, c.DModel),
            O = new PackedLinearF32(st.ReadF32(p + ".o.weight"), null, c.DModel, inner),
        };
        Block ReadBlock(string stack, int i, bool decoder)
        {
            string b = $"{prefix}{stack}.block.{i}.layer.";
            string ffn = b + (decoder ? "2" : "1") + ".DenseReluDense.";
            float[] wi = c.GatedGelu
                ? [.. st.ReadF32(ffn + "wi_0.weight"), .. st.ReadF32(ffn + "wi_1.weight")]
                : st.ReadF32(ffn + "wi.weight");
            return new Block
            {
                Self = ReadAttn(b + "0.SelfAttention"),
                SelfNorm = st.ReadF32(b + "0.layer_norm.weight"),
                Cross = decoder ? ReadAttn(b + "1.EncDecAttention") : null,
                CrossNorm = decoder ? st.ReadF32(b + "1.layer_norm.weight") : null,
                Wi = new PackedLinearF32(wi, null, (c.GatedGelu ? 2 : 1) * c.DFf, c.DModel),
                Wo = new PackedLinearF32(st.ReadF32(ffn + "wo.weight"), null, c.DModel, c.DFf),
                FfnNorm = st.ReadF32(b + (decoder ? "2" : "1") + ".layer_norm.weight"),
            };
        }

        float[] shared = st.ReadF32(prefix + "shared.weight");
        var enc = Enumerable.Range(0, c.NumLayers).Select(i => ReadBlock("encoder", i, false)).ToArray();
        var dec = Enumerable.Range(0, c.NumDecoderLayers).Select(i => ReadBlock("decoder", i, true)).ToArray();
        string lmName = prefix + "lm_head.weight";
        float[] lm = !loadLmHead ? new float[c.DModel] : c.TieWordEmbeddings ? shared : st.ReadF32(lmName);
        return new T5Model(c, shared, enc, dec,
            st.ReadF32(prefix + "encoder.final_layer_norm.weight"), st.ReadF32(prefix + "decoder.final_layer_norm.weight"),
            st.ReadF32(prefix + "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            st.ReadF32(prefix + "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            new PackedLinearF32((float[])lm.Clone(), null, loadLmHead ? c.VocabSize : 1, c.DModel));
    }

    private void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> y)
    {
        float ms = TensorPrimitives.SumOfSquares(x) / x.Length;
        float inv = 1f / MathF.Sqrt(ms + Config.LayerNormEps);
        TensorPrimitives.Multiply(x, inv, y);
        TensorPrimitives.Multiply(y, w, y);
    }

    private void RmsNormRows(float[] x, float[] w, float[] y, int rows)
    {
        int d = Config.DModel;
        Parallel.For(0, rows, r => RmsNorm(x.AsSpan(r * d, d), w, y.AsSpan(r * d, d)));
    }

    private float[] Embed(ReadOnlySpan<int> ids)
    {
        int d = Config.DModel;
        var x = new float[ids.Length * d];
        for (int i = 0; i < ids.Length; i++) _shared.AsSpan(ids[i] * d, d).CopyTo(x.AsSpan(i * d, d));
        return x;
    }

    /// <summary>Per-head bias [heads, q, k] for query positions [q0, q0+qn) against keys [0, kn).</summary>
    private float[] RelativeBias(float[] table, bool bidirectional, int q0, int qn, int kn)
    {
        int heads = Config.NumHeads;
        var bias = new float[heads * qn * kn];
        for (int i = 0; i < qn; i++)
            for (int j = 0; j < kn; j++)
            {
                int bucket = TransformerEncoder.RelativePositionBucket(j - (q0 + i), Config.RelativeBuckets, Config.RelativeMaxDistance, bidirectional);
                for (int h = 0; h < heads; h++) bias[(h * qn + i) * kn + j] = table[bucket * heads + h];
            }
        return bias;
    }

    /// <summary>ctx[qn, inner] = softmax(q·kᵀ (+ bias) (+ causal)) · v, no scaling. q: [qn, inner]; k, v: [kn, inner].</summary>
    private void Attention(float[] q, int qn, float[] k, float[] v, int kn, float[]? bias, int causalOffset, float[] ctx)
    {
        int heads = Config.NumHeads, dk = Config.DKv, inner = heads * dk;
        Array.Clear(ctx, 0, qn * inner);
        Parallel.For(0, heads * qn, hi =>
        {
            int h = hi / qn, i = hi % qn;
            int visible = causalOffset < 0 ? kn : Math.Min(kn, causalOffset + i + 1);
            var scores = new float[visible];
            var qi = q.AsSpan(i * inner + h * dk, dk);
            for (int j = 0; j < visible; j++)
                scores[j] = TensorPrimitives.Dot(qi, k.AsSpan(j * inner + h * dk, dk)) + (bias is null ? 0f : bias[(h * qn + i) * kn + j]);
            TensorPrimitives.SoftMax(scores, scores);
            var o = ctx.AsSpan(i * inner + h * dk, dk);
            for (int j = 0; j < visible; j++)
                TensorPrimitives.MultiplyAdd(v.AsSpan(j * inner + h * dk, dk), scores[j], o, o);
        });
    }

    private void Ffn(Block b, float[] x, int rows)
    {
        int d = Config.DModel, ff = Config.DFf;
        var n = new float[rows * d];
        RmsNormRows(x, b.FfnNorm, n, rows);
        var hid = new float[rows * b.Wi.OutDim];
        b.Wi.Forward(n, hid, rows);
        var act = Config.GatedGelu ? new float[rows * ff] : hid;
        Parallel.For(0, rows, r =>
        {
            if (Config.GatedGelu)
            {
                var g = hid.AsSpan(r * 2 * ff, ff);
                var u = hid.AsSpan(r * 2 * ff + ff, ff);
                var a = act.AsSpan(r * ff, ff);
                for (int i = 0; i < ff; i++) a[i] = GeluTanh(g[i]) * u[i];
            }
            else
            {
                var a = act.AsSpan(r * ff, ff);
                TensorPrimitives.Max(a, 0f, a);
            }
        });
        var o = new float[rows * d];
        b.Wo.Forward(act, o, rows);
        TensorPrimitives.Add(x, o, x);
    }

    /// <summary>HF <c>gelu_new</c> (the tanh approximation), used by gated-gelu T5 v1.1 / flan-t5.</summary>
    private static float GeluTanh(float x) => 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));

    /// <summary>Encoder over token ids: <c>encoder.final_layer_norm</c>'d hidden states [t, DModel].</summary>
    public float[] Encode(ReadOnlySpan<int> ids) => EncodeEmbeddings(Embed(ids), ids.Length);

    /// <summary>Encoder over precomputed input embeddings (e.g. Chronos-Bolt's patch embeddings) [t, DModel].</summary>
    public float[] EncodeEmbeddings(float[] embeddings, int t)
    {
        int d = Config.DModel, inner = Config.NumHeads * Config.DKv;
        var x = (float[])embeddings.Clone();
        var bias = RelativeBias(_encRelBias, bidirectional: true, 0, t, t);
        var n = new float[t * d];
        float[] q = new float[t * inner], k = new float[t * inner], v = new float[t * inner], ctx = new float[t * inner], o = new float[t * d];
        foreach (var b in _enc)
        {
            RmsNormRows(x, b.SelfNorm, n, t);
            b.Self.Q.Forward(n, q, t);
            b.Self.K.Forward(n, k, t);
            b.Self.V.Forward(n, v, t);
            Attention(q, t, k, v, t, bias, -1, ctx);
            b.Self.O.Forward(ctx, o, t);
            TensorPrimitives.Add(x, o, x);
            Ffn(b, x, t);
        }
        var y = new float[t * d];
        RmsNormRows(x, _encFinalNorm, y, t);
        return y;
    }

    /// <summary>Incremental decoder over one encoder output: self-attention K/V cache plus cross-attention K/V computed once.</summary>
    public sealed class Decoder
    {
        private readonly T5Model _m;
        private readonly float[][] _crossK, _crossV, _selfK, _selfV;
        private readonly int _encLen, _capacity;
        public int Length { get; private set; }

        internal Decoder(T5Model m, float[] encoderHidden, int encLen, int capacity)
        {
            (_m, _encLen, _capacity) = (m, encLen, capacity);
            int inner = m.Config.NumHeads * m.Config.DKv, layers = m._dec.Length;
            _crossK = new float[layers][];
            _crossV = new float[layers][];
            _selfK = new float[layers][];
            _selfV = new float[layers][];
            for (int l = 0; l < layers; l++)
            {
                var cross = m._dec[l].Cross!;
                _crossK[l] = new float[encLen * inner];
                _crossV[l] = new float[encLen * inner];
                cross.K.Forward(encoderHidden, _crossK[l], encLen);
                cross.V.Forward(encoderHidden, _crossV[l], encLen);
                _selfK[l] = new float[capacity * inner];
                _selfV[l] = new float[capacity * inner];
            }
        }

        /// <summary>Feeds <paramref name="ids"/> (appended after the cached positions) and returns their final-normed
        /// decoder hidden states [n, DModel].</summary>
        public float[] Step(ReadOnlySpan<int> ids) => StepEmbeddings(_m.Embed(ids), ids.Length);

        public float[] StepEmbeddings(float[] embeddings, int n)
        {
            if (Length + n > _capacity) throw new InvalidOperationException($"T5 decoder capacity {_capacity} exceeded.");
            var c = _m.Config;
            int d = c.DModel, inner = c.NumHeads * c.DKv, pos = Length, total = pos + n;
            var x = (float[])embeddings.Clone();
            var selfBias = _m.RelativeBias(_m._decRelBias, bidirectional: false, pos, n, total);
            var nx = new float[n * d];
            float[] q = new float[n * inner], ctx = new float[n * inner], o = new float[n * d];
            for (int l = 0; l < _m._dec.Length; l++)
            {
                var b = _m._dec[l];
                _m.RmsNormRows(x, b.SelfNorm, nx, n);
                b.Self.Q.Forward(nx, q, n);
                b.Self.K.Forward(nx, _selfK[l].AsSpan(pos * inner, n * inner), n);
                b.Self.V.Forward(nx, _selfV[l].AsSpan(pos * inner, n * inner), n);
                _m.Attention(q, n, _selfK[l], _selfV[l], total, selfBias, pos, ctx);
                b.Self.O.Forward(ctx, o, n);
                TensorPrimitives.Add(x, o, x);

                _m.RmsNormRows(x, b.CrossNorm!, nx, n);
                b.Cross!.Q.Forward(nx, q, n);
                _m.Attention(q, n, _crossK[l], _crossV[l], _encLen, null, -1, ctx);
                b.Cross.O.Forward(ctx, o, n);
                TensorPrimitives.Add(x, o, x);

                _m.Ffn(b, x, n);
            }
            Length = total;
            var y = new float[n * d];
            _m.RmsNormRows(x, _m._decFinalNorm, y, n);
            return y;
        }
    }

    public Decoder StartDecoder(float[] encoderHidden, int encLen, int capacity = 512) => new(this, encoderHidden, encLen, capacity);

    /// <summary>LM logits [n, vocab] for final-normed decoder hidden states (tied embeddings: scaled by d_model^-0.5 first).</summary>
    public float[] Logits(float[] decoderHidden, int n)
    {
        var h = decoderHidden;
        if (Config.TieWordEmbeddings)
        {
            h = new float[decoderHidden.Length];
            TensorPrimitives.Multiply(decoderHidden, 1f / MathF.Sqrt(Config.DModel), h);
        }
        var logits = new float[n * Config.VocabSize];
        _lmHead.Forward(h, logits, n);
        return logits;
    }

    /// <summary>Greedy decoding from <c>decoder_start_token_id</c> until EOS or <paramref name="maxNewTokens"/>.</summary>
    public List<int> GenerateGreedy(ReadOnlySpan<int> inputIds, int maxNewTokens = 64)
    {
        var enc = Encode(inputIds);
        var dec = StartDecoder(enc, inputIds.Length, maxNewTokens + 1);
        var output = new List<int>();
        int token = Config.DecoderStartTokenId;
        for (int s = 0; s < maxNewTokens; s++)
        {
            var logits = Logits(dec.Step([token]), 1);
            token = TensorPrimitives.IndexOfMax(logits);
            if (token == Config.EosTokenId) break;
            output.Add(token);
        }
        return output;
    }

    public void Dispose()
    {
        foreach (var b in _enc) b.Dispose();
        foreach (var b in _dec) b.Dispose();
        _lmHead.Dispose();
    }
}
