using System.Numerics.Tensors;
using System.Text;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Wav2Vec2;

/// <summary>
/// HF <c>Wav2Vec2ForCTC</c> (e.g. facebook/wav2vec2-base-960h, wav2vec2-large-960h-lv60-self, XLS-R CTC fine-tunes),
/// CPU F32 from safetensors, configured from <c>config.json</c> (HF <c>modeling_wav2vec2.py</c>):
/// zero-mean/unit-variance input → 7 conv1d feature-extractor layers (<c>feat_extract_norm</c> "group": GroupNorm on
/// layer 0 only; "layer": LayerNorm over channels after every conv) with exact GELU → feature projection (LayerNorm +
/// Linear) → weight-normalized grouped positional conv (kernel 128, "same" padding minus one frame) + GELU, added →
/// transformer (<c>do_stable_layer_norm</c> false: encoder LayerNorm first, post-LN layers; true: pre-LN layers,
/// encoder LayerNorm last) → <c>lm_head</c> → greedy CTC (collapse repeats, drop the pad/blank id, "|" = space).
/// Convolutions run as im2col + packed GEMM.
/// </summary>
public sealed class Wav2Vec2CtcModel : IDisposable
{
    private sealed record ConvLayer(PackedLinearF32 Conv, int InCh, int OutCh, int Kernel, int Stride, float[]? NormW, float[]? NormB);

    private sealed class Layer : IDisposable
    {
        public required PackedLinearF32 Qkv, Out, Ffn1, Ffn2;
        public required float[] LnW, LnB, FinalLnW, FinalLnB;
        public void Dispose() { Qkv.Dispose(); Out.Dispose(); Ffn1.Dispose(); Ffn2.Dispose(); }
    }

    private readonly ConvLayer[] _conv;
    private readonly bool _groupNorm, _stableLayerNorm;
    private readonly float[] _projLnW, _projLnB, _encLnW, _encLnB;
    private readonly PackedLinearF32 _proj, _lmHead;
    private readonly PackedLinearF32[] _posConv; // one per group: [groupCh, groupCh * posKernel]
    private readonly float[] _posBias;
    private readonly Layer[] _layers;
    private readonly int _hidden, _heads, _posKernel, _posGroups, _padId, _wordDelimiterId;
    private readonly float _lnEps;
    private readonly string[] _vocab;

    public int VocabSize => _vocab.Length;

    private Wav2Vec2CtcModel(ConvLayer[] conv, bool groupNorm, bool stable, float[] projLnW, float[] projLnB, PackedLinearF32 proj,
        PackedLinearF32[] posConv, float[] posBias, int posKernel, int posGroups, float[] encLnW, float[] encLnB, Layer[] layers,
        PackedLinearF32 lmHead, int hidden, int heads, float lnEps, string[] vocab, int padId, int wordDelimiterId)
    {
        (_conv, _groupNorm, _stableLayerNorm, _projLnW, _projLnB, _proj) = (conv, groupNorm, stable, projLnW, projLnB, proj);
        (_posConv, _posBias, _posKernel, _posGroups, _encLnW, _encLnB, _layers) = (posConv, posBias, posKernel, posGroups, encLnW, encLnB, layers);
        (_lmHead, _hidden, _heads, _lnEps, _vocab, _padId, _wordDelimiterId) = (lmHead, hidden, heads, lnEps, vocab, padId, wordDelimiterId);
    }

    public static Wav2Vec2CtcModel Load(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "config.json")));
        var c = doc.RootElement;
        int[] Ints(string n) => c.GetProperty(n).EnumerateArray().Select(x => x.GetInt32()).ToArray();
        string S(string n, string d) => c.TryGetProperty(n, out var v) ? v.GetString() ?? d : d;
        bool B(string n) => c.TryGetProperty(n, out var v) && v.ValueKind == JsonValueKind.True;
        if (S("feat_extract_activation", "gelu") != "gelu" || S("hidden_act", "gelu") != "gelu")
            throw new NotSupportedException("Wav2Vec2: only gelu activations are supported.");
        if (c.TryGetProperty("adapter_attn_dim", out var ad) && ad.ValueKind == JsonValueKind.Number)
            throw new NotSupportedException("Wav2Vec2: adapter layers (MMS) are not supported.");
        string norm = S("feat_extract_norm", "group");
        bool groupNorm = norm == "group";
        if (!groupNorm && norm != "layer") throw new NotSupportedException($"Wav2Vec2 feat_extract_norm '{norm}' not supported.");
        int[] dims = Ints("conv_dim"), kernels = Ints("conv_kernel"), strides = Ints("conv_stride");
        int hidden = c.GetProperty("hidden_size").GetInt32(), heads = c.GetProperty("num_attention_heads").GetInt32();
        int inter = c.GetProperty("intermediate_size").GetInt32(), layersN = c.GetProperty("num_hidden_layers").GetInt32();
        int posKernel = c.GetProperty("num_conv_pos_embeddings").GetInt32(), posGroups = c.GetProperty("num_conv_pos_embedding_groups").GetInt32();
        float eps = c.TryGetProperty("layer_norm_eps", out var e) ? e.GetSingle() : 1e-5f;
        bool convBias = B("conv_bias");

        using var st = SafetensorsLoader.OpenDirectory(dir);
        string p = st.TensorNames.Any(n => n.StartsWith("wav2vec2.", StringComparison.Ordinal)) ? "wav2vec2." : "";
        float[]? Opt(string n) => st.TensorNames.Contains(n) ? st.ReadF32(n) : null;

        var conv = new ConvLayer[dims.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            int inCh = i == 0 ? 1 : dims[i - 1];
            string cp = $"{p}feature_extractor.conv_layers.{i}.";
            bool hasNorm = !groupNorm || i == 0;
            conv[i] = new ConvLayer(
                new PackedLinearF32(st.ReadF32(cp + "conv.weight"), convBias ? st.ReadF32(cp + "conv.bias") : null, dims[i], inCh * kernels[i]),
                inCh, dims[i], kernels[i], strides[i],
                hasNorm ? st.ReadF32(cp + "layer_norm.weight") : null, hasNorm ? st.ReadF32(cp + "layer_norm.bias") : null);
        }

        // Weight-normalized positional conv (weight_norm dim=2): w[o,i,k] = g[k] * v[o,i,k] / ||v[:,:,k]||.
        string pc = $"{p}encoder.pos_conv_embed.conv.";
        float[] g = Opt(pc + "weight_g") ?? st.ReadF32(pc + "parametrizations.weight.original0");
        float[] v = Opt(pc + "weight_v") ?? st.ReadF32(pc + "parametrizations.weight.original1");
        int groupCh = hidden / posGroups;
        if (g.Length != posKernel || v.Length != hidden * groupCh * posKernel)
            throw new InvalidDataException("Wav2Vec2 pos_conv_embed: unexpected weight-norm shapes (expected weight_norm over dim 2).");
        var norms = new double[posKernel];
        for (int idx = 0; idx < v.Length; idx++) norms[idx % posKernel] += (double)v[idx] * v[idx];
        var w = new float[v.Length];
        for (int idx = 0; idx < v.Length; idx++) w[idx] = (float)(g[idx % posKernel] * v[idx] / Math.Sqrt(norms[idx % posKernel]));
        var posConv = new PackedLinearF32[posGroups];
        for (int gi = 0; gi < posGroups; gi++)
            posConv[gi] = new PackedLinearF32(w.AsSpan(gi * groupCh * groupCh * posKernel, groupCh * groupCh * posKernel).ToArray(), null, groupCh, groupCh * posKernel);

        var layers = new Layer[layersN];
        for (int i = 0; i < layersN; i++)
        {
            string lp = $"{p}encoder.layers.{i}.";
            float[] R(string n) => st.ReadF32(lp + n);
            layers[i] = new Layer
            {
                Qkv = new PackedLinearF32([.. R("attention.q_proj.weight"), .. R("attention.k_proj.weight"), .. R("attention.v_proj.weight")],
                    [.. R("attention.q_proj.bias"), .. R("attention.k_proj.bias"), .. R("attention.v_proj.bias")], 3 * hidden, hidden),
                Out = new PackedLinearF32(R("attention.out_proj.weight"), R("attention.out_proj.bias"), hidden, hidden),
                Ffn1 = new PackedLinearF32(R("feed_forward.intermediate_dense.weight"), R("feed_forward.intermediate_dense.bias"), inter, hidden),
                Ffn2 = new PackedLinearF32(R("feed_forward.output_dense.weight"), R("feed_forward.output_dense.bias"), hidden, inter),
                LnW = R("layer_norm.weight"), LnB = R("layer_norm.bias"),
                FinalLnW = R("final_layer_norm.weight"), FinalLnB = R("final_layer_norm.bias"),
            };
        }

        var vocab = ReadVocab(Path.Combine(dir, "vocab.json"));
        var lm = st.ReadF32("lm_head.weight");
        if (lm.Length != vocab.Length * hidden) throw new InvalidDataException($"lm_head has {lm.Length / hidden} rows, vocab.json {vocab.Length}.");
        int padId = c.TryGetProperty("pad_token_id", out var pad) ? pad.GetInt32() : Array.IndexOf(vocab, "<pad>");
        return new Wav2Vec2CtcModel(conv, groupNorm, B("do_stable_layer_norm"),
            st.ReadF32($"{p}feature_projection.layer_norm.weight"), st.ReadF32($"{p}feature_projection.layer_norm.bias"),
            new PackedLinearF32(st.ReadF32($"{p}feature_projection.projection.weight"), st.ReadF32($"{p}feature_projection.projection.bias"), hidden, dims[^1]),
            posConv, st.ReadF32(pc + "bias"), posKernel, posGroups,
            st.ReadF32($"{p}encoder.layer_norm.weight"), st.ReadF32($"{p}encoder.layer_norm.bias"), layers,
            new PackedLinearF32(lm, st.ReadF32("lm_head.bias"), vocab.Length, hidden), hidden, heads, eps, vocab, padId, Array.IndexOf(vocab, "|"));
    }

    private static string[] ReadVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var map = doc.RootElement.EnumerateObject().ToDictionary(x => x.Value.GetInt32(), x => x.Name);
        var vocab = new string[map.Count];
        for (int i = 0; i < vocab.Length; i++) vocab[i] = map.TryGetValue(i, out var s) ? s : throw new InvalidDataException($"vocab.json has no id {i}.");
        return vocab;
    }

    private void LayerNormRows(float[] x, int rows, int dim, float[] w, float[] b, float eps)
    {
        Parallel.For(0, rows, r =>
        {
            var row = x.AsSpan(r * dim, dim);
            float mean = TensorPrimitives.Sum(row) / dim;
            TensorPrimitives.Subtract(row, mean, row);
            float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(row) / dim + eps);
            TensorPrimitives.Multiply(row, inv, row);
            TensorPrimitives.Multiply(row, w, row);
            TensorPrimitives.Add(row, b, row);
        });
    }

    /// <summary>Valid conv1d over frame-major [t, inCh] → [tOut, outCh] via im2col (row = [c0k0..c0kK, c1k0..]).</summary>
    private static float[] Conv1d(float[] x, int t, ConvLayer l, out int tOut)
    {
        tOut = (t - l.Kernel) / l.Stride + 1;
        int kIn = l.InCh * l.Kernel, to = tOut;
        var cols = new float[(long)tOut * kIn];
        Parallel.For(0, to, o =>
        {
            var row = cols.AsSpan(o * kIn, kIn);
            int start = o * l.Stride;
            for (int c = 0; c < l.InCh; c++)
                for (int j = 0; j < l.Kernel; j++) row[c * l.Kernel + j] = x[(start + j) * l.InCh + c];
        });
        var y = new float[(long)tOut * l.OutCh];
        l.Conv.Forward(cols, y, tOut);
        return y;
    }

    /// <summary>CTC logits [frames, vocab] for a 16 kHz mono waveform (normalized here, as the feature extractor does).</summary>
    public float[] Logits(ReadOnlySpan<float> waveform16k, out int frames)
    {
        // Wav2Vec2FeatureExtractor zero_mean_unit_var_norm: (x - mean) / sqrt(var + 1e-7)
        int n = waveform16k.Length;
        double mean = 0;
        foreach (float s in waveform16k) mean += s;
        mean /= n;
        double var = 0;
        foreach (float s in waveform16k) var += (s - mean) * (s - mean);
        var /= n;
        float inv = (float)(1.0 / Math.Sqrt(var + 1e-7));
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(waveform16k[i] - mean) * inv;

        int t = n;
        for (int li = 0; li < _conv.Length; li++)
        {
            var l = _conv[li];
            x = Conv1d(x, t, l, out t);
            if (t <= 0) throw new ArgumentException("Wav2Vec2: audio too short for the feature extractor.");
            if (l.NormW is not null)
            {
                if (_groupNorm) GroupNormPerChannel(x, t, l.OutCh, l.NormW, l.NormB!);
                else LayerNormRows(x, t, l.OutCh, l.NormW, l.NormB!, _lnEps);
            }
            ErfGelu.InPlace(x);
        }

        int convDim = _conv[^1].OutCh, h = _hidden;
        LayerNormRows(x, t, convDim, _projLnW, _projLnB, _lnEps);
        var hs = new float[t * h];
        _proj.Forward(x, hs, t);

        var pos = PositionalConv(hs, t);
        TensorPrimitives.Add(hs, pos, hs);
        if (!_stableLayerNorm) LayerNormRows(hs, t, h, _encLnW, _encLnB, _lnEps);

        var tmp = new float[t * h];
        var qkv = new float[t * 3 * h];
        var ctx = new float[t * h];
        var ffn = new float[t * _layers[0].Ffn1.OutDim];
        foreach (var layer in _layers)
        {
            if (_stableLayerNorm)
            {
                Array.Copy(hs, tmp, hs.Length);
                LayerNormRows(tmp, t, h, layer.LnW, layer.LnB, _lnEps);
                Attention(layer, tmp, t, qkv, ctx);
                layer.Out.Forward(ctx, tmp, t);
                TensorPrimitives.Add(hs, tmp, hs);
                Array.Copy(hs, tmp, hs.Length);
                LayerNormRows(tmp, t, h, layer.FinalLnW, layer.FinalLnB, _lnEps);
                FeedForward(layer, tmp, t, ffn, tmp);
                TensorPrimitives.Add(hs, tmp, hs);
            }
            else
            {
                Attention(layer, hs, t, qkv, ctx);
                layer.Out.Forward(ctx, tmp, t);
                TensorPrimitives.Add(hs, tmp, hs);
                LayerNormRows(hs, t, h, layer.LnW, layer.LnB, _lnEps);
                FeedForward(layer, hs, t, ffn, tmp);
                TensorPrimitives.Add(hs, tmp, hs);
                LayerNormRows(hs, t, h, layer.FinalLnW, layer.FinalLnB, _lnEps);
            }
        }
        if (_stableLayerNorm) LayerNormRows(hs, t, h, _encLnW, _encLnB, _lnEps);

        var logits = new float[t * _vocab.Length];
        _lmHead.Forward(hs, logits, t);
        frames = t;
        return logits;
    }

    /// <summary>GroupNorm with num_groups == channels: each channel normalized over time (frame-major [t, ch]).</summary>
    private static void GroupNormPerChannel(float[] x, int t, int ch, float[] w, float[] b)
    {
        Parallel.For(0, ch, c =>
        {
            double sum = 0, sq = 0;
            for (int i = 0; i < t; i++) sum += x[i * ch + c];
            double mean = sum / t;
            for (int i = 0; i < t; i++) { double d = x[i * ch + c] - mean; sq += d * d; }
            float inv = (float)(1.0 / Math.Sqrt(sq / t + 1e-5));
            for (int i = 0; i < t; i++) x[i * ch + c] = (float)(x[i * ch + c] - mean) * inv * w[c] + b[c];
        });
    }

    /// <summary>Grouped conv1d, kernel K, padding K/2 both sides, last output dropped when K is even (Wav2Vec2SamePadLayer), then GELU.</summary>
    private float[] PositionalConv(float[] hs, int t)
    {
        int h = _hidden, gc = h / _posGroups, k = _posKernel, pad = k / 2, kIn = gc * k;
        var outp = new float[t * h];
        var cols = new float[(long)t * kIn];
        var y = new float[t * gc];
        for (int g = 0; g < _posGroups; g++)
        {
            int g0 = g * gc;
            Parallel.For(0, t, o =>
            {
                var row = cols.AsSpan(o * kIn, kIn);
                for (int c = 0; c < gc; c++)
                    for (int j = 0; j < k; j++)
                    {
                        int src = o + j - pad;
                        row[c * k + j] = src < 0 || src >= t ? 0f : hs[src * h + g0 + c];
                    }
            });
            _posConv[g].Forward(cols, y, t);
            for (int o = 0; o < t; o++) y.AsSpan(o * gc, gc).CopyTo(outp.AsSpan(o * h + g0, gc));
        }
        for (int o = 0; o < t; o++) TensorPrimitives.Add(outp.AsSpan(o * h, h), _posBias, outp.AsSpan(o * h, h));
        ErfGelu.InPlace(outp);
        return outp;
    }

    private void Attention(Layer layer, float[] x, int t, float[] qkv, float[] ctx)
    {
        int h = _hidden, dh = h / _heads, stride = 3 * h;
        layer.Qkv.Forward(x, qkv, t);
        float scale = 1f / MathF.Sqrt(dh);
        Array.Clear(ctx);
        Parallel.For(0, _heads * t, hi =>
        {
            int head = hi / t, i = hi % t, off = head * dh;
            var scores = new float[t];
            var q = qkv.AsSpan(i * stride + off, dh);
            for (int j = 0; j < t; j++) scores[j] = TensorPrimitives.Dot(q, qkv.AsSpan(j * stride + h + off, dh)) * scale;
            TensorPrimitives.Subtract(scores, TensorPrimitives.Max(scores), scores);
            TensorPrimitives.Exp(scores, scores);
            TensorPrimitives.Divide(scores, TensorPrimitives.Sum(scores), scores);
            var o = ctx.AsSpan(i * h + off, dh);
            for (int j = 0; j < t; j++) TensorPrimitives.MultiplyAdd(qkv.AsSpan(j * stride + 2 * h + off, dh), scores[j], o, o);
        });
    }

    private static void FeedForward(Layer layer, float[] x, int t, float[] ffn, float[] output)
    {
        layer.Ffn1.Forward(x, ffn, t);
        ErfGelu.InPlace(ffn.AsSpan(0, t * layer.Ffn1.OutDim));
        layer.Ffn2.Forward(ffn, output, t);
    }

    /// <summary>Greedy CTC: argmax per frame, collapse repeats, drop the pad (blank) id, word delimiter → space.</summary>
    public string DecodeGreedy(float[] logits, int frames)
    {
        var sb = new StringBuilder();
        int prev = -1, v = _vocab.Length;
        for (int f = 0; f < frames; f++)
        {
            int id = TensorPrimitives.IndexOfMax(logits.AsSpan(f * v, v));
            if (id != prev && id != _padId) sb.Append(id == _wordDelimiterId ? " " : _vocab[id]);
            prev = id;
        }
        return string.Join(' ', sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    public string Transcribe(ReadOnlySpan<float> waveform16k) => DecodeGreedy(Logits(waveform16k, out int frames), frames);

    public void Dispose()
    {
        foreach (var c in _conv) c.Conv.Dispose();
        foreach (var g in _posConv) g.Dispose();
        foreach (var l in _layers) l.Dispose();
        _proj.Dispose();
        _lmHead.Dispose();
    }
}
