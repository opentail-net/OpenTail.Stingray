using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Chronos-2 (amazon/chronos-2, <c>Chronos2Model</c>), ported from the vendored reference
/// <c>examples/chronos-forecasting/src/chronos/chronos2/{model,layers}.py</c>. Encoder-only: per series, instance norm
/// (+ arcsinh), NaN-left-padded patches with [time encoding | value | observed mask] features (time index / 8192:
/// context −C..−1, future 0..H−1) → <c>input_patch_embedding</c>, [REG], then one "future" patch per 16 forecast steps
/// (time encoding, no covariates). Each of the 12 blocks: RMS-normed time self-attention with NeoX RoPE (positions =
/// token index, no 1/√d scaling, key mask), group self-attention across the series of a group at each token position
/// (no RoPE; group ∧ observed mask), ReLU MLP. The future tokens' final-normed states go through
/// <c>output_patch_embedding</c> to 21 quantiles × 16 steps. Masked scores get finfo(float32).min added, so a query
/// whose keys are all masked attends uniformly over all keys, exactly as the reference does.
/// Not implemented: future covariates, and horizons beyond max_output_patches × 16 (the reference's quantile unrolling).
/// </summary>
public sealed class Chronos2Model : IDisposable
{
    private sealed class Mha : IDisposable
    {
        public required PackedLinearF32 Qkv, O;
        public required float[] Norm;
        public void Dispose() { Qkv.Dispose(); O.Dispose(); }
    }

    private sealed class Block : IDisposable
    {
        public required Mha Time, Group;
        public required float[] FfnNorm;
        public required PackedLinearF32 Wi, Wo;
        public void Dispose() { Time.Dispose(); Group.Dispose(); Wi.Dispose(); Wo.Dispose(); }
    }

    private readonly ChronosResidualBlock _inPatch, _outPatch;
    private readonly Block[] _blocks;
    private readonly float[] _finalNorm, _regEmbedding;
    private readonly float[] _invFreq;
    private readonly int _d, _heads, _dk, _patch;
    private readonly float _eps, _timeScale;
    private readonly bool _arcsinh;

    public int ContextLength { get; }
    public int MaxOutputPatches { get; }
    public int PatchSize => _patch;
    public float[] Quantiles { get; }

    private Chronos2Model(ChronosResidualBlock inPatch, ChronosResidualBlock outPatch, Block[] blocks, float[] finalNorm, float[] reg,
        int d, int heads, int dk, int patch, float eps, float timeScale, bool arcsinh, float ropeTheta, int ctx, int maxOut, float[] quantiles)
    {
        (_inPatch, _outPatch, _blocks, _finalNorm, _regEmbedding) = (inPatch, outPatch, blocks, finalNorm, reg);
        (_d, _heads, _dk, _patch, _eps, _timeScale, _arcsinh) = (d, heads, dk, patch, eps, timeScale, arcsinh);
        (ContextLength, MaxOutputPatches, Quantiles) = (ctx, maxOut, quantiles);
        _invFreq = new float[dk / 2];
        for (int i = 0; i < _invFreq.Length; i++) _invFreq[i] = 1f / MathF.Pow(ropeTheta, 2f * i / dk);
    }

    public static Chronos2Model Load(string dir)
    {
        string configPath = Path.Combine(dir, "config.json");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var r = doc.RootElement;
        if (!r.TryGetProperty("architectures", out var arch) || arch[0].GetString() != "Chronos2Model")
            throw new NotSupportedException($"{configPath}: not a Chronos2Model checkpoint.");
        var cc = r.GetProperty("chronos_config");
        int patch = cc.GetProperty("input_patch_size").GetInt32();
        if (cc.GetProperty("input_patch_stride").GetInt32() != patch || cc.GetProperty("output_patch_size").GetInt32() != patch)
            throw new NotSupportedException("Chronos-2: patch stride and output patch size must equal the input patch size.");
        if ((r.TryGetProperty("dense_act_fn", out var act) ? act.GetString() : "relu") != "relu" ||
            (r.TryGetProperty("is_gated_act", out var g) && g.ValueKind == JsonValueKind.True))
            throw new NotSupportedException("Chronos-2: only the ReLU, non-gated MLP is supported.");
        int ctx = cc.GetProperty("context_length").GetInt32();
        float timeScale = cc.TryGetProperty("time_encoding_scale", out var ts) ? ts.GetSingle() : ctx;
        bool arcsinh = cc.TryGetProperty("use_arcsinh", out var ua) && ua.ValueKind == JsonValueKind.True;
        bool reg = cc.TryGetProperty("use_reg_token", out var ur) && ur.ValueKind == JsonValueKind.True;
        var quantiles = cc.GetProperty("quantiles").EnumerateArray().Select(x => x.GetSingle()).ToArray();
        int d = r.GetProperty("d_model").GetInt32(), dff = r.GetProperty("d_ff").GetInt32(), dk = r.GetProperty("d_kv").GetInt32();
        int heads = r.GetProperty("num_heads").GetInt32(), layers = r.GetProperty("num_layers").GetInt32();
        float eps = r.TryGetProperty("layer_norm_epsilon", out var e) ? e.GetSingle() : 1e-6f;
        float theta = r.TryGetProperty("rope_theta", out var rt) ? rt.GetSingle() : 10000f;

        using var st = SafetensorsLoader.OpenDirectory(dir);
        int inner = heads * dk;
        Mha ReadMha(string p) => new()
        {
            Qkv = new PackedLinearF32([.. st.ReadF32(p + ".self_attention.q.weight"), .. st.ReadF32(p + ".self_attention.k.weight"),
                .. st.ReadF32(p + ".self_attention.v.weight")], null, 3 * inner, d),
            O = new PackedLinearF32(st.ReadF32(p + ".self_attention.o.weight"), null, d, inner),
            Norm = st.ReadF32(p + ".layer_norm.weight"),
        };
        var blocks = new Block[layers];
        for (int i = 0; i < layers; i++)
        {
            string b = $"encoder.block.{i}.layer.";
            blocks[i] = new Block
            {
                Time = ReadMha(b + "0"),
                Group = ReadMha(b + "1"),
                FfnNorm = st.ReadF32(b + "2.layer_norm.weight"),
                Wi = new PackedLinearF32(st.ReadF32(b + "2.mlp.wi.weight"), null, dff, d),
                Wo = new PackedLinearF32(st.ReadF32(b + "2.mlp.wo.weight"), null, d, dff),
            };
        }
        float[] shared = st.ReadF32("shared.weight");
        return new Chronos2Model(
            ChronosResidualBlock.Load(st, "input_patch_embedding", 3 * patch, dff, d),
            ChronosResidualBlock.Load(st, "output_patch_embedding", d, dff, quantiles.Length * patch),
            blocks, st.ReadF32("encoder.final_layer_norm.weight"), reg ? shared.AsSpan(d, d).ToArray() : [],
            d, heads, dk, patch, eps, timeScale, arcsinh, theta, ctx, cc.GetProperty("max_output_patches").GetInt32(), quantiles);
    }

    private void RmsNorm(float[] x, float[] w, float[] y, int rows) => RowKernels.RmsNormRows(x, y, rows, _d, w, _eps);

    /// <summary>Unscaled softmax attention for one query over <paramref name="keys"/> key rows (masked → finfo.min; all masked → uniform).</summary>
    private static void AttendOne(ReadOnlySpan<float> q, Func<int, ReadOnlySpan<float>> key, Func<int, ReadOnlySpan<float>> value, int keys,
        Func<int, bool> visible, Span<float> output, float[] scores)
    {
        bool any = false;
        for (int j = 0; j < keys; j++)
            if (visible(j)) { scores[j] = TensorPrimitives.Dot(q, key(j)); any = true; }
            else scores[j] = float.NegativeInfinity;
        var s = scores.AsSpan(0, keys);
        if (!any) s.Fill(1f / keys);
        else
        {
            RowKernels.SoftmaxInPlace(s);
        }
        output.Clear();
        for (int j = 0; j < keys; j++) if (s[j] != 0f) TensorPrimitives.MultiplyAdd(value(j), s[j], output, output);
    }

    /// <summary>
    /// Quantile forecasts for a batch of series: result[i] is [Quantiles.Length, predictionLength] (quantile-major).
    /// <paramref name="groupIds"/> (default: each series alone) makes series with the same id attend to each other
    /// (multivariate forecasting). NaN = missing; series are left-padded with NaN to a common length.
    /// </summary>
    public float[][] Predict(IReadOnlyList<float[]> contexts, int predictionLength, int[]? groupIds = null)
    {
        int b = contexts.Count, p = _patch, d = _d, inner = _heads * _dk, nq = Quantiles.Length;
        int outPatches = (predictionLength + p - 1) / p;
        if (outPatches > MaxOutputPatches)
            throw new NotSupportedException($"Chronos-2: prediction length {predictionLength} needs {outPatches} output patches > {MaxOutputPatches} (long-horizon unrolling not implemented).");
        groupIds ??= Enumerable.Range(0, b).ToArray();

        int len = Math.Min(contexts.Max(c => c.Length), ContextLength);
        int ctxPatches = (len + p - 1) / p, padded = ctxPatches * p;
        int regTokens = _regEmbedding.Length > 0 ? 1 : 0, seq = ctxPatches + regTokens + outPatches;
        var norms = new ChronosInstanceNorm[b];
        var feats = new float[b * ctxPatches * 3 * p];
        var mask = new bool[b * seq];
        for (int s = 0; s < b; s++)
        {
            var c = contexts[s].AsSpan(Math.Max(0, contexts[s].Length - ContextLength));
            norms[s] = ChronosInstanceNorm.Fit(c, _arcsinh);
            int lead = padded - c.Length; // NaN left padding
            for (int n = 0; n < ctxPatches; n++)
            {
                var f = feats.AsSpan((s * ctxPatches + n) * 3 * p, 3 * p);
                bool observed = false;
                for (int j = 0; j < p; j++)
                {
                    int idx = n * p + j - lead;
                    float v = idx < 0 ? float.NaN : c[idx];
                    bool ok = !float.IsNaN(v);
                    f[j] = (-padded + n * p + j) / _timeScale;
                    f[p + j] = ok ? norms[s].Apply(v) : 0f;
                    f[2 * p + j] = ok ? 1f : 0f;
                    observed |= ok;
                }
                mask[s * seq + n] = observed;
            }
            for (int t = ctxPatches; t < seq; t++) mask[s * seq + t] = true;
        }

        var futFeats = new float[outPatches * 3 * p];
        for (int n = 0; n < outPatches; n++)
            for (int j = 0; j < p; j++) futFeats[n * 3 * p + j] = (n * p + j) / _timeScale;
        var ctxEmb = _inPatch.Forward(feats, b * ctxPatches);
        var futEmb = _inPatch.Forward(futFeats, outPatches);

        var x = new float[b * seq * d];
        for (int s = 0; s < b; s++)
        {
            ctxEmb.AsSpan(s * ctxPatches * d, ctxPatches * d).CopyTo(x.AsSpan(s * seq * d));
            if (regTokens > 0) _regEmbedding.CopyTo(x.AsSpan((s * seq + ctxPatches) * d, d));
            futEmb.CopyTo(x.AsSpan((s * seq + ctxPatches + regTokens) * d));
        }

        // RoPE tables over token positions 0..seq-1 (NeoX rotate_half layout).
        int half = _dk / 2;
        var cos = new float[seq * half];
        var sin = new float[seq * half];
        for (int t = 0; t < seq; t++)
            for (int i = 0; i < half; i++) { cos[t * half + i] = MathF.Cos(t * _invFreq[i]); sin[t * half + i] = MathF.Sin(t * _invFreq[i]); }

        int rows = b * seq;
        var nx = new float[rows * d];
        var qkv = new float[rows * 3 * inner];
        var ctxOut = new float[rows * inner];
        var o = new float[rows * d];
        var hid = new float[rows * _blocks[0].Wi.OutDim];
        foreach (var blk in _blocks)
        {
            // Time self-attention (per series, RoPE, key mask).
            RmsNorm(x, blk.Time.Norm, nx, rows);
            blk.Time.Qkv.Forward(nx, qkv, rows);
            Parallel.For(0, rows, r =>
            {
                for (int h = 0; h < _heads; h++)
                    for (int part = 0; part < 2; part++)
                    {
                        var v = qkv.AsSpan(r * 3 * inner + part * inner + h * _dk, _dk);
                        int t = r % seq;
                        for (int i = 0; i < half; i++)
                        {
                            float a = v[i], bb = v[i + half], cs = cos[t * half + i], sn = sin[t * half + i];
                            v[i] = a * cs - bb * sn;
                            v[i + half] = bb * cs + a * sn;
                        }
                    }
            });
            Parallel.For(0, rows * _heads, idx =>
            {
                int r = idx / _heads, h = idx % _heads, s = r / seq, i = r % seq, off = h * _dk;
                var scores = new float[seq];
                AttendOne(qkv.AsSpan(r * 3 * inner + off, _dk),
                    j => qkv.AsSpan(((s * seq + j) * 3 + 1) * inner + off, _dk),
                    j => qkv.AsSpan(((s * seq + j) * 3 + 2) * inner + off, _dk),
                    seq, j => mask[s * seq + j], ctxOut.AsSpan(r * inner + off, _dk), scores);
            });
            blk.Time.O.Forward(ctxOut, o, rows);
            TensorPrimitives.Add(x, o, x);

            // Group self-attention (across the series of a group, at each token position).
            RmsNorm(x, blk.Group.Norm, nx, rows);
            blk.Group.Qkv.Forward(nx, qkv, rows);
            Parallel.For(0, rows * _heads, idx =>
            {
                int r = idx / _heads, h = idx % _heads, s = r / seq, t = r % seq, off = h * _dk;
                var scores = new float[b];
                AttendOne(qkv.AsSpan(r * 3 * inner + off, _dk),
                    k => qkv.AsSpan(((k * seq + t) * 3 + 1) * inner + off, _dk),
                    k => qkv.AsSpan(((k * seq + t) * 3 + 2) * inner + off, _dk),
                    b, k => groupIds[k] == groupIds[s] && mask[k * seq + t], ctxOut.AsSpan(r * inner + off, _dk), scores);
            });
            blk.Group.O.Forward(ctxOut, o, rows);
            TensorPrimitives.Add(x, o, x);

            // Feed-forward.
            RmsNorm(x, blk.FfnNorm, nx, rows);
            blk.Wi.Forward(nx, hid, rows);
            TensorPrimitives.Max(hid, 0f, hid);
            blk.Wo.Forward(hid, o, rows);
            TensorPrimitives.Add(x, o, x);
        }
        RmsNorm(x, _finalNorm, nx, rows);

        var result = new float[b][];
        var fut = new float[outPatches * d];
        for (int s = 0; s < b; s++)
        {
            nx.AsSpan((s * seq + ctxPatches + regTokens) * d, outPatches * d).CopyTo(fut);
            var qp = _outPatch.Forward(fut, outPatches); // [n, (q p)]
            var outp = new float[nq * predictionLength];
            for (int k = 0; k < nq; k++)
            {
                for (int n = 0; n < outPatches; n++)
                    for (int j = 0; j < p; j++)
                    {
                        int step = n * p + j;
                        if (step < predictionLength) outp[k * predictionLength + step] = qp[n * nq * p + k * p + j];
                    }
                norms[s].InverseInPlace(outp.AsSpan(k * predictionLength, predictionLength));
            }
            result[s] = outp;
        }
        return result;
    }

    public void Dispose()
    {
        _inPatch.Dispose();
        _outPatch.Dispose();
        foreach (var blk in _blocks) blk.Dispose();
    }
}
