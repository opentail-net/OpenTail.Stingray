using System.Numerics.Tensors;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// DeepSeek-OCR2 vision tower, ported from llama.cpp <c>tools/mtmd/models/deepseekocr2.cpp</c> +
/// <c>deepseekocr.cpp</c> (<c>build_sam</c>), 2026-09-27:
/// <list type="number">
/// <item>SAM ViT-B: 16-px patch conv, learned 64x64 position table, 12 blocks (LayerNorm eps 1e-6,
///   14x14 windowed attention except the global blocks 2/5/8/11, decomposed relative-position bias
///   added to the scores, GELU MLP), then the neck (1x1 conv, LN, 3x3 conv, LN) and two stride-2
///   3x3 convs (<c>net_2</c>, <c>net_3</c>) -> 16x16 x 896.</item>
/// <item>A Qwen2-style encoder over [256 SAM tokens ; 256 learned queries]: RMSNorm, GQA (14/2 heads),
///   NeoX RoPE theta 1e6 over positions 0..511, SwiGLU; image tokens attend only to image tokens,
///   query tokens to all image tokens plus earlier queries. Only the query outputs are kept.</item>
/// <item><c>mm.model.fc</c> (896 -> 1280) and the <c>view_seperator</c> row appended: 257 tokens.</item>
/// </list>
/// Handles the 1024-px global view; 768-px tiles (144 queries) are not implemented yet.
/// </summary>
public sealed unsafe class DeepSeekOcr2VisionEncoder
{
    private const int SamEmbd = 768, SamHeads = 12, SamLayers = 12, SamWindow = 14, SamPatch = 16;
    private const float SamEps = 1e-6f;
    private static readonly int[] GlobalLayers = [2, 5, 8, 11];

    private readonly GgufModel _g;
    private readonly int _qEmbd, _qHeads, _qKvHeads, _qLayers, _qFfn, _projDim;
    private readonly float _qEps;

    private readonly float[] _patchW, _patchB, _posEmbd;
    private readonly SamBlock[] _sam;
    private readonly float[] _neck0, _neck1W, _neck1B, _neck2, _neck3W, _neck3B, _net2, _net3;
    private readonly QBlock[] _q;
    private readonly float[] _query1024, _postLn, _fcB, _viewSep;
    private readonly VisionTensorRef _fcW;

    private sealed class SamBlock
    {
        public required float[] Ln1W, Ln1B, Ln2W, Ln2B, QkvB, OutB, Lin1B, Lin2B, RelH, RelW;
        public required VisionTensorRef QkvW, OutW, Lin1W, Lin2W;
    }

    private sealed class QBlock
    {
        public required float[] Ln1, Ln2, QB, KB, VB;
        public required VisionTensorRef QW, KW, VW, OW, GateW, UpW, DownW;
    }

    public int EmbeddingDim => _projDim;

    public DeepSeekOcr2VisionEncoder(GgufModel gguf)
    {
        _g = gguf;
        _qEmbd = GetInt("clip.vision.embedding_length", 896);
        _qHeads = GetInt("clip.vision.attention.head_count", 14);
        _qKvHeads = GetInt("clip.vision.attention.head_count_kv", 2);
        _qLayers = GetInt("clip.vision.block_count", 24);
        _qFfn = GetInt("clip.vision.feed_forward_length", 4864);
        _projDim = GetInt("clip.vision.projection_dim", 1280);
        _qEps = gguf.Metadata.TryGetValue("clip.vision.attention.layer_norm_epsilon", out var e) ? Convert.ToSingle(e) : 1e-6f;

        _patchW = F32("v.sam.patch_embd.weight");
        _patchB = F32("v.sam.patch_embd.bias");
        _posEmbd = F32("v.sam.pos_embd.weight");
        _sam = new SamBlock[SamLayers];
        for (int l = 0; l < SamLayers; l++)
        {
            string p = $"v.sam.blk.{l}.";
            _sam[l] = new SamBlock
            {
                Ln1W = F32(p + "pre_ln.weight"), Ln1B = F32(p + "pre_ln.bias"),
                Ln2W = F32(p + "post_ln.weight"), Ln2B = F32(p + "post_ln.bias"),
                QkvW = W(p + "attn.qkv.weight"), QkvB = F32(p + "attn.qkv.bias"),
                OutW = W(p + "attn.out.weight"), OutB = F32(p + "attn.out.bias"),
                Lin1W = W(p + "mlp.lin1.weight"), Lin1B = F32(p + "mlp.lin1.bias"),
                Lin2W = W(p + "mlp.lin2.weight"), Lin2B = F32(p + "mlp.lin2.bias"),
                RelH = F32(p + "attn.pos_h.weight"), RelW = F32(p + "attn.pos_w.weight"),
            };
        }
        _neck0 = F32("v.sam.neck.0.weight");
        _neck1W = F32("v.sam.neck.1.weight"); _neck1B = F32("v.sam.neck.1.bias");
        _neck2 = F32("v.sam.neck.2.weight");
        _neck3W = F32("v.sam.neck.3.weight"); _neck3B = F32("v.sam.neck.3.bias");
        _net2 = F32("v.sam.net_2.weight");
        _net3 = F32("v.sam.net_3.weight");

        _q = new QBlock[_qLayers];
        for (int l = 0; l < _qLayers; l++)
        {
            string p = $"v.blk.{l}.";
            _q[l] = new QBlock
            {
                Ln1 = F32(p + "ln1.weight"), Ln2 = F32(p + "ln2.weight"),
                QW = W(p + "attn_q.weight"), QB = F32(p + "attn_q.bias"),
                KW = W(p + "attn_k.weight"), KB = F32(p + "attn_k.bias"),
                VW = W(p + "attn_v.weight"), VB = F32(p + "attn_v.bias"),
                OW = W(p + "attn_out.weight"),
                GateW = W(p + "ffn_gate.weight"), UpW = W(p + "ffn_up.weight"), DownW = W(p + "ffn_down.weight"),
            };
        }
        _query1024 = F32("v.resample_query_1024.weight");
        _postLn = F32("v.post_ln.weight");
        _fcW = W("mm.model.fc.weight");
        _fcB = F32("mm.model.fc.bias");
        _viewSep = F32("v.view_seperator");
    }

    private int GetInt(string key, int def) =>
        _g.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : def;

    private float[] F32(string name) =>
        VisionOps.LoadTensorF32(_g, name) ?? throw new InvalidOperationException($"Missing tensor: {name}");

    private VisionTensorRef W(string name)
    {
        var t = VisionOps.GetTensor(_g, name);
        if (!t.IsValid) throw new InvalidOperationException($"Missing tensor: {name}");
        return t;
    }

    /// <summary>Encodes a 1024x1024 CHW global view; returns 257 x <see cref="EmbeddingDim"/> soft tokens.</summary>
    public float[] EncodeGlobalView(float[] chw, out int tokenCount)
    {
        float[] sam = Sam(chw, DeepSeekOcr2ImagePreprocessor.BaseSize, out int side);   // side x side x 896
        int nImg = side * side;
        const int nQuery = 256;
        if (nImg != nQuery) throw new InvalidOperationException($"DeepSeek-OCR2 global view expects 256 SAM tokens, got {nImg}.");
        float[] queries = Qwen2Encoder(sam, nImg, _query1024, nQuery);

        tokenCount = nQuery + 1;
        var output = new float[tokenCount * _projDim];
        fixed (float* fcB = _fcB) VisionOps.MatVecAny(queries, _fcW, fcB, nQuery, _qEmbd, _projDim, output);
        Array.Copy(_viewSep, 0, output, nQuery * _projDim, _projDim);
        return output;
    }

    // ---------------------------------------------------------------- SAM

    internal float[] Sam(float[] chw, int imageSize, out int outSide)
    {
        int g = imageSize / SamPatch;                                         // 64
        int n = g * g;
        // Patch embedding: 16x16 stride-16 conv as im2col (ci, ky, kx) x matmul.
        int k = 3 * SamPatch * SamPatch;
        var cols = new float[n * k];
        int plane = imageSize * imageSize;
        Parallel.For(0, g, py =>
        {
            for (int px = 0; px < g; px++)
            {
                int o = (py * g + px) * k;
                for (int c = 0; c < 3; c++)
                    for (int dy = 0; dy < SamPatch; dy++)
                        for (int dx = 0; dx < SamPatch; dx++)
                            cols[o + (c * SamPatch + dy) * SamPatch + dx] = chw[c * plane + (py * SamPatch + dy) * imageSize + px * SamPatch + dx];
            }
        });
        var x = new float[n * SamEmbd];
        fixed (float* b = _patchB) VisionOps.MatVec(cols, _patchW, b, n, k, SamEmbd, x);
        if (_posEmbd.Length != x.Length)
            throw new NotSupportedException("DeepSeek-OCR2: SAM position-table interpolation (768-px tiles) is not implemented.");
        TensorPrimitives.Add(x, _posEmbd, x);
        Dump("sam_pos", x);

        var normed = new float[n * SamEmbd];
        var attnOut = new float[n * SamEmbd];
        var mid = new float[n * 4 * SamEmbd];
        for (int l = 0; l < SamLayers; l++)
        {
            var blk = _sam[l];
            Array.Copy(x, normed, x.Length);
            fixed (float* w = blk.Ln1W, bb = blk.Ln1B) VisionOps.LayerNorm(normed, n, SamEmbd, w, bb, SamEps);
            if (Array.IndexOf(GlobalLayers, l) >= 0)
                SamAttention(normed, g, g, blk, attnOut);
            else
                SamWindowedAttention(normed, g, blk, attnOut);
            TensorPrimitives.Add(x, attnOut, x);

            Array.Copy(x, normed, x.Length);
            fixed (float* w = blk.Ln2W, bb = blk.Ln2B) VisionOps.LayerNorm(normed, n, SamEmbd, w, bb, SamEps);
            fixed (float* b1 = blk.Lin1B) VisionOps.MatVecAny(normed, blk.Lin1W, b1, n, SamEmbd, 4 * SamEmbd, mid);
            VisionOps.Gelu(mid);
            fixed (float* b2 = blk.Lin2B) VisionOps.MatVecAny(mid, blk.Lin2W, b2, n, 4 * SamEmbd, SamEmbd, attnOut);
            TensorPrimitives.Add(x, attnOut, x);
            Dump($"sam_layer_out-{l}", x);
        }

        // Neck + downsampling convs (channels-last, raster tokens).
        float[] y = Conv2d(x, g, g, SamEmbd, _neck0, 256, 1, 1, 0, out _, out _);
        ChannelLayerNorm(y, g * g, 256, _neck1W, _neck1B);
        y = Conv2d(y, g, g, 256, _neck2, 256, 3, 1, 1, out _, out _);
        ChannelLayerNorm(y, g * g, 256, _neck3W, _neck3B);
        y = Conv2d(y, g, g, 256, _net2, 512, 3, 2, 1, out int h2, out int w2);
        y = Conv2d(y, h2, w2, 512, _net3, _qEmbd, 3, 2, 1, out int h3, out _);
        Dump("sam_output", y);
        outSide = h3;
        return y;
    }

    private static void ChannelLayerNorm(float[] y, int n, int c, float[] w, float[] b)
    {
        fixed (float* pw = w, pb = b) VisionOps.LayerNorm(y, n, c, pw, pb, SamEps);
    }

    /// <summary>2D conv, no bias, channels-last input [h][w][cin]; weights GGUF [kw, kh, cin, cout] (flat
    /// ((co*cin + ci)*k + ky)*k + kx). Returns [ho][wo][cout].</summary>
    private static float[] Conv2d(float[] input, int h, int w, int cin, float[] weights, int cout, int k, int stride, int pad, out int ho, out int wo)
    {
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        ho = oh; wo = ow;
        int kk = cin * k * k;
        var cols = new float[oh * ow * kk];
        Parallel.For(0, oh, oy =>
        {
            for (int ox = 0; ox < ow; ox++)
            {
                int o = (oy * ow + ox) * kk;
                for (int ky = 0; ky < k; ky++)
                {
                    int iy = oy * stride + ky - pad;
                    for (int kx = 0; kx < k; kx++)
                    {
                        int ix = ox * stride + kx - pad;
                        bool inside = iy >= 0 && iy < h && ix >= 0 && ix < w;
                        int src = (iy * w + ix) * cin;
                        for (int ci = 0; ci < cin; ci++)
                            cols[o + (ci * k + ky) * k + kx] = inside ? input[src + ci] : 0f;
                    }
                }
            }
        });
        var output = new float[oh * ow * cout];
        VisionOps.MatVec(cols, weights, null, oh * ow, kk, cout, output);
        return output;
    }

    /// <summary>Global SAM attention over an <paramref name="gh"/> x <paramref name="gw"/> grid.</summary>
    private static void SamAttention(float[] normed, int gh, int gw, SamBlock blk, float[] output)
    {
        int n = gh * gw;
        var qkv = new float[n * 3 * SamEmbd];
        fixed (float* b = blk.QkvB) VisionOps.MatVecAny(normed, blk.QkvW, b, n, SamEmbd, 3 * SamEmbd, qkv);
        var attn = new float[n * SamEmbd];
        AttendGrid(qkv, gh, gw, blk, attn);
        fixed (float* ob = blk.OutB) VisionOps.MatVecAny(attn, blk.OutW, ob, n, SamEmbd, SamEmbd, output);
    }

    /// <summary>14x14 windowed SAM attention: the grid is zero-padded (after the norm, as ggml_pad) to a
    /// multiple of the window, padded tokens take part as keys (their q/k/v are the qkv bias).</summary>
    private static void SamWindowedAttention(float[] normed, int g, SamBlock blk, float[] output)
    {
        int nw = (g + SamWindow - 1) / SamWindow;
        int gp = nw * SamWindow;
        int wn = SamWindow * SamWindow;
        var win = new float[nw * nw * wn * SamEmbd];
        for (int wy = 0; wy < nw; wy++)
            for (int wx = 0; wx < nw; wx++)
                for (int iy = 0; iy < SamWindow; iy++)
                    for (int ix = 0; ix < SamWindow; ix++)
                    {
                        int y = wy * SamWindow + iy, x = wx * SamWindow + ix;
                        if (y >= g || x >= g) continue;
                        Array.Copy(normed, (y * g + x) * SamEmbd, win, (((wy * nw + wx) * wn) + iy * SamWindow + ix) * SamEmbd, SamEmbd);
                    }
        int total = nw * nw * wn;
        var qkv = new float[total * 3 * SamEmbd];
        fixed (float* b = blk.QkvB) VisionOps.MatVecAny(win, blk.QkvW, b, total, SamEmbd, 3 * SamEmbd, qkv);
        var attnWin = new float[total * SamEmbd];
        Parallel.For(0, nw * nw, wi =>
        {
            var q = new float[wn * 3 * SamEmbd];
            Array.Copy(qkv, wi * wn * 3 * SamEmbd, q, 0, q.Length);
            var o = new float[wn * SamEmbd];
            AttendGrid(q, SamWindow, SamWindow, blk, o, parallel: false);
            Array.Copy(o, 0, attnWin, wi * wn * SamEmbd, o.Length);
        });
        var proj = new float[total * SamEmbd];
        fixed (float* ob = blk.OutB) VisionOps.MatVecAny(attnWin, blk.OutW, ob, total, SamEmbd, SamEmbd, proj);
        for (int wy = 0; wy < nw; wy++)
            for (int wx = 0; wx < nw; wx++)
                for (int iy = 0; iy < SamWindow; iy++)
                    for (int ix = 0; ix < SamWindow; ix++)
                    {
                        int y = wy * SamWindow + iy, x = wx * SamWindow + ix;
                        if (y >= g || x >= g) continue;
                        Array.Copy(proj, (((wy * nw + wx) * wn) + iy * SamWindow + ix) * SamEmbd, output, (y * g + x) * SamEmbd, SamEmbd);
                    }
        _ = gp;
    }

    /// <summary>
    /// Multi-head attention over an <paramref name="h"/> x <paramref name="w"/> token grid (raster, x fastest)
    /// with SAM's decomposed relative-position bias: score(q, k) = scale * q.k + q.Rh[qy - ky + h - 1] +
    /// q.Rw[qx - kx + w - 1] (unscaled q in the bias terms, as ggml builds it into the attention mask).
    /// <paramref name="qkv"/> is per token [q | k | v] (3 x 768).
    /// </summary>
    private static void AttendGrid(float[] qkv, int h, int w, SamBlock blk, float[] output, bool parallel = true)
    {
        const int dh = SamEmbd / SamHeads;
        int n = h * w;
        float scale = 1f / MathF.Sqrt(dh);
        int stride = 3 * SamEmbd;
        void Head(int hd)
        {
            var scores = new float[n];
            var rh = new float[h];
            var rw = new float[w];
            for (int i = 0; i < n; i++)
            {
                int qy = i / w, qx = i % w;
                var qi = new ReadOnlySpan<float>(qkv, i * stride + hd * dh, dh);
                for (int ky = 0; ky < h; ky++) rh[ky] = TensorPrimitives.Dot(qi, new ReadOnlySpan<float>(blk.RelH, (qy - ky + h - 1) * dh, dh));
                for (int kx = 0; kx < w; kx++) rw[kx] = TensorPrimitives.Dot(qi, new ReadOnlySpan<float>(blk.RelW, (qx - kx + w - 1) * dh, dh));
                float max = float.NegativeInfinity;
                for (int j = 0; j < n; j++)
                {
                    float s = TensorPrimitives.Dot(qi, new ReadOnlySpan<float>(qkv, j * stride + SamEmbd + hd * dh, dh)) * scale
                              + rh[j / w] + rw[j % w];
                    scores[j] = s;
                    if (s > max) max = s;
                }
                float sum = 0;
                for (int j = 0; j < n; j++) { float e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
                var o = new Span<float>(output, i * SamEmbd + hd * dh, dh);
                o.Clear();
                float inv = 1f / sum;
                for (int j = 0; j < n; j++)
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(qkv, j * stride + 2 * SamEmbd + hd * dh, dh), scores[j] * inv, o, o);
            }
        }
        if (parallel) Parallel.For(0, SamHeads, Head);
        else for (int hd = 0; hd < SamHeads; hd++) Head(hd);
    }

    // ---------------------------------------------------------------- Qwen2 encoder

    private float[] Qwen2Encoder(float[] image, int nImg, float[] queries, int nQuery)
    {
        int n = nImg + nQuery, d = _qEmbd, dh = d / _qHeads, kvd = _qKvHeads * dh;
        var x = new float[n * d];
        Array.Copy(image, x, nImg * d);
        Array.Copy(queries, 0, x, nImg * d, nQuery * d);

        var normed = new float[n * d];
        var q = new float[n * d];
        var k = new float[n * kvd];
        var v = new float[n * kvd];
        var attn = new float[n * d];
        var tmp = new float[n * d];
        var gate = new float[n * _qFfn];
        var up = new float[n * _qFfn];
        for (int l = 0; l < _qLayers; l++)
        {
            var blk = _q[l];
            Array.Copy(x, normed, x.Length);
            fixed (float* w = blk.Ln1) VisionOps.RmsNorm(normed, n, d, w, _qEps);
            fixed (float* qb = blk.QB, kb = blk.KB, vb = blk.VB)
            {
                VisionOps.MatVecAny(normed, blk.QW, qb, n, d, d, q);
                VisionOps.MatVecAny(normed, blk.KW, kb, n, d, kvd, k);
                VisionOps.MatVecAny(normed, blk.VW, vb, n, d, kvd, v);
            }
            NeoxRope(q, n, _qHeads, dh);
            NeoxRope(k, n, _qKvHeads, dh);
            MaskedGqa(q, k, v, n, nImg, dh, attn);
            VisionOps.MatVecAny(attn, blk.OW, null, n, d, d, tmp);
            TensorPrimitives.Add(x, tmp, x);

            Array.Copy(x, normed, x.Length);
            fixed (float* w = blk.Ln2) VisionOps.RmsNorm(normed, n, d, w, _qEps);
            VisionOps.MatVecAny(normed, blk.GateW, null, n, d, _qFfn, gate);
            VisionOps.MatVecAny(normed, blk.UpW, null, n, d, _qFfn, up);
            VisionOps.Silu(gate);
            TensorPrimitives.Multiply(gate, up, gate);
            VisionOps.MatVecAny(gate, blk.DownW, null, n, _qFfn, d, tmp);
            TensorPrimitives.Add(x, tmp, x);
        }
        fixed (float* w = _postLn) VisionOps.RmsNorm(x, n, d, w, _qEps);
        var outQ = new float[nQuery * d];
        Array.Copy(x, nImg * d, outQ, 0, outQ.Length);
        Dump("qwen2_queries", outQ);
        return outQ;
    }

    /// <summary>NeoX RoPE over the whole head (pairs i, i + dh/2), theta 1e6, position = token index.</summary>
    private static void NeoxRope(float[] t, int n, int heads, int dh)
    {
        int half = dh / 2;
        var freq = new float[half];
        for (int i = 0; i < half; i++) freq[i] = MathF.Pow(1_000_000f, -2f * i / dh);
        Parallel.For(0, n, p =>
        {
            for (int h = 0; h < heads; h++)
            {
                int o = (p * heads + h) * dh;
                for (int i = 0; i < half; i++)
                {
                    float a = p * freq[i], c = MathF.Cos(a), s = MathF.Sin(a);
                    float x0 = t[o + i], x1 = t[o + i + half];
                    t[o + i] = x0 * c - x1 * s;
                    t[o + i + half] = x0 * s + x1 * c;
                }
            }
        });
    }

    /// <summary>GQA with the OCR2 mask: tokens below <paramref name="nImg"/> see image tokens only; query
    /// tokens see every image token plus queries up to and including themselves.</summary>
    private void MaskedGqa(float[] q, float[] k, float[] v, int n, int nImg, int dh, float[] output)
    {
        int d = _qHeads * dh, kvd = _qKvHeads * dh, group = _qHeads / _qKvHeads;
        float scale = 1f / MathF.Sqrt(dh);
        Parallel.For(0, _qHeads * n, job =>
        {
            int h = job / n, i = job % n, kvh = h / group;
            int limit = i < nImg ? nImg : i + 1;
            var scores = new float[limit];
            var qi = new ReadOnlySpan<float>(q, i * d + h * dh, dh);
            float max = float.NegativeInfinity;
            for (int j = 0; j < limit; j++)
            {
                float s = TensorPrimitives.Dot(qi, new ReadOnlySpan<float>(k, j * kvd + kvh * dh, dh)) * scale;
                scores[j] = s;
                if (s > max) max = s;
            }
            float sum = 0;
            for (int j = 0; j < limit; j++) { float e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
            var o = new Span<float>(output, i * d + h * dh, dh);
            o.Clear();
            float inv = 1f / sum;
            for (int j = 0; j < limit; j++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(v, j * kvd + kvh * dh, dh), scores[j] * inv, o, o);
        });
    }

    private static readonly bool s_dump = Environment.GetEnvironmentVariable("STINGRAY_VIS_DUMP") == "1";

    private static void Dump(string name, float[] a)
    {
        if (!s_dump) return;
        double sum = 0;
        foreach (float f in a) sum += f;
        Console.WriteLine($"[dsocr2] {name,-18} sum={sum,16:F4} [{a[0]:F4}, {a[1]:F4}, {a[2]:F4}]");
    }
}
