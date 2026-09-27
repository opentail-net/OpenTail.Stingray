using System.Numerics.Tensors;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// StepFun Step3-VL vision tower, ported from llama.cpp <c>tools/mtmd/models/step3vl.cpp</c> (2026-09-27):
/// 14-px patch conv (no bias) + the learned 52x52 position table resized to the grid (antialiased
/// bilinear), <c>pre_ln</c>, 47 pre-norm blocks (LayerNorm, fused <c>attn_qkv</c>, 2D RoPE on column/row
/// halves, <c>ls1</c>/<c>ls2</c> layer scales, QuickGELU MLP), then two 3x3 stride-2 conv downsamplers with
/// bias (1536 -> 3072 -> 6144, no activation) and the bias-free <c>mm.model.fc</c> projection (6144 -> 4096).
/// A 448-px image gives 32x32 patches -> 8x8 = 64 tokens.
/// </summary>
/// <remarks>
/// Rewritten 2026-09-27 against <c>llama-mtmd-debug</c>. The earlier port read separate q/k/v tensors that
/// the GGUF does not have (it ships a fused <c>attn_qkv</c>), had no RoPE, no layer scales, a gated/GELU FFN
/// instead of QuickGELU, and a pixel-shuffle + linear downsampler instead of the two convolutions.
/// </remarks>
public sealed unsafe class Step3VlVisionEncoder
{
    private readonly Step3VlVisionModel _m;
    private readonly int _embd, _heads, _headDim, _layers, _projDim;
    private readonly float _eps;
    private readonly float[] _patchEmbdW, _posEmbd;
    private readonly float[]? _patchEmbdB, _preLnW, _preLnB, _postLnW, _postLnB;
    private readonly VisionTensorRef _mm0W, _mm1W, _fcW;
    private readonly float[] _mm0B, _mm1B;
    private readonly int _mm0Out, _mm1Out;
    private readonly Block[] _blocks;

    private sealed class Block
    {
        public required float[] Ln1W, Ln1B, Ln2W, Ln2B, QkvB, OutB, UpB, DownB;
        public float[]? Ls1, Ls2;
        public required VisionTensorRef QkvW, OutW, UpW, DownW;
        public int Ffn;
    }

    public int ProjectionDim => _projDim;

    public Step3VlVisionEncoder(Step3VlVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _eps = model.Eps;
        var g = model.Gguf;

        _patchEmbdW = VisionOps.LoadTensorF32(g, "v.patch_embd.weight") ?? throw new InvalidOperationException("Missing v.patch_embd.weight");
        _patchEmbdB = VisionOps.GetTensorArray(g, "v.patch_embd.bias");
        _posEmbd = VisionOps.LoadTensorF32(g, "v.position_embd.weight") ?? [];
        _preLnW = VisionOps.GetTensorArray(g, "v.pre_ln.weight");
        _preLnB = VisionOps.GetTensorArray(g, "v.pre_ln.bias");
        _postLnW = VisionOps.GetTensorArray(g, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorArray(g, "v.post_ln.bias");
        _mm0W = VisionOps.GetTensor(g, "mm.0.weight");
        _mm1W = VisionOps.GetTensor(g, "mm.1.weight");
        _fcW = VisionOps.GetTensor(g, "mm.model.fc.weight");
        _mm0B = VisionOps.GetTensorArray(g, "mm.0.bias") ?? throw new InvalidOperationException("Missing mm.0.bias");
        _mm1B = VisionOps.GetTensorArray(g, "mm.1.bias") ?? throw new InvalidOperationException("Missing mm.1.bias");
        _mm0Out = _mm0B.Length;
        _mm1Out = _mm1B.Length;
        _projDim = _fcW.IsValid ? (int)_fcW.Info.Dimensions[1] : model.ProjectionDim;

        _blocks = new Block[_layers];
        for (int l = 0; l < _layers; l++)
        {
            string p = $"v.blk.{l}.";
            var up = VisionOps.GetTensor(g, p + "ffn_up.weight");
            _blocks[l] = new Block
            {
                Ln1W = Arr(p + "ln1.weight"), Ln1B = Arr(p + "ln1.bias"),
                Ln2W = Arr(p + "ln2.weight"), Ln2B = Arr(p + "ln2.bias"),
                QkvW = VisionOps.GetTensor(g, p + "attn_qkv.weight"), QkvB = Arr(p + "attn_qkv.bias"),
                OutW = VisionOps.GetTensor(g, p + "attn_out.weight"), OutB = Arr(p + "attn_out.bias"),
                UpW = up, UpB = Arr(p + "ffn_up.bias"),
                DownW = VisionOps.GetTensor(g, p + "ffn_down.weight"), DownB = Arr(p + "ffn_down.bias"),
                Ls1 = VisionOps.GetTensorArray(g, p + "ls1.weight"),
                Ls2 = VisionOps.GetTensorArray(g, p + "ls2.weight"),
                Ffn = (int)up.Info.Dimensions[1],
            };
        }

        float[] Arr(string name) => VisionOps.GetTensorArray(g, name) ?? throw new InvalidOperationException($"Missing {name}");
    }

    public float[] Forward(
        float[] chw, int imgWidth, int imgHeight,
        int patchesX, int patchesY,
        out int tokenCount)
    {
        int ps = _m.PatchSize;
        int n = patchesX * patchesY;
        int e = _embd;

        var x = new float[n * e];
        Im2ColAndEmbed(chw, imgWidth, imgHeight, ps, patchesX, patchesY, x);
        if (_posEmbd.Length > 0)
            TensorPrimitives.Add(x, VisionOps.ResizePositionEmbeddings(_posEmbd, e, patchesX, patchesY), x);
        if (_preLnW != null)
            fixed (float* w = _preLnW, b = _preLnB) VisionOps.LayerNorm(x, n, e, w, b, _eps);

        var normed = new float[n * e];
        var qkv = new float[n * 3 * e];
        var q = new float[n * e];
        var k = new float[n * e];
        var v = new float[n * e];
        var attn = new float[n * e];
        var tmp = new float[n * e];
        int maxFfn = _blocks.Max(b => b.Ffn);
        var mid = new float[n * maxFfn];

        for (int l = 0; l < _layers; l++)
        {
            var b = _blocks[l];
            Array.Copy(x, normed, x.Length);
            fixed (float* w = b.Ln1W, bb = b.Ln1B) VisionOps.LayerNorm(normed, n, e, w, bb, _eps);
            fixed (float* qb = b.QkvB) VisionOps.MatVecAny(normed, b.QkvW, qb, n, e, 3 * e, qkv);
            for (int t = 0; t < n; t++)
            {
                Array.Copy(qkv, t * 3 * e, q, t * e, e);
                Array.Copy(qkv, t * 3 * e + e, k, t * e, e);
                Array.Copy(qkv, t * 3 * e + 2 * e, v, t * e, e);
            }
            VisionOps.Rope2dHalves(q, k, patchesX, patchesY, _heads, _headDim, _m.RopeTheta);
            VisionOps.Attention(q, k, v, n, _heads, _headDim, attn);
            fixed (float* ob = b.OutB) VisionOps.MatVecAny(attn, b.OutW, ob, n, e, e, tmp);
            if (b.Ls1 != null) ScaleChannels(tmp, n, e, b.Ls1);
            TensorPrimitives.Add(x, tmp, x);

            Array.Copy(x, normed, x.Length);
            fixed (float* w = b.Ln2W, bb = b.Ln2B) VisionOps.LayerNorm(normed, n, e, w, bb, _eps);
            var m = new Span<float>(mid, 0, n * b.Ffn);
            fixed (float* ub = b.UpB) VisionOps.MatVecAny(normed, b.UpW, ub, n, e, b.Ffn, mid);
            VisionOps.QuickGelu(m);
            fixed (float* db = b.DownB) VisionOps.MatVecAny(mid, b.DownW, db, n, b.Ffn, e, tmp);
            if (b.Ls2 != null) ScaleChannels(tmp, n, e, b.Ls2);
            TensorPrimitives.Add(x, tmp, x);
        }
        if (_postLnW != null)
            fixed (float* w = _postLnW, bb = _postLnB) VisionOps.LayerNorm(x, n, e, w, bb, _eps);

        // Two 3x3 stride-2 convs with bias, no activation (step3vl.cpp downsample_0/_1).
        float[] cols0 = VisionOps.Im2Col(x, patchesY, patchesX, e, 3, 2, 1, out int h1, out int w1);
        var d0 = new float[h1 * w1 * _mm0Out];
        fixed (float* b0 = _mm0B) VisionOps.MatVecAny(cols0, _mm0W, b0, h1 * w1, e * 9, _mm0Out, d0);
        float[] cols1 = VisionOps.Im2Col(d0, h1, w1, _mm0Out, 3, 2, 1, out int h2, out int w2);
        var d1 = new float[h2 * w2 * _mm1Out];
        fixed (float* b1 = _mm1B) VisionOps.MatVecAny(cols1, _mm1W, b1, h2 * w2, _mm0Out * 9, _mm1Out, d1);

        tokenCount = h2 * w2;
        var output = new float[tokenCount * _projDim];
        VisionOps.MatVecAny(d1, _fcW, null, tokenCount, _mm1Out, _projDim, output);
        return output;
    }

    private static void ScaleChannels(float[] a, int n, int c, float[] scale)
    {
        for (int t = 0; t < n; t++)
        {
            var row = new Span<float>(a, t * c, c);
            TensorPrimitives.Multiply(row, scale, row);
        }
    }

    private void Im2ColAndEmbed(float[] chw, int imgW, int imgH, int ps, int px, int py, float[] dst)
    {
        int nP = px * py;
        int patchDim = 3 * ps * ps;
        var patches = new float[nP * patchDim];
        Parallel.For(0, py, iy =>
        {
            for (int ix = 0; ix < px; ix++)
            {
                int dstOff = (iy * px + ix) * patchDim;
                for (int c = 0; c < 3; c++)
                {
                    int cOff = c * imgW * imgH;
                    for (int sy = 0; sy < ps; sy++)
                        for (int sx = 0; sx < ps; sx++)
                        {
                            int srcY = iy * ps + sy, srcX = ix * ps + sx;
                            if (srcY < imgH && srcX < imgW)
                                patches[dstOff + c * ps * ps + sy * ps + sx] = chw[cOff + srcY * imgW + srcX];
                        }
                }
            }
        });
        fixed (float* b = _patchEmbdB) VisionOps.MatVec(patches, _patchEmbdW, b, nP, patchDim, _embd, dst);
    }
}
