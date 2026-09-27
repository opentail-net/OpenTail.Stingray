using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). Mamba-2 mixer layers for Mamba-2
// hybrids (IBM Granite 4.0-H, `granitehybrid`): a layer whose attention.head_count_kv entry is 0
// runs this recurrent mixer in place of attention, then shares the usual residual + FFN path.
// Reference: llama.cpp src/models/mamba-base.cpp build_mamba2_layer and the ggml CPU ssm_conv /
// ssm_scan kernels (ggml-cpu/ops.cpp).
public sealed unsafe partial class ForwardPass
{
    private Mamba2Config? _m2;
    private TensorRef[]? _m2In, _m2Out, _m2ConvW, _m2ConvB, _m2DtB, _m2A, _m2D, _m2Norm;
    // Per-layer recurrent state (null for attention layers):
    //   conv: [convDim][convKernel - 1], oldest input first per channel
    //   ssm:  [numHeads][headDim][stateSize]
    private float*[]? _m2ConvState, _m2SsmState;
    private float* _m2Proj, _m2Xbc, _m2Y, _m2ZeroKv;

    private bool IsMamba2Layer(int layer) => _hp.IsMamba2Layer is { } rec && rec[layer];

    /// <summary>The paged KV cache allocates each position's block on layer 0's append and tracks positions per
    /// layer, so layers without attention append a zero row. TODO(perf): skip the storage.</summary>
    private void AppendZeroKv(int layer)
    {
        var zero = new ReadOnlySpan<float>(_m2ZeroKv, _kvCache.KvDim);
        if (_tqKvCache != null) _tqKvCache.Append(layer, zero, zero);
        else _kvCache.Append(layer, zero, zero);
    }

    private void InitMamba2(int numLayers)
    {
        if (_hp.Mamba2 is not { } c || _hp.IsMamba2Layer is null) return;
        _m2 = c;
        _m2In = new TensorRef[numLayers];
        _m2Out = new TensorRef[numLayers];
        _m2ConvW = new TensorRef[numLayers];
        _m2ConvB = new TensorRef[numLayers];
        _m2DtB = new TensorRef[numLayers];
        _m2A = new TensorRef[numLayers];
        _m2D = new TensorRef[numLayers];
        _m2Norm = new TensorRef[numLayers];
        _m2ConvState = new float*[numLayers];
        _m2SsmState = new float*[numLayers];
        int convState = c.ConvDim * (c.ConvKernel - 1);
        int ssmState = c.NumHeads * c.HeadDim * c.StateSize;
        for (int i = 0; i < numLayers; i++)
        {
            if (!IsMamba2Layer(i)) continue;
            _m2In[i] = ResolveTensor($"blk.{i}.ssm_in.weight");
            _m2Out[i] = ResolveTensor($"blk.{i}.ssm_out.weight");
            _m2ConvW[i] = ResolveTensor($"blk.{i}.ssm_conv1d.weight");
            if (_model.FindTensor($"blk.{i}.ssm_conv1d.bias") is not null)
                _m2ConvB[i] = ResolveTensor($"blk.{i}.ssm_conv1d.bias");
            _m2DtB[i] = ResolveTensor($"blk.{i}.ssm_dt.bias");
            _m2A[i] = ResolveTensor($"blk.{i}.ssm_a");
            _m2D[i] = ResolveTensor($"blk.{i}.ssm_d");
            if (_model.FindTensor($"blk.{i}.ssm_norm.weight") is not null)
                _m2Norm[i] = ResolveTensor($"blk.{i}.ssm_norm.weight");
            _m2ConvState[i] = Alloc(convState);
            _m2SsmState[i] = Alloc(ssmState);
        }
        _m2Proj = Alloc(c.InProjDim);
        _m2Xbc = Alloc(c.ConvDim);
        _m2Y = Alloc(c.InnerSize);
        _m2ZeroKv = Alloc(Math.Max(1, _numKvHeads * _maxHeadDim));
        new Span<float>(_m2ZeroKv, Math.Max(1, _numKvHeads * _maxHeadDim)).Clear();
        ResetMamba2State();
    }

    /// <summary>Zeroes every recurrent state (a new sequence). The state cannot be rewound to an earlier
    /// position the way a KV cache can.</summary>
    private void ResetMamba2State()
    {
        if (_m2 is not { } c) return;
        int convState = c.ConvDim * (c.ConvKernel - 1);
        int ssmState = c.NumHeads * c.HeadDim * c.StateSize;
        for (int i = 0; i < _m2ConvState!.Length; i++)
        {
            if (_m2ConvState[i] is null) continue;
            new Span<float>(_m2ConvState[i], convState).Clear();
            new Span<float>(_m2SsmState![i], ssmState).Clear();
        }
    }

    /// <summary>One token through the Mamba-2 mixer of <paramref name="layer"/>: <paramref name="input"/> is the
    /// normed hidden state [embDim], <paramref name="output"/> receives the mixer output [embDim] (before the
    /// residual add).</summary>
    private void Mamba2Step(int layer, float* input, float* output)
    {
        var c = _m2!;
        int di = c.InnerSize, convDim = c.ConvDim, k = c.ConvKernel, nh = c.NumHeads, hd = c.HeadDim;
        int ds = c.StateSize, ng = c.NumGroups;

        // in_proj -> [z | xBC | dt]
        FusedMatVec(_m2Proj, _m2In![layer], input, c.InProjDim, _embDim);
        float* z = _m2Proj;
        float* xbcIn = _m2Proj + di;
        float* dt = _m2Proj + di + convDim;

        // Causal depthwise conv over the last k inputs (state holds the k-1 previous), bias, SiLU (ggml_ssm_conv).
        float* w = GetNormWeight(_m2ConvW![layer]);
        float* bias = _m2ConvB![layer].DataPtr is null ? null : GetNormWeight(_m2ConvB[layer]);
        float* st = _m2ConvState![layer];
        int km1 = k - 1;
        for (int ch = 0; ch < convDim; ch++)
        {
            float* sc = st + ch * km1;
            float* wc = w + ch * k;
            float sum = 0f;
            for (int j = 0; j < km1; j++) sum += sc[j] * wc[j];
            sum += xbcIn[ch] * wc[km1];
            for (int j = 0; j + 1 < km1; j++) sc[j] = sc[j + 1];
            if (km1 > 0) sc[km1 - 1] = xbcIn[ch];
            if (bias is not null) sum += bias[ch];
            _m2Xbc[ch] = sum / (1f + MathF.Exp(-sum));
        }

        // Selective scan, one step (ggml_ssm_scan, Mamba-2 scalar-A branch).
        float* x = _m2Xbc;
        float* bMat = _m2Xbc + di;
        float* cMat = bMat + ng * ds;
        float* a = GetNormWeight(_m2A![layer]);
        float* d = GetNormWeight(_m2D![layer]);
        float* dtb = GetNormWeight(_m2DtB![layer]);
        float* s = _m2SsmState![layer];
        int headsPerGroup = nh / ng;
        float* yOut = _m2Y;
        bool ggmlOrder = Fma.IsSupported && Avx.IsSupported && Sse3.IsSupported && ds % 32 == 0;
        // Heads are independent: spread them over threads, and vectorise the d_state loop (SIMD over n).
        Parallel.For(0, nh, h =>
        {
            float dtv = dt[h] + dtb[h];
            float sp = dtv > 20f ? dtv : MathF.Log(1f + MathF.Exp(dtv));
            float dA = MathF.Exp(sp * a[h]);
            int g = h / headsPerGroup;
            float* bg = bMat + g * ds;
            float* cg = cMat + g * ds;
            int vw = System.Numerics.Vector<float>.Count;
            int nVec = ds - ds % vw;
            var vdA = new System.Numerics.Vector<float>(dA);
            for (int e = 0; e < hd; e++)
            {
                int ii = h * hd + e;
                float xdt = x[ii] * sp;
                var vxdt = new System.Numerics.Vector<float>(xdt);
                float* sRow = s + (long)ii * ds;
                int n = 0;
                float acc;
                if (ggmlOrder)
                {
                    // Same arithmetic order as ggml's AVX2 ssm_scan (GGML_F32_STEP 32 = 4 accumulators x 8 lanes):
                    // t0 = s*dA + B*x*dt unfused, sum[j] = fma(t0, C, sum[j]), then GGML_F32_VEC_REDUCE. The
                    // recurrence is sensitive to reduction order at about the 0.3% perplexity level, so match it.
                    var adA = Vector256.Create(dA);
                    var axdt = Vector256.Create(xdt);
                    Vector256<float> s0 = Vector256<float>.Zero, s1 = s0, s2 = s0, s3 = s0;
                    for (; n + 32 <= ds; n += 32)
                    {
                        var t0 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + n), adA), Avx.Multiply(Avx.LoadVector256(bg + n), axdt));
                        Avx.Store(sRow + n, t0);
                        s0 = Fma.MultiplyAdd(t0, Avx.LoadVector256(cg + n), s0);
                        var t1 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + n + 8), adA), Avx.Multiply(Avx.LoadVector256(bg + n + 8), axdt));
                        Avx.Store(sRow + n + 8, t1);
                        s1 = Fma.MultiplyAdd(t1, Avx.LoadVector256(cg + n + 8), s1);
                        var t2 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + n + 16), adA), Avx.Multiply(Avx.LoadVector256(bg + n + 16), axdt));
                        Avx.Store(sRow + n + 16, t2);
                        s2 = Fma.MultiplyAdd(t2, Avx.LoadVector256(cg + n + 16), s2);
                        var t3 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + n + 24), adA), Avx.Multiply(Avx.LoadVector256(bg + n + 24), axdt));
                        Avx.Store(sRow + n + 24, t3);
                        s3 = Fma.MultiplyAdd(t3, Avx.LoadVector256(cg + n + 24), s3);
                    }
                    s0 = Avx.Add(s0, s2);
                    s1 = Avx.Add(s1, s3);
                    s0 = Avx.Add(s0, s1);
                    var h128 = Sse.Add(s0.GetLower(), s0.GetUpper());
                    h128 = Sse3.HorizontalAdd(h128, h128);
                    h128 = Sse3.HorizontalAdd(h128, h128);
                    acc = h128.ToScalar();
                }
                else
                {
                    var vacc = System.Numerics.Vector<float>.Zero;
                    for (; n < nVec; n += vw)
                    {
                        var v = System.Numerics.Vector.Load(sRow + n) * vdA + System.Numerics.Vector.Load(bg + n) * vxdt;
                        System.Numerics.Vector.Store(v, sRow + n);
                        vacc += v * System.Numerics.Vector.Load(cg + n);
                    }
                    acc = System.Numerics.Vector.Sum(vacc);
                }
                for (; n < ds; n++)
                {
                    float v = sRow[n] * dA + bg[n] * xdt;
                    sRow[n] = v;
                    acc += v * cg[n];
                }
                // y += x * D, then gate with silu(z) (ggml_swiglu_split(z, y)).
                float y = acc + x[ii] * d[h];
                float zv = z[ii];
                yOut[ii] = y * (zv / (1f + MathF.Exp(-zv)));
            }
        });

        // Grouped RMSNorm over d_inner / n_group channels.
        if (_m2Norm![layer].DataPtr is not null)
        {
            float* nw = GetNormWeight(_m2Norm[layer]);
            int gs = di / ng;
            for (int g = 0; g < ng; g++)
            {
                float* yg = _m2Y + g * gs;
                double ss = 0;
                for (int i = 0; i < gs; i++) ss += (double)yg[i] * yg[i];
                float inv = 1f / MathF.Sqrt((float)(ss / gs) + _hp.RmsNormEps);
                for (int i = 0; i < gs; i++) yg[i] = yg[i] * inv * nw[g * gs + i];
            }
        }

        FusedMatVec(output, _m2Out![layer], _m2Y, _embDim, di);
    }
}
