using System.Runtime.InteropServices;
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
    private void EnsureZeroKv()
    {
        if (_m2ZeroKv is not null) return;
        int n = Math.Max(1, _numKvHeads * _maxHeadDim);
        _m2ZeroKv = Alloc(n);
        new Span<float>(_m2ZeroKv, n).Clear();
    }

    /// <summary>True when some layers carry no attention (Mamba-2, short conv, MLP-only), so the paged cache's block
    /// has to be reserved per token instead of being allocated by layer 0's append.</summary>
    private bool HasNonAttentionLayers => HasRecurrentState || _hp.HybridFfnOnlyLayer is not null;

    /// <summary>
    /// Bookkeeping for a layer without attention. With the paged cache nothing is stored: <see cref="RunTrunk"/>
    /// reserves the block (<see cref="PagedKvCache.ReserveBlock"/>) and pages are allocated per layer on first
    /// write, so these layers never allocate KV memory (docs/103 item 9; until 2026-09-27 each wrote a zero row).
    /// The TurboQuant cache has no reserve path, so it still gets a zero row.
    /// </summary>
    private void AppendZeroKv(int layer)
    {
        if (_tqKvCache is null) return;
        var zero = new ReadOnlySpan<float>(_m2ZeroKv, _kvCache.KvDim);
        _tqKvCache.Append(layer, zero, zero);
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
        EnsureZeroKv();
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
        FusedMatVec(_m2Proj, _m2In![layer], input, c.InProjDim, _embDim);
        Mamba2Mix(layer, _m2Proj, 1, _m2Y, _m2Xbc);
        FusedMatVec(output, _m2Out![layer], _m2Y, _embDim, c.InnerSize);
        StageCapture.Record("cpu", layer, StageCapture.Stages.OProj, new ReadOnlySpan<float>(output, _embDim));
    }

    /// <summary>
    /// Batched-prefill Mamba-2 mixer for <paramref name="n"/> consecutive tokens: in_proj and out_proj as batched
    /// matmuls over the rows of <paramref name="input"/> / <paramref name="output"/> ([n, embDim]), the conv and scan in
    /// token order through <see cref="Mamba2Mix"/> (docs/103 item 12).
    /// </summary>
    private void Mamba2Prefill(int layer, float* input, float* output, int n)
    {
        var c = _m2!;
        float* proj = Alloc(n * c.InProjDim);
        float* xbc = Alloc(n * c.ConvDim);
        float* y = Alloc(n * c.InnerSize);
        try
        {
            MatMulBatchedCached(proj, in _m2In![layer], input, n, c.InProjDim, _embDim);
            Mamba2Mix(layer, proj, n, y, xbc);
            MatMulBatchedCached(output, in _m2Out![layer], y, n, _embDim, c.InnerSize);
        }
        finally
        {
            NativeMemory.Free(proj);
            NativeMemory.Free(xbc);
            NativeMemory.Free(y);
        }
    }

    /// <summary>
    /// The Mamba-2 mixer between the projections, for <paramref name="n"/> tokens in order: causal depthwise conv
    /// + bias + SiLU (ggml_ssm_conv), selective scan (ggml_ssm_scan, scalar-A), D skip, SiLU(z) gate and grouped
    /// RMSNorm. <paramref name="proj"/> rows are in_proj outputs [z | xBC | dt] ([n, InProjDim]); <paramref name="y"/>
    /// receives [n, InnerSize]; <paramref name="xbc"/> is [n, ConvDim] scratch. The conv runs channel by channel and the
    /// scan head by head, each walking the tokens in order, so every (token, channel) and (token, head) sees exactly the
    /// operations of a one-token call: n = 1 is the decode step.
    /// </summary>
    private void Mamba2Mix(int layer, float* proj, int n, float* y, float* xbc)
    {
        var c = _m2!;
        int di = c.InnerSize, convDim = c.ConvDim, k = c.ConvKernel, nh = c.NumHeads, hd = c.HeadDim;
        int ds = c.StateSize, ng = c.NumGroups, inProj = c.InProjDim;

        // Causal depthwise conv over the last k inputs (state holds the k-1 previous), bias, SiLU (ggml_ssm_conv).
        float* w = GetNormWeight(_m2ConvW![layer]);
        float* bias = _m2ConvB![layer].DataPtr is null ? null : GetNormWeight(_m2ConvB[layer]);
        float* st = _m2ConvState![layer];
        int km1 = k - 1;
        void ConvChannel(int ch)
        {
            float* sc = st + ch * km1;
            float* wc = w + ch * k;
            for (int t = 0; t < n; t++)
            {
                float xin = proj[(long)t * inProj + di + ch];
                float sum = 0f;
                for (int j = 0; j < km1; j++) sum += sc[j] * wc[j];
                sum += xin * wc[km1];
                for (int j = 0; j + 1 < km1; j++) sc[j] = sc[j + 1];
                if (km1 > 0) sc[km1 - 1] = xin;
                if (bias is not null) sum += bias[ch];
                xbc[(long)t * convDim + ch] = sum / (1f + MathF.Exp(-sum));
            }
        }
        if (n == 1) for (int ch = 0; ch < convDim; ch++) ConvChannel(ch);
        else Parallel.For(0, convDim, ConvChannel);

        // Selective scan (ggml_ssm_scan, Mamba-2 scalar-A branch).
        float* a = GetNormWeight(_m2A![layer]);
        float* d = GetNormWeight(_m2D![layer]);
        float* dtb = GetNormWeight(_m2DtB![layer]);
        float* s = _m2SsmState![layer];
        int headsPerGroup = nh / ng;
        bool ggmlOrder = Fma.IsSupported && Avx.IsSupported && Sse3.IsSupported && ds % 32 == 0;
        // Heads are independent: spread them over threads (each walks the tokens in order), and vectorise the
        // d_state loop (SIMD over n).
        SimdKernels.ParallelForUncapped(0, nh, h =>
        {
            float ah = a[h], dh = d[h], dtbh = dtb[h];
            int g = h / headsPerGroup;
            for (int t = 0; t < n; t++)
            {
                float* x = xbc + (long)t * convDim;
                float* bg = x + di + g * ds;
                float* cg = x + di + ng * ds + g * ds;
                float* z = proj + (long)t * inProj;
                float* dt = z + di + convDim;
                float* yOut = y + (long)t * di;
                float dtv = dt[h] + dtbh;
                float sp = dtv > 20f ? dtv : MathF.Log(1f + MathF.Exp(dtv));
                float dA = MathF.Exp(sp * ah);
                int vw = System.Numerics.Vector<float>.Count;
                int nVec = ds - ds % vw;
                var vdA = new System.Numerics.Vector<float>(dA);
                for (int e = 0; e < hd; e++)
                {
                    int ii = h * hd + e;
                    float xdt = x[ii] * sp;
                    var vxdt = new System.Numerics.Vector<float>(xdt);
                    float* sRow = s + (long)ii * ds;
                    int nn = 0;
                    float acc;
                    if (ggmlOrder)
                    {
                        // Same arithmetic order as ggml's AVX2 ssm_scan (GGML_F32_STEP 32 = 4 accumulators x 8 lanes):
                        // t0 = s*dA + B*x*dt unfused, sum[j] = fma(t0, C, sum[j]), then GGML_F32_VEC_REDUCE. The
                        // recurrence is sensitive to reduction order at about the 0.3% perplexity level, so match it.
                        var adA = Vector256.Create(dA);
                        var axdt = Vector256.Create(xdt);
                        Vector256<float> s0 = Vector256<float>.Zero, s1 = s0, s2 = s0, s3 = s0;
                        for (; nn + 32 <= ds; nn += 32)
                        {
                            var t0 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + nn), adA), Avx.Multiply(Avx.LoadVector256(bg + nn), axdt));
                            Avx.Store(sRow + nn, t0);
                            s0 = Fma.MultiplyAdd(t0, Avx.LoadVector256(cg + nn), s0);
                            var t1 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + nn + 8), adA), Avx.Multiply(Avx.LoadVector256(bg + nn + 8), axdt));
                            Avx.Store(sRow + nn + 8, t1);
                            s1 = Fma.MultiplyAdd(t1, Avx.LoadVector256(cg + nn + 8), s1);
                            var t2 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + nn + 16), adA), Avx.Multiply(Avx.LoadVector256(bg + nn + 16), axdt));
                            Avx.Store(sRow + nn + 16, t2);
                            s2 = Fma.MultiplyAdd(t2, Avx.LoadVector256(cg + nn + 16), s2);
                            var t3 = Avx.Add(Avx.Multiply(Avx.LoadVector256(sRow + nn + 24), adA), Avx.Multiply(Avx.LoadVector256(bg + nn + 24), axdt));
                            Avx.Store(sRow + nn + 24, t3);
                            s3 = Fma.MultiplyAdd(t3, Avx.LoadVector256(cg + nn + 24), s3);
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
                        for (; nn < nVec; nn += vw)
                        {
                            var v = System.Numerics.Vector.Load(sRow + nn) * vdA + System.Numerics.Vector.Load(bg + nn) * vxdt;
                            System.Numerics.Vector.Store(v, sRow + nn);
                            vacc += v * System.Numerics.Vector.Load(cg + nn);
                        }
                        acc = System.Numerics.Vector.Sum(vacc);
                    }
                    for (; nn < ds; nn++)
                    {
                        float v = sRow[nn] * dA + bg[nn] * xdt;
                        sRow[nn] = v;
                        acc += v * cg[nn];
                    }
                    // y += x * D, then gate with silu(z) (ggml_swiglu_split(z, y)).
                    float yv = acc + x[ii] * dh;
                    float zv = z[ii];
                    yOut[ii] = yv * (zv / (1f + MathF.Exp(-zv)));
                }
            }
        });

        // Grouped RMSNorm over d_inner / n_group channels.
        if (_m2Norm![layer].DataPtr is not null)
        {
            float* nw = GetNormWeight(_m2Norm[layer]);
            int gs = di / ng;
            for (int t = 0; t < n; t++)
            {
                for (int g = 0; g < ng; g++)
                {
                    float* yg = y + (long)t * di + g * gs;
                    double ss = 0;
                    for (int i = 0; i < gs; i++) ss += (double)yg[i] * yg[i];
                    float inv = 1f / MathF.Sqrt((float)(ss / gs) + _hp.RmsNormEps);
                    for (int i = 0; i < gs; i++) yg[i] = yg[i] * inv * nw[g * gs + i];
                }
            }
        }
    }
}
