using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). Gated short-convolution mixer layers for
// Liquid LFM2 (`lfm2`): a layer whose attention.head_count_kv entry is 0 runs this mixer in place of attention,
// then shares the usual residual + FFN path. Reference: llama.cpp src/models/lfm2.cpp build_shortconv_block.
public sealed unsafe partial class ForwardPass
{
    private int _scKernel;
    private TensorRef[]? _scIn, _scOut, _scConv;
    // Per-layer conv state (null for attention layers): [embDim][kernel - 1], oldest input first per channel.
    private float*[]? _scState;
    private float* _scBcx, _scY;

    private bool IsShortConvLayer(int layer) => _hp.IsShortConvLayer is { } sc && sc[layer];

    /// <summary>True when any layer keeps recurrent state (Mamba-2 or short conv): prefill runs token by token and
    /// the cache cannot be partially rewound.</summary>
    private bool HasRecurrentState => _m2 is not null || _scIn is not null;

    /// <summary>Batched prefill for models with Mamba-2 / short-conv layers (docs/103 item 12): projections batched over
    /// the chunk, conv and scan in token order. <c>STINGRAY_RECURRENT_BATCHED_PREFILL=0</c> restores the per-token prefill.</summary>
    public static bool RecurrentBatchedPrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_RECURRENT_BATCHED_PREFILL") != "0";

    /// <summary>
    /// LFM2-MoE batched recurrent trunk. The batched/per-token drift seen on the real Q4_K_M checkpoint
    /// (2026-10-01) was the 64-wide flash attention's online softmax at 256+ tokens, which is not
    /// bit-identical by design; with <c>STINGRAY_PREFILL_ATTN_FLASH64=0</c> the batched trunk matches
    /// token-by-token bit-for-bit (PPL [512,1024): 7.3425 both vs 7.2882 with flash, llama.cpp
    /// 7.9130 ± 1.09). On by default (~2.3x faster prefill); set
    /// <c>STINGRAY_LFM2_MOE_BATCHED_PREFILL=0</c> for the exact per-token trunk.
    /// </summary>
    public static bool Lfm2MoeBatchedPrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_LFM2_MOE_BATCHED_PREFILL") != "0";

    /// <summary>True when the batched trunk can run this model: no recurrent state, or the batched recurrent path is on
    /// (not with the TurboQuant cache, whose sibling trunk has no recurrent layers).</summary>
    private bool RecurrentBatchedPrefillApplies => !HasRecurrentState ||
        (RecurrentBatchedPrefillEnabled
            && (!(_hp.IsMoE && _hp.IsShortConvLayer is not null) || Lfm2MoeBatchedPrefillEnabled)
            && _tqKvCache is null);

    private void InitShortConv(int numLayers)
    {
        if (_hp.IsShortConvLayer is null || _hp.ShortConvKernel < 2) return;
        _scKernel = _hp.ShortConvKernel;
        _scIn = new TensorRef[numLayers];
        _scOut = new TensorRef[numLayers];
        _scConv = new TensorRef[numLayers];
        _scState = new float*[numLayers];
        int state = _embDim * (_scKernel - 1);
        for (int i = 0; i < numLayers; i++)
        {
            if (!IsShortConvLayer(i)) continue;
            _scIn[i] = ResolveTensor($"blk.{i}.shortconv.in_proj.weight");
            _scOut[i] = ResolveTensor($"blk.{i}.shortconv.out_proj.weight");
            _scConv[i] = ResolveTensor($"blk.{i}.shortconv.conv.weight");
            _scState[i] = Alloc(state);
        }
        _scBcx = Alloc(3 * _embDim);
        _scY = Alloc(_embDim);
        EnsureZeroKv();
        ResetShortConvState();
    }

    private void ResetShortConvState()
    {
        if (_scState is null) return;
        int state = _embDim * (_scKernel - 1);
        foreach (var s in _scState)
            if (s is not null) new Span<float>(s, state).Clear();
    }

    /// <summary>One token through the short-conv mixer of <paramref name="layer"/>: in_proj -> [b | c | x],
    /// causal depthwise conv of b*x over the last <c>kernel</c> steps (no bias, no activation; ggml_ssm_conv),
    /// gate by c, out_proj. <paramref name="output"/> receives the mixer output before the residual add.</summary>
    private void ShortConvStep(int layer, float* input, float* output)
    {
        int d = _embDim;
        FusedMatVec(_scBcx, _scIn![layer], input, 3 * d, d);
        ShortConvMix(layer, _scBcx, 1, _scY);
        FusedMatVec(output, _scOut![layer], _scY, d, d);
        // Mixer output, comparable with llama.cpp's `conv.out_proj` (and `o_proj` for attention layers).
        StageCapture.Record("cpu", layer, StageCapture.Stages.OProj, new ReadOnlySpan<float>(output, d));
    }

    /// <summary>Batched-prefill short-conv mixer for <paramref name="n"/> consecutive tokens: in_proj and out_proj as
    /// batched matmuls over [n, embDim] rows, the conv in token order through <see cref="ShortConvMix"/> (docs/103
    /// item 12).</summary>
    private void ShortConvPrefill(int layer, float* input, float* output, int n)
    {
        int d = _embDim;
        float* bcx = Alloc(n * 3 * d);
        float* y = Alloc(n * d);
        try
        {
            MatMulBatchedCached(bcx, in _scIn![layer], input, n, 3 * d, d);
            ShortConvMix(layer, bcx, n, y);
            MatMulBatchedCached(output, in _scOut![layer], y, n, d, d);
        }
        finally
        {
            NativeMemory.Free(bcx);
            NativeMemory.Free(y);
        }
    }

    /// <summary>The short-conv mixer between the projections for <paramref name="n"/> tokens in order:
    /// <paramref name="bcx"/> rows are [b | c | x] ([n, 3*embDim]), <paramref name="y"/> receives c * conv(b*x)
    /// ([n, embDim]). Each channel walks the tokens in order with exactly the one-token arithmetic (n = 1 is the decode
    /// step); channels are independent.</summary>
    private void ShortConvMix(int layer, float* bcx, int n, float* y)
    {
        int d = _embDim, k = _scKernel, km1 = k - 1;
        float* w = GetNormWeight(_scConv![layer]);
        float* st = _scState![layer];
        void Channel(int ch)
        {
            float* sc = st + ch * km1;
            float* wc = w + ch * k;
            for (int t = 0; t < n; t++)
            {
                float* row = bcx + (long)t * 3 * d;
                float bx = row[ch] * row[2 * d + ch];
                float sum = 0f;
                for (int j = 0; j < km1; j++) sum += sc[j] * wc[j];
                sum += bx * wc[km1];
                for (int j = 0; j + 1 < km1; j++) sc[j] = sc[j + 1];
                sc[km1 - 1] = bx;
                y[(long)t * d + ch] = row[d + ch] * sum;
            }
        }
        if (n == 1) for (int ch = 0; ch < d; ch++) Channel(ch);
        else Parallel.For(0, d, Channel);
    }
}
