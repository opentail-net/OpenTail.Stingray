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
        int d = _embDim, k = _scKernel, km1 = k - 1;
        FusedMatVec(_scBcx, _scIn![layer], input, 3 * d, d);
        float* b = _scBcx, c = _scBcx + d, x = _scBcx + 2 * d;
        float* w = GetNormWeight(_scConv![layer]);
        float* st = _scState![layer];
        for (int ch = 0; ch < d; ch++)
        {
            float bx = b[ch] * x[ch];
            float* sc = st + ch * km1;
            float* wc = w + ch * k;
            float sum = 0f;
            for (int j = 0; j < km1; j++) sum += sc[j] * wc[j];
            sum += bx * wc[km1];
            for (int j = 0; j + 1 < km1; j++) sc[j] = sc[j + 1];
            sc[km1 - 1] = bx;
            _scY[ch] = c[ch] * sum;
        }
        FusedMatVec(output, _scOut![layer], _scY, d, d);
        // Mixer output, comparable with llama.cpp's `conv.out_proj` (and `o_proj` for attention layers).
        StageCapture.Record("cpu", layer, StageCapture.Stages.OProj, new ReadOnlySpan<float>(output, d));
    }
}
