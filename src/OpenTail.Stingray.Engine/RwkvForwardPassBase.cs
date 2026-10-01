using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// What RWKV-6 and RWKV-7 share (llama.cpp <c>llm_build_rwkv6</c> / <c>llm_build_rwkv7</c>): a
/// LayerNorm'd token embedding, per-layer recurrent state (the previous token's attn_norm and
/// ffn_norm for token shift, and the WKV matrix state), a final LayerNorm and the LM head, run in
/// chunks — one token for decode, up to <see cref="MaxChunk"/> for prefill, as llama.cpp runs a
/// ubatch. Every projection is one batched matmul over a chunk's rows; only the WKV recurrence
/// steps through tokens in order. A derived class implements one layer.
/// </summary>
public abstract unsafe class RwkvForwardPassBase : IForwardPass
{
    protected const int MaxChunk = 256;
    protected const float GroupNormEps = 64e-5f;    // rwkv{6,7}-base.cpp: ggml_norm(ctx0, cur, 64e-5f)

    protected readonly int D, NumLayer, HeadSize, Heads, FfnDim;
    protected readonly float NormEps;
    private readonly int _vocab;
    private readonly DeepSeek4TensorRef _tokEmbd, _tokNormW, _tokNormB, _outNormW, _outNormB, _output;

    /// <summary>Per layer: previous token's attn_norm / ffn_norm, and the heads × hs × hs WKV state.</summary>
    protected readonly float[][] AttShift, FfnShift, WkvState;
    private int _length, _cap;
    private float[] _x = [];
    private readonly float[] _logits, _lastNorm;

    protected RwkvForwardPassBase(GgufModel model, string arch)
    {
        int Int(string key) => Convert.ToInt32(model.Metadata[$"{arch}.{key}"]);
        D = Int("embedding_length");
        NumLayer = Int("block_count");
        HeadSize = Int("wkv.head_size");
        FfnDim = Int("feed_forward_length");
        Heads = D / HeadSize;
        NormEps = model.Metadata.TryGetValue($"{arch}.attention.layer_norm_epsilon", out var e) ? Convert.ToSingle(e) : 1e-5f;
        _vocab = model.Metadata.TryGetValue($"{arch}.vocab_size", out var v)
            ? Convert.ToInt32(v)
            : ((object[])model.Metadata["tokenizer.ggml.tokens"]).Length;

        _tokEmbd = Req(model, "token_embd.weight");
        _tokNormW = Req(model, "token_embd_norm.weight");
        _tokNormB = Req(model, "token_embd_norm.bias");
        _outNormW = Req(model, "output_norm.weight");
        _outNormB = Req(model, "output_norm.bias");
        _output = Req(model, "output.weight");

        AttShift = new float[NumLayer][];
        FfnShift = new float[NumLayer][];
        WkvState = new float[NumLayer][];
        for (int il = 0; il < NumLayer; il++)
        {
            AttShift[il] = new float[D];
            FfnShift[il] = new float[D];
            WkvState[il] = new float[Heads * HeadSize * HeadSize];
        }
        _logits = new float[_vocab];
        _lastNorm = new float[D];
    }

    /// <summary>True for the architectures that run on an RWKV pass (rwkv6, rwkv7).</summary>
    public static bool IsRwkv(string? arch) => arch is "rwkv6" or "rwkv7";

    /// <summary>The RWKV pass for <paramref name="model"/>'s architecture.</summary>
    public static RwkvForwardPassBase Create(GgufModel model) =>
        Convert.ToString(model.Metadata["general.architecture"]) switch
        {
            "rwkv6" => new Rwkv6ForwardPass(model),
            "rwkv7" => new Rwkv7ForwardPass(model),
            var a => throw new NotSupportedException($"No RWKV forward pass for architecture '{a}'."),
        };

    public int VocabSize => _vocab;
    public int MaxSeqLen => int.MaxValue;

    protected static DeepSeek4TensorRef Req(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidOperationException($"Missing required RWKV tensor: {name}");
        return new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info));
    }

    protected static DeepSeek4TensorRef? Opt(GgufModel model, string name) =>
        model.FindTensor(name) is { } info ? new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info)) : null;

    /// <summary><c>output[n × rows] = input[n × cols] · Wᵀ</c>. Rows are positions of one prompt,
    /// so the int8 prefill tier (when enabled) is a legitimate choice.</summary>
    protected static void MatMul(float* output, in DeepSeek4TensorRef w, float* input, int n)
    {
        int cols = (int)w.Info.Dimensions[0], rows = (int)w.Info.Dimensions[1];
        if (n == 1) SimdKernels.MatVec(output, w.DataPtr, input, rows, cols, w.DType);
        else SimdKernels.MatMulBatched(output, w.DataPtr, input, n, rows, cols, w.DType, allowQ8: true);
    }

    protected static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>LayerNorm (weight + bias) of each of <paramref name="n"/> rows.</summary>
    protected void LayerNormRows(float* dst, float* src, in DeepSeek4TensorRef weight, in DeepSeek4TensorRef bias, int n)
    {
        for (int t = 0; t < n; t++)
            SimdKernels.LayerNorm(dst + t * D, src + t * D, (float*)weight.DataPtr, (float*)bias.DataPtr, D, NormEps);
    }

    /// <summary>
    /// Token-shift lerp: <c>dst[t] = (prev[t] - cur[t]) * lerp + cur[t]</c>, where <c>prev[0]</c> is
    /// <paramref name="shift"/> (the last token of the previous chunk) and <c>prev[t]</c> is
    /// <c>cur[t-1]</c>. <paramref name="lerpRows"/>, when non-null, adds a per-row term to the lerp
    /// (RWKV-6's data-dependent mix).
    /// </summary>
    protected void TokenShiftLerp(float* dst, float* cur, float* shift, float* lerp, float* lerpRows, int n)
    {
        for (int t = 0; t < n; t++)
        {
            float* c = cur + t * D, prev = t == 0 ? shift : cur + (t - 1) * D, o = dst + t * D;
            if (lerpRows is null)
                for (int i = 0; i < D; i++) o[i] = (prev[i] - c[i]) * lerp[i] + c[i];
            else
            {
                float* lr = lerpRows + t * D;
                for (int i = 0; i < D; i++) o[i] = (prev[i] - c[i]) * (lerp[i] + lr[i]) + c[i];
            }
        }
    }

    /// <summary>Copies a chunk's last row into a layer's token-shift state.</summary>
    protected void SaveShift(float* rows, int n, float[] shift)
    {
        fixed (float* s = shift)
            Buffer.MemoryCopy(rows + (n - 1) * D, s, D * sizeof(float), D * sizeof(float));
    }

    /// <summary>Per-head group norm of one row (eps 64e-5), then the ln weight and bias.</summary>
    protected void GroupNorm(float* row, in DeepSeek4TensorRef weight, in DeepSeek4TensorRef bias)
    {
        float* w = (float*)weight.DataPtr, b = (float*)bias.DataPtr;
        for (int h = 0; h < Heads; h++)
        {
            int o = h * HeadSize;
            SimdKernels.LayerNorm(row + o, row + o, w + o, b + o, HeadSize, GroupNormEps);
        }
    }

    /// <summary>Grow the derived class's chunk scratch to at least <paramref name="n"/> rows.</summary>
    protected abstract void EnsureScratch(int n);

    /// <summary>Run layer <paramref name="il"/> on <paramref name="n"/> rows of <paramref name="x"/> in place.</summary>
    protected abstract void RunLayer(int il, float* x, int n);

    public ReadOnlySpan<float> Forward(int token, int position) => Prefill([token], position);

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        if (startPos != _length)
            throw new NotSupportedException($"{GetType().Name} is recurrent: expected position {_length}, got {startPos}.");
        if (tokens.Count == 0) return _logits;
        for (int s = 0; s < tokens.Count; s += MaxChunk)
            RunChunk(tokens, s, Math.Min(MaxChunk, tokens.Count - s));
        fixed (float* last = _lastNorm, logits = _logits)
            MatMul(logits, _output, last, 1);
        return _logits;
    }

    private void RunChunk(IReadOnlyList<int> tokens, int start, int n)
    {
        if (n > _cap)
        {
            _x = new float[n * D];
            EnsureScratch(n);
            _cap = n;
        }
        int bytesPerRow = D / DTypeInfo.BlockSize(_tokEmbd.DType) * DTypeInfo.BytesPerBlock(_tokEmbd.DType);
        fixed (float* x = _x, last = _lastNorm)
        {
            for (int t = 0; t < n; t++)
                SimdKernels.DequantRow(_tokEmbd.DataPtr + (long)tokens[start + t] * bytesPerRow, x + t * D, D, _tokEmbd.DType);
            LayerNormRows(x, x, _tokNormW, _tokNormB, n);
            for (int il = 0; il < NumLayer; il++)
                RunLayer(il, x, n);
            SimdKernels.LayerNorm(last, x + (n - 1) * D, (float*)_outNormW.DataPtr, (float*)_outNormB.DataPtr, D, NormEps);
        }
        _length += n;
    }

    /// <summary>A recurrent state cannot be rewound: only a full reset or a no-op is possible.</summary>
    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        if (length != _length)
            throw new NotSupportedException($"{GetType().Name} is recurrent and cannot rewind from {_length} to {length}.");
    }

    public void ResetCache()
    {
        foreach (var s in AttShift) Array.Clear(s);
        foreach (var s in FfnShift) Array.Clear(s);
        foreach (var s in WkvState) Array.Clear(s);
        _length = 0;
    }

    public void Dispose() { }
}
