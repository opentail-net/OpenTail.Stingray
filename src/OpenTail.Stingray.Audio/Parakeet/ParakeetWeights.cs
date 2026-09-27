
namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Container for NVIDIA NeMo Parakeet FastConformer CTC ASR GGUF weights (`general.architecture
/// = canary_ctc`). Full architecture spec, tensor naming, and BN-fold formula verified against
/// `examples/crispasr/src/canary_ctc.cpp` and `src/core/fastconformer.h` -- see
/// docs/audio-review-progress.md's Parakeet section for the full derivation. Eagerly dequantizes
/// every tensor to float32 in file storage order (same convention as Chatterbox/Kokoro weights).
/// </summary>
public sealed class ParakeetWeights : IDisposable
{
    public GgufModel Model { get; }

    public int NumLayers { get; } = 24;
    public int HiddenDim { get; } = 1024;
    public int NumHeads { get; } = 8;
    public int HeadDim { get; } = 128;
    public int FfDim { get; } = 4096;
    public int ConvKernel { get; } = 9;
    public int VocabSize { get; } = 1024;
    public int BlankTokenId { get; } = 1024;
    public int SubsampleFactor { get; } = 8;
    public int SubsampleChannels { get; } = 256;
    public int NMels { get; } = 80;
    public int NFft { get; } = 512;
    public int WinLength { get; } = 400;
    public int HopLength { get; } = 160;
    public int SampleRate { get; } = 16000;

    public const float LayerNormEps = 1e-5f;

    // --- Subsampling front-end (dw_striding, 8x) ---
    public float[] PreConv0Weight { get; }  // [3,3,1,256] full conv2d, in=1
    public float[] PreConv0Bias { get; }
    public float[] PreConv2Weight { get; }  // [3,3,1,256] depthwise
    public float[] PreConv2Bias { get; }
    public float[] PreConv3Weight { get; }  // [1,1,256,256] pointwise
    public float[] PreConv3Bias { get; }
    public float[] PreConv5Weight { get; }  // [3,3,1,256] depthwise
    public float[] PreConv5Bias { get; }
    public float[] PreConv6Weight { get; }  // [1,1,256,256] pointwise
    public float[] PreConv6Bias { get; }
    public ParakeetLinear PreOut { get; }

    // --- Mel preprocessing (shipped in checkpoint, don't recompute) ---
    public float[] MelFilterbank { get; }   // [257, 80]
    public float[] MelWindow { get; }       // [400]

    // --- CTC head (null for a TDT-only checkpoint such as parakeet-tdt-0.6b-v2) ---
    public ParakeetLinear? Ctc { get; }
    public bool HasCtc => Ctc is not null;

    // --- TDT transducer head (null for a CTC checkpoint). CrispASR GGUF layout (convert-parakeet-to-gguf.py). ---
    public ParakeetTdtWeights? Tdt { get; }

    public ParakeetConformerLayer[] Layers { get; }

    public ParakeetWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Parakeet GGUF model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        // Metadata prefix: "canary_ctc" (CTC checkpoints) or "parakeet" (CrispASR's TDT checkpoints).
        string p = Model.Metadata.ContainsKey("parakeet.d_model") ? "parakeet" : "canary_ctc";
        NumLayers = GetInt($"{p}.n_layers", NumLayers);
        HiddenDim = GetInt($"{p}.d_model", HiddenDim);
        NumHeads = GetInt($"{p}.n_heads", NumHeads);
        HeadDim = GetInt($"{p}.head_dim", HiddenDim / NumHeads);
        FfDim = GetInt($"{p}.ff_dim", FfDim);
        ConvKernel = GetInt($"{p}.conv_kernel", ConvKernel);
        VocabSize = GetInt($"{p}.vocab_size", VocabSize);
        BlankTokenId = GetInt($"{p}.blank_id", BlankTokenId);
        SubsampleFactor = GetInt($"{p}.subsampling_factor", SubsampleFactor);
        SubsampleChannels = GetInt($"{p}.subsampling_channels", SubsampleChannels);
        NMels = GetInt($"{p}.n_mels", NMels);
        NFft = GetInt($"{p}.n_fft", NFft);
        WinLength = GetInt($"{p}.win_length", WinLength);
        HopLength = GetInt($"{p}.hop_length", HopLength);
        SampleRate = GetInt($"{p}.sample_rate", SampleRate);

        PreConv0Weight = GetTensor("encoder.pre.conv.0.weight");
        PreConv0Bias = GetTensor("encoder.pre.conv.0.bias");
        PreConv2Weight = GetTensor("encoder.pre.conv.2.weight");
        PreConv2Bias = GetTensor("encoder.pre.conv.2.bias");
        PreConv3Weight = GetTensor("encoder.pre.conv.3.weight");
        PreConv3Bias = GetTensor("encoder.pre.conv.3.bias");
        PreConv5Weight = GetTensor("encoder.pre.conv.5.weight");
        PreConv5Bias = GetTensor("encoder.pre.conv.5.bias");
        PreConv6Weight = GetTensor("encoder.pre.conv.6.weight");
        PreConv6Bias = GetTensor("encoder.pre.conv.6.bias");
        PreOut = Pack("encoder.pre.out", HiddenDim, -1);

        MelFilterbank = GetTensor("preprocessor.fb");
        MelWindow = GetTensor("preprocessor.window");

        Ctc = Model.FindTensor("ctc.weight") is null ? null : Pack("ctc", VocabSize + 1, HiddenDim);
        if (Model.FindTensor("joint.out.weight") is not null)
            Tdt = new ParakeetTdtWeights(this, p);
        if (Ctc is null && Tdt is null)
            throw new InvalidDataException("Parakeet GGUF has neither a CTC head (ctc.weight) nor a TDT head (joint.out.weight).");

        Layers = new ParakeetConformerLayer[NumLayers];
        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new ParakeetConformerLayer(this, $"encoder.layers.{i}", ConvKernel);
    }

    internal int GetInt(string key, int fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    /// <summary>
    /// Binds <c>{stem}.weight</c> (and <c>{stem}.bias</c> if present) as a batched linear. Quantized tensors stay in
    /// the memory-mapped GGUF and run through <see cref="SimdKernels.MatMulBatched"/>; F32/F16 tensors are packed for
    /// the F32 SGEMM. Dequantizing everything to packed F32 took 2.85 GB for the 378 MB q4_k TDT file (2026-09-27).
    /// <paramref name="inDim"/> -1 infers it from the tensor shape.
    /// </summary>
    internal ParakeetLinear Pack(string stem, int outDim, int inDim)
    {
        var info = Model.FindTensor(stem + ".weight") ?? throw new InvalidDataException($"Parakeet GGUF missing required tensor '{stem}.weight'.");
        if (inDim < 0) inDim = (int)(info.ElementCount / outDim);
        var linear = new ParakeetLinear(this, info, TryGetTensor(stem + ".bias"), outDim, inDim);
        _linears.Add(linear);
        return linear;
    }

    private readonly List<ParakeetLinear> _linears = [];

    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Parakeet GGUF missing required tensor '{name}'.");
        return DequantTensor(Model, info);
    }

    public float[]? TryGetTensor(string name)
    {
        var info = Model.FindTensor(name);
        return info is null ? null : DequantTensor(Model, info.Value);
    }

    private static float[] DequantTensor(GgufModel model, GgufTensorInfo info)
    {
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose()
    {
        foreach (var l in _linears) l.Dispose();
        Model.Dispose();
    }
}

/// <summary>
/// One Conformer block's weights (macaron FFN1 -> rel-pos self-attn -> conv module -> FFN2 ->
/// final LayerNorm), plus the BatchNorm-folded depthwise conv weight/bias. This checkpoint's
/// q/k/v/out/ff linears carry NO bias (canary_ctc arch, confirmed against the converter);
/// `attn.pos` (the relative-position projection) also has no bias.
/// </summary>
public sealed class ParakeetConformerLayer
{
    public float[] NormFf1Weight { get; }
    public float[] NormFf1Bias { get; }
    public ParakeetLinear Ff1Linear1 { get; }
    public ParakeetLinear Ff1Linear2 { get; }

    public float[] NormAttnWeight { get; }
    public float[] NormAttnBias { get; }
    public ParakeetLinear AttnQ { get; }
    public ParakeetLinear AttnK { get; }
    public ParakeetLinear AttnV { get; }
    public ParakeetLinear AttnOut { get; }
    public ParakeetLinear AttnPos { get; }
    public float[] AttnPosBiasU { get; }  // [head_dim, n_heads]
    public float[] AttnPosBiasV { get; }  // [head_dim, n_heads]

    public float[] NormConvWeight { get; }
    public float[] NormConvBias { get; }
    public ParakeetLinear ConvPw1 { get; }
    /// <summary>Depthwise conv weight, BatchNorm-folded at load time (raw checkpoint ships unfused BN tensors -- see docs/audio-review-progress.md).</summary>
    public float[] ConvDwWeight { get; }    // [K, d]
    public float[] ConvDwBias { get; }
    public ParakeetLinear ConvPw2 { get; }

    public float[] NormFf2Weight { get; }
    public float[] NormFf2Bias { get; }
    public ParakeetLinear Ff2Linear1 { get; }
    public ParakeetLinear Ff2Linear2 { get; }

    public float[] NormOutWeight { get; }
    public float[] NormOutBias { get; }

    public ParakeetConformerLayer(ParakeetWeights w, string prefix, int convKernel)
    {
        NormFf1Weight = w.GetTensor($"{prefix}.norm_ff1.weight");
        NormFf1Bias = w.GetTensor($"{prefix}.norm_ff1.bias");
        Ff1Linear1 = w.Pack($"{prefix}.ff1.linear1", w.FfDim, w.HiddenDim);
        Ff1Linear2 = w.Pack($"{prefix}.ff1.linear2", w.HiddenDim, w.FfDim);

        NormAttnWeight = w.GetTensor($"{prefix}.norm_attn.weight");
        NormAttnBias = w.GetTensor($"{prefix}.norm_attn.bias");
        AttnQ = w.Pack($"{prefix}.attn.q", w.HiddenDim, w.HiddenDim);
        AttnK = w.Pack($"{prefix}.attn.k", w.HiddenDim, w.HiddenDim);
        AttnV = w.Pack($"{prefix}.attn.v", w.HiddenDim, w.HiddenDim);
        AttnOut = w.Pack($"{prefix}.attn.out", w.HiddenDim, w.HiddenDim);
        AttnPos = w.Pack($"{prefix}.attn.pos", w.HiddenDim, w.HiddenDim);
        AttnPosBiasU = w.GetTensor($"{prefix}.attn.pos_bias_u");
        AttnPosBiasV = w.GetTensor($"{prefix}.attn.pos_bias_v");

        NormConvWeight = w.GetTensor($"{prefix}.norm_conv.weight");
        NormConvBias = w.GetTensor($"{prefix}.norm_conv.bias");
        ConvPw1 = w.Pack($"{prefix}.conv.pw1", 2 * w.HiddenDim, w.HiddenDim);
        ConvPw2 = w.Pack($"{prefix}.conv.pw2", w.HiddenDim, w.HiddenDim);

        var dwWeightRaw = w.GetTensor($"{prefix}.conv.dw.weight");  // [K, 1, d] storage order
        var dwBiasRaw = w.GetTensor($"{prefix}.conv.dw.bias");
        var bnWeight = w.GetTensor($"{prefix}.conv.bn.weight");
        var bnBias = w.GetTensor($"{prefix}.conv.bn.bias");
        var bnMean = w.GetTensor($"{prefix}.conv.bn.running_mean");
        var bnVar = w.GetTensor($"{prefix}.conv.bn.running_var");
        (ConvDwWeight, ConvDwBias) = FoldBatchNorm(dwWeightRaw, dwBiasRaw, bnWeight, bnBias, bnMean, bnVar, w.HiddenDim, convKernel);

        NormFf2Weight = w.GetTensor($"{prefix}.norm_ff2.weight");
        NormFf2Bias = w.GetTensor($"{prefix}.norm_ff2.bias");
        Ff2Linear1 = w.Pack($"{prefix}.ff2.linear1", w.FfDim, w.HiddenDim);
        Ff2Linear2 = w.Pack($"{prefix}.ff2.linear2", w.HiddenDim, w.FfDim);

        NormOutWeight = w.GetTensor($"{prefix}.norm_out.weight");
        NormOutBias = w.GetTensor($"{prefix}.norm_out.bias");
    }

    /// <summary>
    /// Folds BatchNorm1d (applied after the depthwise conv, before SiLU, in the original NeMo
    /// module) into the depthwise conv's weight/bias, matching `canary_ctc.cpp`'s
    /// `cc_fold_batchnorm` exactly: `s[c] = bn_w[c] / sqrt(bn_var[c] + eps)`,
    /// `w_folded[k,c] = w[k,c] * s[c]`, `b_folded[c] = s[c]*orig_b[c] - bn_mean[c]*s[c] + bn_b[c]`.
    /// </summary>
    private static (float[] weight, float[] bias) FoldBatchNorm(
        float[] dwWeight, float[] dwBias, float[] bnWeight, float[] bnBias, float[] bnMean, float[] bnVar,
        int channels, int kernel)
    {
        const float eps = 1e-5f; // BN eps, per canary_ctc.cpp's cc_fold_batchnorm (numerically same as LayerNormEps, tracked separately since they're conceptually distinct)
        var s = new float[channels];
        for (int c = 0; c < channels; c++)
            s[c] = bnWeight[c] / MathF.Sqrt(bnVar[c] + eps);

        var foldedWeight = new float[dwWeight.Length];
        // GGML ne=[K,1,d] (K fastest-varying) -> flat index = k + c*K, confirmed against
        // canary_ctc.cpp's cc_fold_batchnorm: `w_f32[ki + c * K] *= s[c]` (an earlier version
        // of this port assumed k*channels+c, the wrong order -- caught by reading the actual
        // fold loop's indexing instead of guessing from the GGUF dims listing alone).
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < kernel; k++)
                foldedWeight[c * kernel + k] = dwWeight[c * kernel + k] * s[c];

        var foldedBias = new float[channels];
        for (int c = 0; c < channels; c++)
            foldedBias[c] = s[c] * dwBias[c] - bnMean[c] * s[c] + bnBias[c];

        return (foldedWeight, foldedBias);
    }
}

/// <summary>
/// TDT transducer head: a 2-layer LSTM prediction network over the last emitted token and a joint network
/// <c>out(relu(enc_proj(enc_t) + pred_proj(g)))</c> whose logits are [vocab + blank | durations]
/// (CrispASR src/parakeet.cpp, NeMo RNNTJoint with ReLU). Weights are row-major [out, in].
/// </summary>
public sealed class ParakeetTdtWeights
{
    public int PredHidden { get; }
    public int JointHidden { get; }
    public int PredLayers { get; }
    public int[] Durations { get; }

    public float[] Embed { get; }                 // [vocab + 1, PredHidden]; the blank row is the SOS input
    public float[][] LstmWih { get; }             // per layer [4 * PredHidden, in], gate order i, f, g, o
    public float[][] LstmWhh { get; }             // per layer [4 * PredHidden, PredHidden]
    public float[][] LstmBih { get; }
    public float[][] LstmBhh { get; }
    public float[] JointEncWeight { get; }        // [JointHidden, d_model]
    public float[] JointEncBias { get; }
    public float[] JointPredWeight { get; }       // [JointHidden, PredHidden]
    public float[] JointPredBias { get; }
    public float[] JointOutWeight { get; }        // [vocab + 1 + durations, JointHidden]
    public float[] JointOutBias { get; }
    public int OutputDim => JointOutBias.Length;

    internal ParakeetTdtWeights(ParakeetWeights w, string prefix)
    {
        PredHidden = w.GetInt($"{prefix}.pred_hidden", 640);
        JointHidden = w.GetInt($"{prefix}.joint_hidden", 640);
        PredLayers = w.GetInt($"{prefix}.pred_layers", 2);
        Durations = w.Model.Metadata.TryGetValue($"{prefix}.tdt_durations", out var dv) && dv is System.Collections.IEnumerable e
            ? e.Cast<object>().Select(Convert.ToInt32).ToArray()
            : [0, 1, 2, 3, 4];

        Embed = w.GetTensor("decoder.embed.weight");
        LstmWih = new float[PredLayers][];
        LstmWhh = new float[PredLayers][];
        LstmBih = new float[PredLayers][];
        LstmBhh = new float[PredLayers][];
        for (int l = 0; l < PredLayers; l++)
        {
            LstmWih[l] = w.GetTensor($"decoder.lstm.{l}.w_ih");
            LstmWhh[l] = w.GetTensor($"decoder.lstm.{l}.w_hh");
            LstmBih[l] = w.GetTensor($"decoder.lstm.{l}.b_ih");
            LstmBhh[l] = w.GetTensor($"decoder.lstm.{l}.b_hh");
        }
        JointEncWeight = w.GetTensor("joint.enc.weight");
        JointEncBias = w.GetTensor("joint.enc.bias");
        JointPredWeight = w.GetTensor("joint.pred.weight");
        JointPredBias = w.GetTensor("joint.pred.bias");
        JointOutWeight = w.GetTensor("joint.out.weight");
        JointOutBias = w.GetTensor("joint.out.bias");
    }
}

/// <summary>
/// One encoder linear over a batch of frames: <c>output[m, OutDim] = input[m, InDim] @ W^T (+ bias)</c>. Quantized
/// weights are read in place from the memory-mapped GGUF (the frames of one utterance are quantized together, like
/// an LLM prefill, so <c>allowQ8</c> is sound); F32/F16 weights use the packed F32 SGEMM.
/// </summary>
public sealed unsafe class ParakeetLinear : IDisposable
{
    private readonly byte* _data;
    private readonly DType _dtype;
    // Q4_K weights repacked into 8-row groups for the batched Q4Kx8 kernel (same size as the Q4_K data), built once.
    private byte* _q4kx8;
    private readonly PackedLinearF32? _packed;
    private readonly float[]? _bias;

    public int OutDim { get; }
    public int InDim { get; }

    internal ParakeetLinear(ParakeetWeights w, GgufTensorInfo info, float[]? bias, int outDim, int inDim)
    {
        (OutDim, InDim, _bias) = (outDim, inDim, bias);
        // Q8_0 (the conv pointwise layers) is also packed as F32: the int8 path has no batched Q8_0 kernel and ran
        // ~0.2 s slower per 14 s clip (CTC encoder 1.39 s vs 1.19 s, 2026-09-27) for ~290 MB saved; a batched Q8_0
        // 4-input wrapper measured no gain.
        if (info.DType is DType.Float32 or DType.Float16 or DType.BFloat16 or DType.Q8_0)
            _packed = new PackedLinearF32(w.GetTensor(info.Name), bias, outDim, inDim);
        else
        {
            _dtype = info.DType;
            _data = w.Model.GetTensorDataPtr(info);
            if (_dtype == DType.Q4_K && SimdKernels.CanRepackQ4Kx8(outDim, inDim))
            {
                _q4kx8 = (byte*)NativeMemory.Alloc((nuint)SimdKernels.Q4Kx8PackedBytes(outDim, inDim));
                SimdKernels.RepackQ4KMatrix(_data, _q4kx8, outDim, inDim);
            }
        }
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output, int m)
    {
        if (_packed is not null) { _packed.Forward(input, output, m); return; }
        fixed (float* xp = input, yp = output)
        {
            if (_q4kx8 == null || !SimdKernels.TryMatMulBatchedQ4Kx8(yp, _q4kx8, xp, m, OutDim, InDim))
                SimdKernels.MatMulBatched(yp, _data, xp, m, OutDim, InDim, _dtype, allowQ8: true);
        }
        if (_bias is not null)
            for (int r = 0; r < m; r++)
            {
                var row = output.Slice(r * OutDim, OutDim);
                TensorPrimitives.Add(row, _bias, row);
            }
    }

    public void Dispose()
    {
        _packed?.Dispose();
        if (_q4kx8 != null) { NativeMemory.Free(_q4kx8); _q4kx8 = null; }
    }
}
