
namespace OpenTail.Stingray.Audio.NemotronAsr;

/// <summary>
/// Container for NVIDIA Nemotron 3.5 ASR (streaming FastConformer-RNNT) GGUF weights
/// (`general.architecture = asr`, `asr.head_type = rnnt`). Confirmed directly from the real
/// downloaded checkpoint's own metadata/tensor listing (`nvidia/nemotron-3.5-asr-streaming-0.6b`,
/// GGUF Q8_0 quant) -- NOT a guess: 24 Conformer layers with real Transformer-XL-style relative
/// positional self-attention (`self_attn.linear_pos`/`pos_bias_u`/`pos_bias_v`, matching
/// `encoder.layers.{i}.self_attn.*` exactly), `dw_striding` 8x subsampling front-end
/// (`pre_encode.conv.{0,2,3,5,6}` + `pre_encode.out`), a real 2-layer LSTM prediction network
/// (`decoder.prediction.dec_rnn.lstm.{ih,hh}_l{0,1}`), and a joint network
/// (`joint.{enc,pred,joint_net.2}`). Unlike Parakeet/canary_ctc's depthwise-conv module (real
/// BatchNorm1d, folded at load time), this checkpoint's conv module normalization is a real
/// LayerNorm (`asr.encoder.conv_norm = layer_norm` in the checkpoint's own metadata, and
/// `conv.batch_norm.{weight,bias}` has no accompanying running_mean/running_var tensors --
/// confirming it's a plain affine LayerNorm despite the "batch_norm" tensor name, not a real
/// BatchNorm requiring statistics folding). See docs/audio-review-progress.md's Nemotron ASR
/// section for the full architecture derivation and reuse assessment against Parakeet.
/// </summary>
public sealed class NemotronAsrWeights : IDisposable
{
    public GgufModel Model { get; }

    public int NumLayers { get; }
    public int HiddenDim { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int FfDim { get; }
    public int ConvKernel { get; }
    public int FeatIn { get; }
    public int SubsampleFactor { get; }
    public int SubsampleChannels { get; }

    public int VocabSize { get; }
    public int BlankTokenId { get; }
    public int PredHidden { get; }
    public int PredEmbedDim { get; }
    public int PredNumLayers { get; }
    public int JointDim { get; }
    public int MaxSymbolsPerStep { get; }
    public int NumPrompts { get; }
    public int PromptIntermediateSize { get; }

    public int NMels { get; }
    public int NFft { get; }
    public float SampleRate { get; }
    public float WindowSizeSeconds { get; }
    public float WindowStrideSeconds { get; }
    public float Preemph { get; }

    public const float LayerNormEps = 1e-5f;

    // --- Subsampling front-end (dw_striding, 8x): 3 stride-2 stages ---
    public float[] PreConv0Weight { get; } // [3,3,1,256] full conv2d, in=1
    public float[] PreConv0Bias { get; }
    public float[] PreConv2Weight { get; } // [3,3,1,256] depthwise
    public float[] PreConv2Bias { get; }
    public float[] PreConv3Weight { get; } // [1,1,256,256] pointwise
    public float[] PreConv3Bias { get; }
    public float[] PreConv5Weight { get; } // [3,3,1,256] depthwise
    public float[] PreConv5Bias { get; }
    public float[] PreConv6Weight { get; } // [1,1,256,256] pointwise
    public float[] PreConv6Bias { get; }
    public float[] PreOutWeight { get; }   // [4352, 1024]
    public float[] PreOutBias { get; }

    // --- Relative positional encoding table (precomputed, shipped in checkpoint) ---
    public float[] PosEncTable { get; } // [1024, 9999]

    public float[] MelFilterbank { get; } // [257, 128]

    public NemotronAsrConformerLayer[] Layers { get; }

    // --- RNNT prediction network (2-layer LSTM) ---
    public float[] PredEmbedWeight { get; } // [640, vocab] (real HF row-major -> GGUF dims reversed)
    public NemotronAsrLstmLayer PredLstm0 { get; }
    public NemotronAsrLstmLayer PredLstm1 { get; }

    // --- Joint network ---
    public float[] JointEncWeight { get; } // [1024, 640]
    public float[] JointEncBias { get; }
    public float[] JointPredWeight { get; } // [640, 640]
    public float[] JointPredBias { get; }
    public float[] JointNet2Weight { get; } // [640, vocab]
    public float[] JointNet2Bias { get; }

    // --- Prompt (language/task) conditioning MLP, applied after the Conformer stack ---
    public float[] PromptKernel0Weight { get; } // [1152, 2048] = [hidden+num_prompts, prompt_intermediate_size]
    public float[] PromptKernel0Bias { get; }
    public float[] PromptKernel2Weight { get; } // [2048, 1024]
    public float[] PromptKernel2Bias { get; }

    public NemotronAsrWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Nemotron ASR GGUF model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        NumLayers = GetInt("asr.encoder.n_layers", 24);
        HiddenDim = GetInt("asr.encoder.d_model", 1024);
        NumHeads = GetInt("asr.encoder.n_heads", 8);
        HeadDim = HiddenDim / NumHeads;
        FfDim = GetInt("asr.encoder.d_ff", 4096);
        ConvKernel = GetInt("asr.encoder.conv_kernel_size", 9);
        FeatIn = GetInt("asr.encoder.feat_in", 128);
        SubsampleFactor = GetInt("asr.encoder.subsampling_factor", 8);
        SubsampleChannels = GetInt("asr.encoder.subsampling_conv_channels", 256);

        VocabSize = GetInt("asr.rnnt.vocab_size", 13088);
        BlankTokenId = GetInt("asr.rnnt.blank_id", VocabSize - 1);
        PredHidden = GetInt("asr.rnnt.pred_hidden", 640);
        PredEmbedDim = GetInt("asr.rnnt.pred_embed_dim", 640);
        PredNumLayers = GetInt("asr.rnnt.pred_num_layers", 2);
        JointDim = GetInt("asr.rnnt.joint_dim", 640);
        MaxSymbolsPerStep = GetInt("asr.rnnt.max_symbols_per_step", 10);
        NumPrompts = GetInt("asr.rnnt.num_prompts", 128);
        PromptIntermediateSize = GetInt("asr.rnnt.prompt_intermediate_size", 2048);

        NMels = FeatIn;
        NFft = GetInt("asr.preprocessor.n_fft", 512);
        SampleRate = GetFloat("asr.preprocessor.sample_rate", 16000f);
        WindowSizeSeconds = GetFloat("asr.preprocessor.window_size", 0.025f);
        WindowStrideSeconds = GetFloat("asr.preprocessor.window_stride", 0.01f);
        Preemph = GetFloat("asr.preprocessor.preemph", 0.97f);

        PreConv0Weight = GetTensor("encoder.pre_encode.conv.0.weight");
        PreConv0Bias = GetTensor("encoder.pre_encode.conv.0.bias");
        PreConv2Weight = GetTensor("encoder.pre_encode.conv.2.weight");
        PreConv2Bias = GetTensor("encoder.pre_encode.conv.2.bias");
        PreConv3Weight = GetTensor("encoder.pre_encode.conv.3.weight");
        PreConv3Bias = GetTensor("encoder.pre_encode.conv.3.bias");
        PreConv5Weight = GetTensor("encoder.pre_encode.conv.5.weight");
        PreConv5Bias = GetTensor("encoder.pre_encode.conv.5.bias");
        PreConv6Weight = GetTensor("encoder.pre_encode.conv.6.weight");
        PreConv6Bias = GetTensor("encoder.pre_encode.conv.6.bias");
        PreOutWeight = GetTensor("encoder.pre_encode.out.weight");
        PreOutBias = GetTensor("encoder.pre_encode.out.bias");

        PosEncTable = GetTensor("encoder.pos_enc.pe");
        MelFilterbank = GetTensor("preprocessor.fb");

        Layers = new NemotronAsrConformerLayer[NumLayers];
        for (int i = 0; i < NumLayers; i++)
            Layers[i] = new NemotronAsrConformerLayer(this, $"encoder.layers.{i}");

        PredEmbedWeight = GetTensor("decoder.prediction.embed.weight");
        PredLstm0 = new NemotronAsrLstmLayer(this, "decoder.prediction.dec_rnn.lstm", 0);
        PredLstm1 = new NemotronAsrLstmLayer(this, "decoder.prediction.dec_rnn.lstm", 1);

        JointEncWeight = GetTensor("joint.enc.weight");
        JointEncBias = GetTensor("joint.enc.bias");
        JointPredWeight = GetTensor("joint.pred.weight");
        JointPredBias = GetTensor("joint.pred.bias");
        JointNet2Weight = GetTensor("joint.joint_net.2.weight");
        JointNet2Bias = GetTensor("joint.joint_net.2.bias");

        PromptKernel0Weight = GetTensor("prompt_kernel.0.weight");
        PromptKernel0Bias = GetTensor("prompt_kernel.0.bias");
        PromptKernel2Weight = GetTensor("prompt_kernel.2.weight");
        PromptKernel2Bias = GetTensor("prompt_kernel.2.bias");
    }

    private int GetInt(string key, int fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    private float GetFloat(string key, float fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Nemotron ASR GGUF missing required tensor '{name}'.");
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
        Model.Dispose();
    }
}

/// <summary>
/// One Conformer block: macaron FFN1 -> Transformer-XL relative-position self-attention ->
/// depthwise-conv+GLU conv module (real LayerNorm, not BatchNorm -- see
/// <see cref="NemotronAsrWeights"/>'s doc comment) -> macaron FFN2 -> final LayerNorm. All
/// q/k/v/out/pos linears carry NO bias (confirmed: no `*.self_attn.*.bias` tensors in the real
/// checkpoint).
/// </summary>
public sealed class NemotronAsrConformerLayer
{
    public float[] NormFf1Weight { get; }
    public float[] NormFf1Bias { get; }
    public float[] Ff1Linear1Weight { get; }
    public float[]? Ff1Linear1Bias { get; }
    public float[] Ff1Linear2Weight { get; }
    public float[]? Ff1Linear2Bias { get; }

    public float[] NormAttnWeight { get; }
    public float[] NormAttnBias { get; }
    public float[] AttnQWeight { get; }
    public float[] AttnKWeight { get; }
    public float[] AttnVWeight { get; }
    public float[] AttnOutWeight { get; }
    public float[] AttnPosWeight { get; }
    public float[] AttnPosBiasU { get; } // [head_dim, n_heads] storage order
    public float[] AttnPosBiasV { get; }

    public float[] NormConvWeight { get; }
    public float[] NormConvBias { get; }
    public float[] ConvPw1Weight { get; }  // [1024, 2048]
    public float[] ConvDwWeight { get; }   // [9, 1, 1024]
    public float[]? ConvDwBias { get; }
    public float[] ConvNormWeight { get; } // real LayerNorm affine (mis-named "batch_norm" in checkpoint)
    public float[] ConvNormBias { get; }
    public float[] ConvPw2Weight { get; }  // [1024, 1024]

    public float[] NormFf2Weight { get; }
    public float[] NormFf2Bias { get; }
    public float[] Ff2Linear1Weight { get; }
    public float[]? Ff2Linear1Bias { get; }
    public float[] Ff2Linear2Weight { get; }
    public float[]? Ff2Linear2Bias { get; }

    public float[] NormOutWeight { get; }
    public float[] NormOutBias { get; }

    public NemotronAsrConformerLayer(NemotronAsrWeights w, string prefix)
    {
        NormFf1Weight = w.GetTensor($"{prefix}.norm_feed_forward1.weight");
        NormFf1Bias = w.GetTensor($"{prefix}.norm_feed_forward1.bias");
        Ff1Linear1Weight = w.GetTensor($"{prefix}.feed_forward1.linear1.weight");
        Ff1Linear1Bias = w.TryGetTensor($"{prefix}.feed_forward1.linear1.bias");
        Ff1Linear2Weight = w.GetTensor($"{prefix}.feed_forward1.linear2.weight");
        Ff1Linear2Bias = w.TryGetTensor($"{prefix}.feed_forward1.linear2.bias");

        NormAttnWeight = w.GetTensor($"{prefix}.norm_self_att.weight");
        NormAttnBias = w.GetTensor($"{prefix}.norm_self_att.bias");
        AttnQWeight = w.GetTensor($"{prefix}.self_attn.linear_q.weight");
        AttnKWeight = w.GetTensor($"{prefix}.self_attn.linear_k.weight");
        AttnVWeight = w.GetTensor($"{prefix}.self_attn.linear_v.weight");
        AttnOutWeight = w.GetTensor($"{prefix}.self_attn.linear_out.weight");
        AttnPosWeight = w.GetTensor($"{prefix}.self_attn.linear_pos.weight");
        AttnPosBiasU = w.GetTensor($"{prefix}.self_attn.pos_bias_u");
        AttnPosBiasV = w.GetTensor($"{prefix}.self_attn.pos_bias_v");

        NormConvWeight = w.GetTensor($"{prefix}.norm_conv.weight");
        NormConvBias = w.GetTensor($"{prefix}.norm_conv.bias");
        ConvPw1Weight = w.GetTensor($"{prefix}.conv.pointwise_conv1.weight");
        ConvDwWeight = w.GetTensor($"{prefix}.conv.depthwise_conv.weight");
        ConvDwBias = w.TryGetTensor($"{prefix}.conv.depthwise_conv.bias");
        ConvNormWeight = w.GetTensor($"{prefix}.conv.batch_norm.weight");
        ConvNormBias = w.GetTensor($"{prefix}.conv.batch_norm.bias");
        ConvPw2Weight = w.GetTensor($"{prefix}.conv.pointwise_conv2.weight");

        NormFf2Weight = w.GetTensor($"{prefix}.norm_feed_forward2.weight");
        NormFf2Bias = w.GetTensor($"{prefix}.norm_feed_forward2.bias");
        Ff2Linear1Weight = w.GetTensor($"{prefix}.feed_forward2.linear1.weight");
        Ff2Linear1Bias = w.TryGetTensor($"{prefix}.feed_forward2.linear1.bias");
        Ff2Linear2Weight = w.GetTensor($"{prefix}.feed_forward2.linear2.weight");
        Ff2Linear2Bias = w.TryGetTensor($"{prefix}.feed_forward2.linear2.bias");

        NormOutWeight = w.GetTensor($"{prefix}.norm_out.weight");
        NormOutBias = w.GetTensor($"{prefix}.norm_out.bias");
    }
}

/// <summary>Real PyTorch `nn.LSTM` per-layer weights (input-to-hidden and hidden-to-hidden, each
/// `[4*hidden, in]` with gates packed in real PyTorch order `i,f,g,o`).</summary>
public sealed class NemotronAsrLstmLayer
{
    public float[] WeightIh { get; } // [4*hidden, in]
    public float[] BiasIh { get; }
    public float[] WeightHh { get; } // [4*hidden, hidden]
    public float[] BiasHh { get; }

    public NemotronAsrLstmLayer(NemotronAsrWeights w, string prefix, int layer)
    {
        WeightIh = w.GetTensor($"{prefix}.ih_l{layer}.weight");
        BiasIh = w.GetTensor($"{prefix}.ih_l{layer}.bias");
        WeightHh = w.GetTensor($"{prefix}.hh_l{layer}.weight");
        BiasHh = w.GetTensor($"{prefix}.hh_l{layer}.bias");
    }
}
