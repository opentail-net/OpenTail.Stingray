
namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Real Piper VITS weights, loaded directly from the .onnx file's initializer tensors (see
/// OpenTail.Stingray.Core.OnnxModel -- Piper ships only ONNX, no GGUF conversion exists, so this
/// reads the ONNX weight bytes directly rather than going through a GGUF loader like every other
/// pipeline in this codebase). All tensors are eagerly decoded to float32 in ONNX/torch storage
/// order (NOT reversed like GGUF's dims convention -- OnnxTensor.Dims is already torch-shape
/// order), matching the "wBase = outIdx*inFeatures" access pattern used throughout this session's
/// other weight-consuming code.
///
/// Architecture (from the initializer name sweep -- see docs/audio-review-progress.md's Piper
/// section): standard VITS -- enc_p (TextEncoder: embedding + N-layer windowed-relative-position
/// Transformer + proj to mu/logs), dp (StochasticDurationPredictor), flow
/// (ResidualCouplingBlock), dec (HiFi-GAN decoder). This loader currently covers enc_p (stage 1).
/// </summary>
public sealed class PiperOnnxWeights
{
    public OnnxModel Model { get; }

    public int HiddenDim { get; } = 192;
    public int NumHeads { get; } = 2;
    public int HeadDim => HiddenDim / NumHeads;
    public int WindowSize { get; } = 4; // relative position window radius (emb_rel_k/v are [1, 2*window+1, headDim])
    public int NumEncoderLayers { get; }
    public int FfnKernel { get; } = 3;
    public int VocabSize { get; }

    public float[] EmbeddingWeight { get; }   // [VocabSize, HiddenDim] -- misleadingly named "sid" in this ONNX export
    public float[] ProjWeight { get; }        // [2*HiddenDim, HiddenDim, 1] conv1x1 -> mu/logs
    public float[] ProjBias { get; }

    public PiperEncoderLayerWeights[] Layers { get; }

    // --- dp.* : StochasticDurationPredictor ---
    public float[] DpPreWeight { get; }
    public float[] DpPreBias { get; }
    public VitsDdsConvWeights DpConvs { get; }
    public float[] DpProjWeight { get; }
    public float[] DpProjBias { get; }
    public float[] DpFlow0M { get; }              // dp.flows.0.m, ElementwiseAffine's m [2]
    public float[] DpFlow0ExpNegLogs { get; }      // /dp/flows.0/Exp_output_0, constant-folded exp(-logs) [2]
    public VitsConvFlowWeights DpFlow3 { get; }
    public VitsConvFlowWeights DpFlow5 { get; }
    public VitsConvFlowWeights DpFlow7 { get; }

    // --- flow.* : ResidualCouplingBlock (4 ResidualCouplingLayers at list-indices 0,2,4,6;
    // indices 1,3,5,7 are parameter-free Flip). Reverse (inference) execution order is
    // list-reversed: Flip(7), Layer(6), Flip(5), Layer(4), Flip(3), Layer(2), Flip(1), Layer(0).
    public const int FlowWnLayers = 4;
    public const int FlowWnKernel = 5;
    public const int FlowWnDilation = 1; // constant across layers for this checkpoint (not the vanilla-VITS exponential default)
    public const int FlowHalfChannels = 96;
    public PiperCouplingLayerWeights FlowLayer0 { get; }
    public PiperCouplingLayerWeights FlowLayer2 { get; }
    public PiperCouplingLayerWeights FlowLayer4 { get; }
    public PiperCouplingLayerWeights FlowLayer6 { get; }

    // --- dec.* : HiFi-GAN vocoder. conv_pre (192->256, k=7), 3 upsample stages
    // (ConvTranspose1d + 3-way-averaged 2-conv resblocks), conv_post (k=7, no bias) + Tanh.
    public float[] DecConvPreWeight { get; }
    public float[] DecConvPreBias { get; }
    public float[][] DecUpsWeight { get; } = new float[3][];
    public float[][] DecUpsBias { get; } = new float[3][];
    public PiperResBlockWeights[] DecResblocks { get; } = new PiperResBlockWeights[9];
    public float[] DecConvPostWeight { get; }

    public PiperOnnxWeights(string onnxPath)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException($"Piper ONNX model file not found: {onnxPath}");

        Model = OnnxModel.Open(onnxPath);

        var emb = Model.GetTensor("sid");
        VocabSize = emb.Dims[0];
        HiddenDim = emb.Dims[1];
        EmbeddingWeight = OnnxModel.ToFloat32(emb);

        ProjWeight = GetFloat("enc_p.proj.weight");
        ProjBias = GetFloat("enc_p.proj.bias");

        int n = 0;
        while (Model.TryGetTensor($"enc_p.encoder.attn_layers.{n}.conv_q.weight") is not null) n++;
        NumEncoderLayers = n;

        Layers = new PiperEncoderLayerWeights[NumEncoderLayers];
        for (int i = 0; i < NumEncoderLayers; i++)
            Layers[i] = new PiperEncoderLayerWeights(this, i);

        DpPreWeight = GetFloat("dp.pre.weight");
        DpPreBias = GetFloat("dp.pre.bias");
        DpConvs = new VitsDdsConvWeights(GetFloat, "dp.convs");
        DpProjWeight = GetFloat("dp.proj.weight");
        DpProjBias = GetFloat("dp.proj.bias");
        DpFlow0M = GetFloat("dp.flows.0.m");
        // The reference's exp(-logs) is a fixed (untrained-input-independent) parameter, which
        // ONNX export constant-folds into a plain graph value rather than an initializer -- read
        // via TryGetTensor's fallback to the model's node-output-named constants (OnnxModel treats
        // any protobuf initializer entry uniformly regardless of its Constant-node-style name).
        DpFlow0ExpNegLogs = GetFloat("/dp/flows.0/Exp_output_0");
        DpFlow3 = new VitsConvFlowWeights(GetFloat, "dp.flows.3");
        DpFlow5 = new VitsConvFlowWeights(GetFloat, "dp.flows.5");
        DpFlow7 = new VitsConvFlowWeights(GetFloat, "dp.flows.7");

        FlowLayer0 = new PiperCouplingLayerWeights(this, "flow.flows.0");
        FlowLayer2 = new PiperCouplingLayerWeights(this, "flow.flows.2");
        FlowLayer4 = new PiperCouplingLayerWeights(this, "flow.flows.4");
        FlowLayer6 = new PiperCouplingLayerWeights(this, "flow.flows.6");

        DecConvPreWeight = GetFloat("dec.conv_pre.weight");
        DecConvPreBias = GetFloat("dec.conv_pre.bias");
        for (int i = 0; i < 3; i++)
        {
            DecUpsWeight[i] = GetFloat($"dec.ups.{i}.weight");
            DecUpsBias[i] = GetFloat($"dec.ups.{i}.bias");
        }
        for (int i = 0; i < 9; i++)
            DecResblocks[i] = new PiperResBlockWeights(this, $"dec.resblocks.{i}");
        DecConvPostWeight = GetFloat("dec.conv_post.weight");
    }

    /// <summary>
    /// Resolves a Conv node's real weight tensor, which PyTorch's weight_norm reparametrization
    /// causes the ONNX exporter to fuse into an anonymously-named constant (e.g. "onnx::Conv_8240")
    /// rather than a "{module}.weight"-named initializer -- see OnnxModel.GetWeightViaNode's doc
    /// comment. nodeName is the ONNX node's own name (e.g.
    /// "/flow/flows.0/enc/in_layers.0/Conv"), NOT a module path.
    /// </summary>
    public float[] GetFloatViaNode(string nodeName) => OnnxModel.ToFloat32(Model.GetWeightViaNode(nodeName, 1));

    public float[] GetFloat(string name) => OnnxModel.ToFloat32(Model.GetTensor(name));
}

public sealed class PiperEncoderLayerWeights
{
    public float[] EmbRelK { get; }   // [2*window+1, headDim]
    public float[] EmbRelV { get; }
    public float[] ConvQWeight { get; } // [hidden, hidden, 1]
    public float[] ConvQBias { get; }
    public float[] ConvKWeight { get; }
    public float[] ConvKBias { get; }
    public float[] ConvVWeight { get; }
    public float[] ConvVBias { get; }
    public float[] ConvOWeight { get; }
    public float[] ConvOBias { get; }

    public float[] Norm1Gamma { get; }
    public float[] Norm1Beta { get; }

    public float[] Ffn1Weight { get; }  // conv_1 [ffnHidden, hidden, kernel]
    public float[] Ffn1Bias { get; }
    public float[] Ffn2Weight { get; }  // conv_2 [hidden, ffnHidden, kernel]
    public float[] Ffn2Bias { get; }

    public float[] Norm2Gamma { get; }
    public float[] Norm2Beta { get; }

    public PiperEncoderLayerWeights(PiperOnnxWeights w, int i)
    {
        string p = $"enc_p.encoder.attn_layers.{i}";
        EmbRelK = w.GetFloat($"{p}.emb_rel_k");
        EmbRelV = w.GetFloat($"{p}.emb_rel_v");
        ConvQWeight = w.GetFloat($"{p}.conv_q.weight");
        ConvQBias = w.GetFloat($"{p}.conv_q.bias");
        ConvKWeight = w.GetFloat($"{p}.conv_k.weight");
        ConvKBias = w.GetFloat($"{p}.conv_k.bias");
        ConvVWeight = w.GetFloat($"{p}.conv_v.weight");
        ConvVBias = w.GetFloat($"{p}.conv_v.bias");
        ConvOWeight = w.GetFloat($"{p}.conv_o.weight");
        ConvOBias = w.GetFloat($"{p}.conv_o.bias");

        Norm1Gamma = w.GetFloat($"enc_p.encoder.norm_layers_1.{i}.gamma");
        Norm1Beta = w.GetFloat($"enc_p.encoder.norm_layers_1.{i}.beta");

        string f = $"enc_p.encoder.ffn_layers.{i}";
        Ffn1Weight = w.GetFloat($"{f}.conv_1.weight");
        Ffn1Bias = w.GetFloat($"{f}.conv_1.bias");
        Ffn2Weight = w.GetFloat($"{f}.conv_2.weight");
        Ffn2Bias = w.GetFloat($"{f}.conv_2.bias");

        Norm2Gamma = w.GetFloat($"enc_p.encoder.norm_layers_2.{i}.gamma");
        Norm2Beta = w.GetFloat($"enc_p.encoder.norm_layers_2.{i}.beta");
    }
}

/// <summary>
/// ResidualCouplingLayer (mean_only=True, confirmed by post.weight's output channel count
/// equalling half_channels rather than 2*half_channels -- so logs is always 0, i.e. reverse
/// mode's `(x1-m)*exp(-logs)` degrades to a plain `x1-m`): pre (1x1) -> WN -> post (1x1, m only).
/// </summary>
public sealed class PiperCouplingLayerWeights
{
    public float[] PreWeight { get; }
    public float[] PreBias { get; }
    public PiperWnWeights Enc { get; }
    public float[] PostWeight { get; } // [halfChannels, hidden, 1] -- m only (mean_only)
    public float[] PostBias { get; }

    public PiperCouplingLayerWeights(PiperOnnxWeights w, string prefix)
    {
        PreWeight = w.GetFloat($"{prefix}.pre.weight");
        PreBias = w.GetFloat($"{prefix}.pre.bias");
        Enc = new PiperWnWeights(w, prefix);
        PostWeight = w.GetFloat($"{prefix}.post.weight");
        PostBias = w.GetFloat($"{prefix}.post.bias");
    }
}

/// <summary>
/// WN (WaveNet-style dilated conv stack), gin_channels=0 (no speaker conditioning -- this
/// checkpoint is single-speaker). in_layers/res_skip_layers' real weights are anonymously-named
/// ONNX constants (PyTorch weight_norm fusion) resolved via the node graph, not by module path --
/// see PiperOnnxWeights.GetFloatViaNode.
/// </summary>
public sealed class PiperWnWeights
{
    public float[][] InWeight { get; } = new float[PiperOnnxWeights.FlowWnLayers][];   // [2*hidden, hidden, kernel]
    public float[][] InBias { get; } = new float[PiperOnnxWeights.FlowWnLayers][];
    public float[][] ResSkipWeight { get; } = new float[PiperOnnxWeights.FlowWnLayers][]; // [2*hidden or hidden, hidden, 1]
    public float[][] ResSkipBias { get; } = new float[PiperOnnxWeights.FlowWnLayers][];

    private static string ReplaceFirstDot(string s)
    {
        int idx = s.IndexOf('.');
        return idx < 0 ? s : s[..idx] + "/" + s[(idx + 1)..];
    }

    public PiperWnWeights(PiperOnnxWeights w, string prefix)
    {
        for (int i = 0; i < PiperOnnxWeights.FlowWnLayers; i++)
        {
            // Node names keep the "flow.flows.N" segment dotted (only the top-level "flow" gets its
            // own path component) -- e.g. "/flow/flows.0/enc/in_layers.0/Conv", NOT
            // "/flow/flows/0/enc/...". Only replace the FIRST dot (module/instance separator).
            string nodePrefix = "/" + ReplaceFirstDot(prefix) + "/enc";
            InWeight[i] = w.GetFloatViaNode($"{nodePrefix}/in_layers.{i}/Conv");
            InBias[i] = w.GetFloat($"{prefix}.enc.in_layers.{i}.bias");
            ResSkipWeight[i] = w.GetFloatViaNode($"{nodePrefix}/res_skip_layers.{i}/Conv");
            ResSkipBias[i] = w.GetFloat($"{prefix}.enc.res_skip_layers.{i}.bias");
        }
    }
}

/// <summary>HiFi-GAN "resblock2"-style residual block: 2 dilated convs (dilations [1,2]), each
/// preceded by LeakyReLU, with a residual add after each. Weight/bias names are plain module-path
/// initializers (not weight_norm-anonymized) for this checkpoint's decoder.</summary>
public sealed class PiperResBlockWeights
{
    public float[][] ConvWeight { get; } = new float[2][];
    public float[][] ConvBias { get; } = new float[2][];

    public PiperResBlockWeights(PiperOnnxWeights w, string prefix)
    {
        for (int i = 0; i < 2; i++)
        {
            ConvWeight[i] = w.GetFloat($"{prefix}.convs.{i}.weight");
            ConvBias[i] = w.GetFloat($"{prefix}.convs.{i}.bias");
        }
    }
}
