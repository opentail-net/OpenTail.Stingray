using System;
using System.IO;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// Real MeloTTS VITS2 weights, loaded directly from the .onnx file's initializer tensors
/// (same technique as `Piper.PiperOnnxWeights` -- MeloTTS ships only ONNX, no GGUF conversion).
///
/// Architecture (confirmed by walking `models/melotts-zh_en.onnx`'s real node graph, and cross-
/// checked against the real PyTorch source recovered from PyPI's `MeloTTS==0.1.1` sdist,
/// extracted to `examples/melotts-py/`): VITS2 -- enc_p (TextEncoder: token+tone+language
/// embeddings + zero'd bert/ja_bert conv + speaker-conditioned N-layer relative-attention
/// Transformer + proj to mu/logs), sdp+dp (blended StochasticDurationPredictor/
/// DurationPredictor), flow (TransformerCouplingBlock -- NOT Piper's plain ResidualCouplingBlock,
/// each coupling layer has its OWN internal relative-attention encoder), dec (HiFi-GAN, 44.1kHz).
/// This loader currently covers enc_p only (stage 1 of the port, see
/// docs/audio-review-progress.md's MeloTTS section for full architecture notes + remaining
/// stages).
///
/// IMPORTANT quirks of this specific exported checkpoint, confirmed empirically via
/// scratch-llamacpp-ref/melo_golden_textenc.py (NOT assumptions -- re-verify if a different
/// MeloTTS ONNX export is ever substituted):
/// 1. `bert`/`ja_bert` graph inputs don't exist at all in this export -- `bert_proj`/
///    `ja_bert_proj`'s conv input traces back to a `ConstantOfShape` (all-zero) node, so their
///    contribution is bias-only (confirmed: bert_proj's output is constant across every time
///    step). No real BERT features are needed to match this checkpoint.
/// 2. `language` is likewise not a real graph input -- `language_emb`'s Gather index resolves
///    (regardless of the actual token/tone values fed in) to a fixed alternating pattern purely
///    keyed by token POSITION PARITY: language_id[i] = 0 if i is even, else 3 (confirmed for
///    T=8 and T=13 with varying token/tone inputs -- always the same 0,3,0,3,... pattern). This
///    is almost certainly a frozen artifact of whatever example input was used when this
///    checkpoint's export script called torch.onnx.export (not real dynamic per-token language
///    detection) -- but it IS the ground truth this exact graph computes, so the C# port must
///    reproduce it exactly to match golden output, not "fix" it to something more sensible.
/// </summary>
public sealed class MeloOnnxWeights
{
    public OnnxModel Model { get; }

    public int HiddenDim { get; }
    public int NumHeads { get; } = 2;
    public int WindowSize { get; } = 4; // emb_rel_k/v are [1, 2*window+1, headDim] = [1,9,96] -> window=4
    public int NumEncoderLayers { get; }
    public int FfnKernel { get; } = 3;
    public int VocabSize { get; }
    public int NumTones { get; }
    public int NumLanguages { get; }
    public int GinChannels { get; } = 256;
    public int SpkEmbCondLayerIdx { get; } = 2; // attentions.py Encoder's default cond_layer_idx

    public float[] EmbWeight { get; }        // [VocabSize, HiddenDim]
    public float[] ToneEmbWeight { get; }    // [NumTones, HiddenDim]
    public float[] LanguageEmbWeight { get; } // [NumLanguages, HiddenDim]
    public float[] BertProjBias { get; }     // [HiddenDim] -- weight unused, input is always zero (see class doc)
    public float[] JaBertProjBias { get; }   // [HiddenDim] -- same
    public float[] EmbGWeight { get; }       // [NumSpeakers, GinChannels] speaker embedding table

    public float[] ProjWeight { get; }
    public float[] ProjBias { get; }

    public float[] SpkEmbLinearBias { get; } // [HiddenDim]; weight resolved via node (weight_norm-anonymized)
    private readonly float[] _spkEmbLinearWeight; // [HiddenDim, GinChannels]
    public float[] SpkEmbLinearWeight => _spkEmbLinearWeight;

    public MeloEncoderLayerWeights[] Layers { get; }

    // --- dp.* : plain DurationPredictor. filter_channels=256 (models.py DurationPredictor default). ---
    public float[] DpCondWeight { get; }
    public float[] DpCondBias { get; }
    public float[] DpConv1Weight { get; }
    public float[] DpConv1Bias { get; }
    public float[] DpNorm1Gamma { get; }
    public float[] DpNorm1Beta { get; }
    public float[] DpConv2Weight { get; }
    public float[] DpConv2Bias { get; }
    public float[] DpNorm2Gamma { get; }
    public float[] DpNorm2Beta { get; }
    public float[] DpProjWeight { get; }
    public float[] DpProjBias { get; }
    public int DpFilterChannels { get; }

    // --- sdp.* : StochasticDurationPredictor. filter_channels == in_channels == HiddenDim here
    // (models.py line 167: "filter_channels = in_channels # it needs to be removed from future
    // version" -- a real, if oddly-named, override in the reference itself, not a mistake here).
    public float[] SdpCondWeight { get; }
    public float[] SdpCondBias { get; }
    public float[] SdpPreWeight { get; }
    public float[] SdpPreBias { get; }
    public VitsDdsConvWeights SdpConvs { get; }
    public float[] SdpProjWeight { get; }
    public float[] SdpProjBias { get; }
    public float[] SdpFlow0M { get; }
    public float[] SdpFlow0ExpNegLogs { get; }
    public VitsConvFlowWeights SdpFlow3 { get; }
    public VitsConvFlowWeights SdpFlow5 { get; }
    public VitsConvFlowWeights SdpFlow7 { get; }

    // --- flow.* : TransformerCouplingBlock, 4 coupling layers (indices 0,2,4,6 in ONNX weight
    // paths; odd indices are parameter-free Flip) interleaved with Flip -- see MeloFlow.
    public MeloFlowLayerWeights[] FlowLayers { get; }
    public const int FlowFfnKernel = 5; // models.py: TransformerCouplingBlock(..., kernel_size=5, ...) -- differs from enc_p's 3

    // --- dec.* : Generator (HiFi-GAN, ResBlock1-style -- 3 conv PAIRS per resblock at fixed
    // dilations [1,3,5], confirmed via real ONNX Conv node attribute inspection -- structurally
    // different from Piper's simpler resblock, see MeloGenerator's class doc). 5 upsample stages
    // 512->256->128->64->32->16 channels, kernels [16,16,8,2,2], strides [8,8,2,2,2] (total
    // upsample factor 512, matching MeloTTS's 44.1kHz/hop-512 mel frame rate). 3 resblock kernel
    // sizes [3,7,11] per stage (15 resblocks total = 5 stages * 3 kernels).
    public float[] DecConvPreWeight { get; }
    public float[] DecConvPreBias { get; }
    public float[] DecCondWeight { get; }
    public float[] DecCondBias { get; }
    public float[][] DecUpsWeight { get; } = new float[5][];
    public float[][] DecUpsBias { get; } = new float[5][];
    public MeloResBlockWeights[] DecResblocks { get; } = new MeloResBlockWeights[15];
    public float[] DecConvPostWeight { get; }

    public MeloOnnxWeights(string onnxPath)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException($"MeloTTS ONNX model file not found: {onnxPath}");

        Model = OnnxModel.Open(onnxPath);

        EmbWeight = GetFloat("model.model.enc_p.emb.weight", out int vocab, out int hidden);
        VocabSize = vocab;
        HiddenDim = hidden;

        ToneEmbWeight = GetFloat("model.model.enc_p.tone_emb.weight", out int numTones, out _);
        NumTones = numTones;

        LanguageEmbWeight = GetFloat("model.model.enc_p.language_emb.weight", out int numLangs, out _);
        NumLanguages = numLangs;

        BertProjBias = GetFloat1D("model.model.enc_p.bert_proj.bias");
        JaBertProjBias = GetFloat1D("model.model.enc_p.ja_bert_proj.bias");
        EmbGWeight = GetFloat("model.model.emb_g.weight", out _, out _);

        ProjWeight = GetFloat1D("model.model.enc_p.proj.weight");
        ProjBias = GetFloat1D("model.model.enc_p.proj.bias");

        SpkEmbLinearBias = GetFloat1D("model.model.enc_p.encoder.spk_emb_linear.bias");
        _spkEmbLinearWeight = OnnxModel.ToFloat32(Model.GetWeightViaNode("/enc_p/encoder/spk_emb_linear/MatMul", 1));

        int n = 0;
        while (TryResolveNode($"/enc_p/encoder/attn_layers.{n}/conv_q/Conv") is not null) n++;
        NumEncoderLayers = n;

        Layers = new MeloEncoderLayerWeights[NumEncoderLayers];
        for (int i = 0; i < NumEncoderLayers; i++)
            Layers[i] = new MeloEncoderLayerWeights(this, i);

        DpCondWeight = GetFloat1D("model.model.dp.cond.weight");
        DpCondBias = GetFloat1D("model.model.dp.cond.bias");
        DpConv1Weight = GetFloat1D("model.model.dp.conv_1.weight");
        DpConv1Bias = GetFloat1D("model.model.dp.conv_1.bias");
        DpNorm1Gamma = GetFloat1D("model.model.dp.norm_1.gamma");
        DpNorm1Beta = GetFloat1D("model.model.dp.norm_1.beta");
        DpConv2Weight = GetFloat1D("model.model.dp.conv_2.weight");
        DpConv2Bias = GetFloat1D("model.model.dp.conv_2.bias");
        DpNorm2Gamma = GetFloat1D("model.model.dp.norm_2.gamma");
        DpNorm2Beta = GetFloat1D("model.model.dp.norm_2.beta");
        DpProjWeight = GetFloat1D("model.model.dp.proj.weight");
        DpProjBias = GetFloat1D("model.model.dp.proj.bias");
        DpFilterChannels = DpConv1Bias.Length;

        SdpCondWeight = GetFloat1D("model.model.sdp.cond.weight");
        SdpCondBias = GetFloat1D("model.model.sdp.cond.bias");
        SdpPreWeight = GetFloat1D("model.model.sdp.pre.weight");
        SdpPreBias = GetFloat1D("model.model.sdp.pre.bias");
        SdpConvs = new VitsDdsConvWeights(GetFloatNamed, "model.model.sdp.convs");
        SdpProjWeight = GetFloat1D("model.model.sdp.proj.weight");
        SdpProjBias = GetFloat1D("model.model.sdp.proj.bias");
        SdpFlow0M = GetFloat1D("model.model.sdp.flows.0.m");
        SdpFlow0ExpNegLogs = GetFloatViaNode("/sdp/flows.0/Exp", 0);
        SdpFlow3 = new VitsConvFlowWeights(GetFloatNamed, "model.model.sdp.flows.3");
        SdpFlow5 = new VitsConvFlowWeights(GetFloatNamed, "model.model.sdp.flows.5");
        SdpFlow7 = new VitsConvFlowWeights(GetFloatNamed, "model.model.sdp.flows.7");

        FlowLayers = new MeloFlowLayerWeights[4];
        for (int i = 0; i < 4; i++)
            FlowLayers[i] = new MeloFlowLayerWeights(this, i * 2);

        DecConvPreWeight = GetFloat1D("model.model.dec.conv_pre.weight");
        DecConvPreBias = GetFloat1D("model.model.dec.conv_pre.bias");
        DecCondWeight = GetFloat1D("model.model.dec.cond.weight");
        DecCondBias = GetFloat1D("model.model.dec.cond.bias");
        for (int i = 0; i < 5; i++)
        {
            DecUpsWeight[i] = GetFloatViaNode($"/dec/ups.{i}/ConvTranspose");
            DecUpsBias[i] = GetFloat1D($"model.model.dec.ups.{i}.bias");
        }
        for (int i = 0; i < 15; i++)
            DecResblocks[i] = new MeloResBlockWeights(this, i);
        DecConvPostWeight = GetFloat1D("model.model.dec.conv_post.weight");
    }

    private OnnxTensor? TryResolveNode(string nodeName)
    {
        try { return Model.GetWeightViaNode(nodeName, 1); }
        catch (InvalidDataException) { return null; }
    }

    /// <summary>Reads a 2D tensor initializer by its (clean, non-anonymized) name; dims[0]=rows, dims[1]=cols.</summary>
    private float[] GetFloat(string name, out int rows, out int cols)
    {
        var t = Model.GetTensor(name);
        rows = t.Dims[0];
        cols = t.Dims.Length > 1 ? t.Dims[1] : 1;
        return OnnxModel.ToFloat32(t);
    }

    private float[] GetFloat1D(string name) => OnnxModel.ToFloat32(Model.GetTensor(name));

    /// <summary>Resolves an anonymized (weight_norm-fused) Conv/MatMul weight via its owning node's name.</summary>
    public float[] GetFloatViaNode(string nodeName, int inputIndex = 1) =>
        OnnxModel.ToFloat32(Model.GetWeightViaNode(nodeName, inputIndex));

    public float[] GetFloatNamed(string name) => OnnxModel.ToFloat32(Model.GetTensor(name));
}

/// <summary>
/// One `enc_p.encoder` relative-attention layer. Conv weights are anonymized by PyTorch's
/// weight_norm-style ONNX export fusion (same as Piper) -- resolved via node name, not module
/// path. Bias tensors keep clean module-path names.
/// </summary>
public sealed class MeloEncoderLayerWeights
{
    public float[] EmbRelK { get; }
    public float[] EmbRelV { get; }
    public float[] ConvQWeight { get; }
    public float[] ConvQBias { get; }
    public float[] ConvKWeight { get; }
    public float[] ConvKBias { get; }
    public float[] ConvVWeight { get; }
    public float[] ConvVBias { get; }
    public float[] ConvOWeight { get; }
    public float[] ConvOBias { get; }

    public float[] Norm1Gamma { get; }
    public float[] Norm1Beta { get; }

    public float[] Ffn1Weight { get; }
    public float[] Ffn1Bias { get; }
    public float[] Ffn2Weight { get; }
    public float[] Ffn2Bias { get; }

    public float[] Norm2Gamma { get; }
    public float[] Norm2Beta { get; }

    public MeloEncoderLayerWeights(MeloOnnxWeights w, int i)
        : this(w, "model.model.enc_p.encoder", "/enc_p/encoder", i, weightNormAnonymized: true)
    {
    }

    /// <summary>
    /// General form: modBase/nodeBase identify the owning `attentions.Encoder` instance (either
    /// enc_p's top-level encoder, or one `flow.flows.{k}.enc` per coupling layer -- same module
    /// shape, `attentions.py`'s Encoder class, just at a different weight path). enc_p's conv
    /// weights are anonymized by weight_norm-style ONNX export fusion (must resolve via node
    /// name); the flow's internal encoders are NOT weight_norm'd in this checkpoint (confirmed by
    /// directly inspecting the ONNX initializer list -- clean `model.model.flow.flows.0.enc.
    /// attn_layers.0.conv_q.weight` names exist directly, no `onnx::Conv_NNNNN` indirection), so
    /// weightNormAnonymized=false reads them by name instead.
    /// </summary>
    public MeloEncoderLayerWeights(MeloOnnxWeights w, string modBase, string nodeBase, int i, bool weightNormAnonymized)
    {
        string modPrefix = $"{modBase}.attn_layers.{i}";
        string nodePrefix = $"{nodeBase}/attn_layers.{i}";

        EmbRelK = w.GetFloatNamed($"{modPrefix}.emb_rel_k");
        EmbRelV = w.GetFloatNamed($"{modPrefix}.emb_rel_v");

        if (weightNormAnonymized)
        {
            ConvQWeight = w.GetFloatViaNode($"{nodePrefix}/conv_q/Conv");
            ConvKWeight = w.GetFloatViaNode($"{nodePrefix}/conv_k/Conv");
            ConvVWeight = w.GetFloatViaNode($"{nodePrefix}/conv_v/Conv");
            ConvOWeight = w.GetFloatViaNode($"{nodePrefix}/conv_o/Conv");
        }
        else
        {
            ConvQWeight = w.GetFloatNamed($"{modPrefix}.conv_q.weight");
            ConvKWeight = w.GetFloatNamed($"{modPrefix}.conv_k.weight");
            ConvVWeight = w.GetFloatNamed($"{modPrefix}.conv_v.weight");
            ConvOWeight = w.GetFloatNamed($"{modPrefix}.conv_o.weight");
        }
        ConvQBias = w.GetFloatNamed($"{modPrefix}.conv_q.bias");
        ConvKBias = w.GetFloatNamed($"{modPrefix}.conv_k.bias");
        ConvVBias = w.GetFloatNamed($"{modPrefix}.conv_v.bias");
        ConvOBias = w.GetFloatNamed($"{modPrefix}.conv_o.bias");

        Norm1Gamma = w.GetFloatNamed($"{modBase}.norm_layers_1.{i}.gamma");
        Norm1Beta = w.GetFloatNamed($"{modBase}.norm_layers_1.{i}.beta");

        string ffnModPrefix = $"{modBase}.ffn_layers.{i}";
        string ffnNodePrefix = $"{nodeBase}/ffn_layers.{i}";
        if (weightNormAnonymized)
        {
            Ffn1Weight = w.GetFloatViaNode($"{ffnNodePrefix}/conv_1/Conv");
            Ffn2Weight = w.GetFloatViaNode($"{ffnNodePrefix}/conv_2/Conv");
        }
        else
        {
            Ffn1Weight = w.GetFloatNamed($"{ffnModPrefix}.conv_1.weight");
            Ffn2Weight = w.GetFloatNamed($"{ffnModPrefix}.conv_2.weight");
        }
        Ffn1Bias = w.GetFloatNamed($"{ffnModPrefix}.conv_1.bias");
        Ffn2Bias = w.GetFloatNamed($"{ffnModPrefix}.conv_2.bias");

        Norm2Gamma = w.GetFloatNamed($"{modBase}.norm_layers_2.{i}.gamma");
        Norm2Beta = w.GetFloatNamed($"{modBase}.norm_layers_2.{i}.beta");
    }
}

/// <summary>
/// One `flow.flows.{2k}` `TransformerCouplingLayer` (k=0..3; the checkpoint's `flow` has 4 such
/// layers interleaved with parameter-free `Flip`s -- see `MeloFlow`). Unlike Piper's plain
/// `ResidualCouplingBlock`, each coupling layer here has its own internal 3-layer relative-
/// attention encoder (`enc`, reusing <see cref="MeloRelativeEncoder"/>) plus its own
/// speaker-conditioning `spk_emb_linear`.
/// </summary>
public sealed class MeloFlowLayerWeights
{
    public float[] PreWeight { get; }
    public float[] PreBias { get; }
    public float[] PostWeight { get; }
    public float[] PostBias { get; }
    public float[] SpkEmbLinearBias { get; }
    public float[] SpkEmbLinearWeight { get; } // [HiddenDim, GinChannels], anonymized MatMul (same layout quirk as enc_p's)
    public MeloEncoderLayerWeights[] Layers { get; }

    public MeloFlowLayerWeights(MeloOnnxWeights w, int flowIdx)
    {
        string modPrefix = $"model.model.flow.flows.{flowIdx}";
        string nodePrefix = $"/flow/flows.{flowIdx}";

        PreWeight = w.GetFloatNamed($"{modPrefix}.pre.weight");
        PreBias = w.GetFloatNamed($"{modPrefix}.pre.bias");
        PostWeight = w.GetFloatNamed($"{modPrefix}.post.weight");
        PostBias = w.GetFloatNamed($"{modPrefix}.post.bias");

        SpkEmbLinearBias = w.GetFloatNamed($"{modPrefix}.enc.spk_emb_linear.bias");
        SpkEmbLinearWeight = w.GetFloatViaNode($"{nodePrefix}/enc/spk_emb_linear/MatMul", 1);

        Layers = new MeloEncoderLayerWeights[3];
        for (int i = 0; i < 3; i++)
            Layers[i] = new MeloEncoderLayerWeights(w, $"{modPrefix}.enc", $"{nodePrefix}/enc", i, weightNormAnonymized: false);
    }
}

/// <summary>
/// One `dec.resblocks.{i}` `ResBlock1` (modules.py): 3 conv PAIRS (convs1[j] at dilation
/// dilations[j], then convs2[j] at dilation 1), residual-added in sequence -- NOT Piper's
/// simpler 2-layer resblock. All 6 conv weights are anonymized by weight_norm-style ONNX export
/// fusion (resolved via node name, same as enc_p's attention convs).
/// </summary>
public sealed class MeloResBlockWeights
{
    public float[][] Convs1Weight { get; } = new float[3][];
    public float[][] Convs1Bias { get; } = new float[3][];
    public float[][] Convs2Weight { get; } = new float[3][];
    public float[][] Convs2Bias { get; } = new float[3][];

    public MeloResBlockWeights(MeloOnnxWeights w, int i)
    {
        string modPrefix = $"model.model.dec.resblocks.{i}";
        string nodePrefix = $"/dec/resblocks.{i}";
        for (int j = 0; j < 3; j++)
        {
            Convs1Weight[j] = w.GetFloatViaNode($"{nodePrefix}/convs1.{j}/Conv");
            Convs1Bias[j] = w.GetFloatNamed($"{modPrefix}.convs1.{j}.bias");
            Convs2Weight[j] = w.GetFloatViaNode($"{nodePrefix}/convs2.{j}/Conv");
            Convs2Bias[j] = w.GetFloatNamed($"{modPrefix}.convs2.{j}.bias");
        }
    }
}
