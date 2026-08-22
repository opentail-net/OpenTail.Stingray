using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real GGUF weight loader for Alibaba's Paraformer (FunASR family), `general.architecture =
/// "paraformer"`. Confirmed real architecture directly from `models/paraformer-q8.gguf`'s own
/// metadata/tensor names this session (not guessed): a SAN-M (Self-Attention with Memory
/// block, i.e. multi-head self-attention + a depthwise FSMN conv memory term) encoder with one
/// special first layer (`encoder.encoders0.0`, 560-dim input -- CMVN'd 80-mel x 7-frame splice)
/// followed by <see cref="EncoderLayers"/> (50) main 512-dim layers (`encoder.encoders.N`); a
/// CIF (Continuous Integrate-and-Fire) predictor (`predictor.cif_conv1d` + `predictor.
/// cif_output`) that both counts and boundary-detects acoustic tokens from the encoder output;
/// and a non-autoregressive cross-attention decoder with <see cref="DecoderLayers"/> (16) main
/// layers (`decoder.decoders.N`, self_attn is FSMN-only -- no Q/K/V, matching Paraformer's
/// non-autoregressive design where the decoder doesn't attend to its own un-emitted future
/// tokens) plus one extra FFN-only layer (`decoder.decoders3.0`) and a real vocab projection
/// (`decoder.output_layer`, 512-&gt;8404) to this checkpoint's real, GGUF-embedded 8404-token
/// vocabulary (`pf.vocab` metadata, plain string array -- real, not synthetic).
///
/// <para><b>IMPORTANT, found this session, do not re-derive</b>: the only local C++ reference
/// (`examples/paraformer.cpp/src/csrc/paraformer-offline.cpp`) has the encoder's FSMN memory
/// term ADD DISABLED (`// todo open when conv depth wise 1d with group implement finished` --
/// see its line ~1767) -- i.e. that specific reference is a known-incomplete/broken
/// implementation on exactly the detail that matters most for SAN-M correctness. Do NOT port
/// its encoder forward pass as-is; the real FSMN formula needs deriving from FunASR's actual
/// Python source (the `funasr` package's `funasr/models/sanm/attention.py`/`encoder.py`, not
/// yet fetched this session) before writing `FunAsrEncoder.cs`'s forward pass. This class only
/// loads weights -- no forward-pass math lives here.</para>
/// </summary>
public sealed class FunAsrWeights : IDisposable
{
    public GgufModel Model { get; }

    public int EncoderLayers { get; } = 49; // main encoders.N stack only; encoders0.0 is separate (see constructor comment)
    public int EncoderHeads { get; } = 4;
    public int EncoderDim { get; } = 512;
    public int DecoderLayers { get; } = 16;
    public int DecoderHeads { get; } = 4;
    public int FsmnKernelSize { get; } = 11;
    public float CifThreshold { get; } = 1.0f;
    public float CifTailThreshold { get; } = 0.45f;
    public int VocabSize { get; } = 8404;

    /// <summary>Real 8404-entry vocabulary, index = token id. FunASR/ESPnet BPE convention: a trailing `@@` marks "this token continues into the next" (join without a space); no `@@` means the next token starts a new word (join with a space) -- confirmed by inspecting the real vocab strings directly (e.g. `and@@`).</summary>
    public string[] Vocab { get; }

    public float[] CmvnScale { get; }
    public float[] CmvnShift { get; }

    public FunAsrEncoderLayerWeights Encoders0Layer { get; }
    public FunAsrEncoderLayerWeights[] EncoderLayerWeights { get; }
    public float[] EncoderAfterNormWeight { get; }
    public float[] EncoderAfterNormBias { get; }

    public float[] PredictorCifConv1dWeight { get; } // [3, 512, 512] kernel=3 depthwise-ish conv, see predictor forward (not yet ported)
    public float[] PredictorCifConv1dBias { get; }
    public float[] PredictorCifOutputWeight { get; } // [512, 1]
    public float[] PredictorCifOutputBias { get; }

    public FunAsrDecoderLayerWeights[] DecoderLayerWeights { get; }
    public FunAsrDecoderFfnLayerWeights Decoders3Layer { get; }
    public float[] DecoderAfterNormWeight { get; }
    public float[] DecoderAfterNormBias { get; }
    public float[] DecoderOutputWeight { get; } // [512, 8404]
    public float[] DecoderOutputBias { get; }   // [8404]

    public FunAsrWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Paraformer model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        // pf.enc.num_blocks (50) counts encoders0.0 (the special 560-dim first layer) PLUS the
        // main encoders.N stack combined -- confirmed by direct inspection: the real GGUF only
        // has encoder.encoders.{0..48} (49 layers), not 50. EncoderLayers here is the main-stack
        // count only (49); encoders0 is loaded separately as Encoders0Layer below.
        if (Model.Metadata.TryGetValue("pf.enc.num_blocks", out var enb) && enb is int enbi) EncoderLayers = enbi - 1;
        if (Model.Metadata.TryGetValue("pf.enc.attention_heads", out var enh) && enh is int enhi) EncoderHeads = enhi;
        if (Model.Metadata.TryGetValue("pf.enc.output_size", out var eos) && eos is int eosi) EncoderDim = eosi;
        if (Model.Metadata.TryGetValue("pf.dec.num_blocks", out var dnb) && dnb is int dnbi) DecoderLayers = dnbi;
        if (Model.Metadata.TryGetValue("pf.dec.attention_heads", out var dnh) && dnh is int dnhi) DecoderHeads = dnhi;
        if (Model.Metadata.TryGetValue("pf.enc.kernel_size", out var ks) && ks is int ksi) FsmnKernelSize = ksi;
        if (Model.Metadata.TryGetValue("pf.predictor.threshold", out var pt) && pt is float pti) CifThreshold = pti;
        if (Model.Metadata.TryGetValue("pf.predictor.tail_threshold", out var ptt) && ptt is float ptti) CifTailThreshold = ptti;
        if (Model.Metadata.TryGetValue("pf.vocab_size", out var vs) && vs is int vsi) VocabSize = vsi;

        if (!Model.Metadata.TryGetValue("pf.vocab", out var vocabObj))
            throw new InvalidDataException("Paraformer GGUF missing 'pf.vocab' metadata.");
        var vocabArray = (object[])vocabObj;
        Vocab = new string[vocabArray.Length];
        for (int i = 0; i < vocabArray.Length; i++) Vocab[i] = (string)vocabArray[i];

        CmvnScale = GetTensor("cmvn.scale");
        CmvnShift = GetTensor("cmvn.shift");

        Encoders0Layer = LoadEncoderLayer("encoder.encoders0.0", inputDim: 560);
        EncoderLayerWeights = new FunAsrEncoderLayerWeights[EncoderLayers];
        for (int i = 0; i < EncoderLayers; i++)
            EncoderLayerWeights[i] = LoadEncoderLayer($"encoder.encoders.{i}", inputDim: EncoderDim);
        EncoderAfterNormWeight = GetTensor("encoder.after_norm.weight");
        EncoderAfterNormBias = GetTensor("encoder.after_norm.bias");

        PredictorCifConv1dWeight = GetTensor("predictor.cif_conv1d.weight");
        PredictorCifConv1dBias = GetTensor("predictor.cif_conv1d.bias");
        PredictorCifOutputWeight = GetTensor("predictor.cif_output.weight");
        PredictorCifOutputBias = GetTensor("predictor.cif_output.bias");

        DecoderLayerWeights = new FunAsrDecoderLayerWeights[DecoderLayers];
        for (int i = 0; i < DecoderLayers; i++)
            DecoderLayerWeights[i] = LoadDecoderLayer($"decoder.decoders.{i}");
        Decoders3Layer = LoadDecoderFfnLayer("decoder.decoders3.0");
        DecoderAfterNormWeight = GetTensor("decoder.after_norm.weight");
        DecoderAfterNormBias = GetTensor("decoder.after_norm.bias");
        DecoderOutputWeight = GetTensor("decoder.output_layer.weight");
        DecoderOutputBias = GetTensor("decoder.output_layer.bias");
    }

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Paraformer GGUF missing required tensor '{name}'.");
        var bytes = Model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    private FunAsrEncoderLayerWeights LoadEncoderLayer(string prefix, int inputDim) => new()
    {
        Norm1Weight = GetTensor($"{prefix}.norm1.weight"),
        Norm1Bias = GetTensor($"{prefix}.norm1.bias"),
        Norm2Weight = GetTensor($"{prefix}.norm2.weight"),
        Norm2Bias = GetTensor($"{prefix}.norm2.bias"),
        AttnQkvWeight = GetTensor($"{prefix}.self_attn.linear_q_k_v.weight"),
        AttnQkvBias = GetTensor($"{prefix}.self_attn.linear_q_k_v.bias"),
        AttnFsmnWeight = GetTensor($"{prefix}.self_attn.fsmn_block.weight"),
        AttnOutWeight = GetTensor($"{prefix}.self_attn.linear_out.weight"),
        AttnOutBias = GetTensor($"{prefix}.self_attn.linear_out.bias"),
        FfnW1Weight = GetTensor($"{prefix}.feed_forward.w_1.weight"),
        FfnW1Bias = GetTensor($"{prefix}.feed_forward.w_1.bias"),
        FfnW2Weight = GetTensor($"{prefix}.feed_forward.w_2.weight"),
        FfnW2Bias = GetTensor($"{prefix}.feed_forward.w_2.bias"),
        InputDim = inputDim,
    };

    private FunAsrDecoderLayerWeights LoadDecoderLayer(string prefix) => new()
    {
        Norm1Weight = GetTensor($"{prefix}.norm1.weight"),
        Norm1Bias = GetTensor($"{prefix}.norm1.bias"),
        Norm2Weight = GetTensor($"{prefix}.norm2.weight"),
        Norm2Bias = GetTensor($"{prefix}.norm2.bias"),
        Norm3Weight = GetTensor($"{prefix}.norm3.weight"),
        Norm3Bias = GetTensor($"{prefix}.norm3.bias"),
        SelfAttnFsmnWeight = GetTensor($"{prefix}.self_attn.fsmn_block.weight"),
        SrcAttnQWeight = GetTensor($"{prefix}.src_attn.linear_q.weight"),
        SrcAttnQBias = GetTensor($"{prefix}.src_attn.linear_q.bias"),
        SrcAttnKvWeight = GetTensor($"{prefix}.src_attn.linear_k_v.weight"),
        SrcAttnKvBias = GetTensor($"{prefix}.src_attn.linear_k_v.bias"),
        SrcAttnOutWeight = GetTensor($"{prefix}.src_attn.linear_out.weight"),
        SrcAttnOutBias = GetTensor($"{prefix}.src_attn.linear_out.bias"),
        FfnNormWeight = GetTensor($"{prefix}.feed_forward.norm.weight"),
        FfnNormBias = GetTensor($"{prefix}.feed_forward.norm.bias"),
        FfnW1Weight = GetTensor($"{prefix}.feed_forward.w_1.weight"),
        FfnW1Bias = GetTensor($"{prefix}.feed_forward.w_1.bias"),
        FfnW2Weight = GetTensor($"{prefix}.feed_forward.w_2.weight"),
    };

    private FunAsrDecoderFfnLayerWeights LoadDecoderFfnLayer(string prefix) => new()
    {
        Norm1Weight = GetTensor($"{prefix}.norm1.weight"),
        Norm1Bias = GetTensor($"{prefix}.norm1.bias"),
        FfnNormWeight = GetTensor($"{prefix}.feed_forward.norm.weight"),
        FfnNormBias = GetTensor($"{prefix}.feed_forward.norm.bias"),
        FfnW1Weight = GetTensor($"{prefix}.feed_forward.w_1.weight"),
        FfnW1Bias = GetTensor($"{prefix}.feed_forward.w_1.bias"),
        FfnW2Weight = GetTensor($"{prefix}.feed_forward.w_2.weight"),
    };

    public void Dispose() => Model.Dispose();
}

/// <summary>One SAN-M encoder layer's real weights (either `encoders0.0` at 560-dim input or a main `encoders.N` at 512-dim).</summary>
public sealed class FunAsrEncoderLayerWeights
{
    public required int InputDim { get; init; }
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] AttnQkvWeight { get; init; } // [inputDim, 1536]
    public required float[] AttnQkvBias { get; init; }
    public required float[] AttnFsmnWeight { get; init; } // [512, 11] depthwise conv over V
    public required float[] AttnOutWeight { get; init; } // [512, 512]
    public required float[] AttnOutBias { get; init; }
    public required float[] FfnW1Weight { get; init; } // [512, 2048]
    public required float[] FfnW1Bias { get; init; }
    public required float[] FfnW2Weight { get; init; } // [2048, 512]
    public required float[] FfnW2Bias { get; init; }
}

/// <summary>One decoder layer's real weights. self_attn is FSMN-only (no Q/K/V -- non-autoregressive design), src_attn is real cross-attention to the encoder output.</summary>
public sealed class FunAsrDecoderLayerWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] Norm3Weight { get; init; }
    public required float[] Norm3Bias { get; init; }
    public required float[] SelfAttnFsmnWeight { get; init; } // [512, 11]
    public required float[] SrcAttnQWeight { get; init; } // [512, 512]
    public required float[] SrcAttnQBias { get; init; }
    public required float[] SrcAttnKvWeight { get; init; } // [512, 1024] combined K+V
    public required float[] SrcAttnKvBias { get; init; }
    public required float[] SrcAttnOutWeight { get; init; } // [512, 512]
    public required float[] SrcAttnOutBias { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required float[] FfnNormBias { get; init; }
    public required float[] FfnW1Weight { get; init; } // [512, 2048]
    public required float[] FfnW1Bias { get; init; }
    public required float[] FfnW2Weight { get; init; } // [2048, 512], no bias per real tensor dump
}

/// <summary>`decoder.decoders3.0` -- one extra FFN-only decoder layer (no self/src attention tensors at all, confirmed against the real tensor dump).</summary>
public sealed class FunAsrDecoderFfnLayerWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required float[] FfnNormBias { get; init; }
    public required float[] FfnW1Weight { get; init; }
    public required float[] FfnW1Bias { get; init; }
    public required float[] FfnW2Weight { get; init; }
}
