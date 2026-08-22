using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Vad;

/// <summary>
/// Real weight loader for Silero VAD's 16kHz code path, extracted directly from
/// `models/silero_vad.onnx` (NOT the messy auto-converted `.gguf` -- see docs/audio-review-
/// progress.md's Silero VAD section for why). Confirmed this session by walking the actual
/// ONNX graph with Python's `onnx` package (not guessed, not inferred from `SileroVad.cs`'s
/// old comments which described the WRONG architecture): the top-level graph is just
/// `Equal(sr, 16000) -> If -> Identity`, with the entire real model living inside the `If`
/// node's `then_branch` subgraph (54 nodes) for 16kHz input (the `else_branch`, a separate
/// 8kHz code path, is not used by this codebase -- every pipeline here runs 16kHz).
///
/// Real 16kHz forward pass, confirmed node-by-node:
/// 1. `padded = ReflectPad(input, pad_end=64)` -- pads ONLY the end of the raw waveform by 64
///    samples, reflect mode (NOT symmetric head+tail padding like the old fake implementation
///    assumed).
/// 2. `stft = Conv1d(padded.unsqueeze(1), stft.forward_basis_buffer[258,1,256], stride=128,
///    pad=0)` -- a LEARNED STFT frontend (not a hand-rolled Hann-window DFT): 258 output
///    channels = 129 real + 129 imaginary DFT coefficients (channels [0,129) and [129,258)
///    respectively, confirmed via the real Slice node ranges).
/// 3. `mag = sqrt(stft[0:129]^2 + stft[129:258]^2)` -- magnitude spectrogram, [129, T'].
/// 4. 4x reparam_conv (Silero's newer FUSED conv architecture -- one Conv+bias per stage, not
///    separate conv+norm+activation): `ReLU(Conv1d(mag, encoder.0, k=3,s=1,p=1))` (129->128) ->
///    `ReLU(Conv1d(_, encoder.1, k=3,s=2,p=1))` (128->64) -> `ReLU(Conv1d(_, encoder.2,
///    k=3,s=2,p=1))` (64->64) -> `ReLU(Conv1d(_, encoder.3, k=3,s=1,p=1))` (64->128).
/// 5. A standard single-layer ONNX LSTM (hidden_size=128) over the encoder output (time-major),
///    seeded from the real `state` input (h0=state[0], c0=state[1]) -- weight/recurrence/bias
///    tensors are auto-generated-named (`Unsqueeze_N_output_0`) by the ONNX exporter's constant
///    folding rather than clean module-path names, but resolve to real, correctly-shaped
///    values (confirmed via <see cref="OnnxModel.Initializers"/> lookup, not guessed).
/// 6. `p = Sigmoid(Conv1d(lstm_output, decoder.decoder.2[1,128,1], k=1))`, then mean over time
///    if more than one output frame exists (this codebase always processes one 512-sample
///    frame -> one output frame at a time, so the mean is a no-op in practice, but is included
///    for correctness against arbitrary-length input).
/// </summary>
public sealed class SileroVadWeights
{
    public const int SttStride = 128;
    public const int SttKernel = 256;
    public const int PadEnd = 64;
    public const int NumFreqBins = 129; // SttKernel/2 + 1
    public const int HiddenDim = 128;

    public float[] StftBasis { get; } // [258, 1, 256] -- row-major, real rows [0,129), imag rows [129,258)

    public float[] Encoder0Weight { get; } // [128, 129, 3]
    public float[] Encoder0Bias { get; }   // [128]
    public float[] Encoder1Weight { get; } // [64, 128, 3]
    public float[] Encoder1Bias { get; }   // [64]
    public float[] Encoder2Weight { get; } // [64, 64, 3]
    public float[] Encoder2Bias { get; }   // [64]
    public float[] Encoder3Weight { get; } // [128, 64, 3]
    public float[] Encoder3Bias { get; }   // [128]

    /// <summary>ONNX LSTM convention: [1, 4*hidden, input] gate order i,o,f,c (ONNX spec order, NOT PyTorch's i,f,g,o -- confirmed against the real exported tensor, do not assume PyTorch order when porting the gate math).</summary>
    public float[] LstmWih { get; } // [1, 512, 128]
    public float[] LstmWhh { get; } // [1, 512, 128]
    /// <summary>ONNX LSTM bias convention: [1, 8*hidden] = concat(Wb[4*hidden], Rb[4*hidden]); this checkpoint's real bias initializer is only [1,1024] = 8*128, i.e. Wb and Rb ARE both present and must be summed per-gate (not just Wb alone).</summary>
    public float[] LstmBias { get; } // [1, 1024]

    public float[] DecoderWeight { get; } // [1, 128, 1]
    public float[] DecoderBias { get; }   // [1]

    public SileroVadWeights(string onnxPath)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException($"Silero VAD ONNX model not found: {onnxPath}");

        var model = OnnxModel.Open(onnxPath);
        const string p = "If_0_then_branch__Inline_0__";

        StftBasis = OnnxModel.ToFloat32(model.GetTensor($"{p}stft.forward_basis_buffer"));

        Encoder0Weight = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.0.reparam_conv.weight"));
        Encoder0Bias = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.0.reparam_conv.bias"));
        Encoder1Weight = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.1.reparam_conv.weight"));
        Encoder1Bias = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.1.reparam_conv.bias"));
        Encoder2Weight = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.2.reparam_conv.weight"));
        Encoder2Bias = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.2.reparam_conv.bias"));
        Encoder3Weight = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.3.reparam_conv.weight"));
        Encoder3Bias = OnnxModel.ToFloat32(model.GetTensor($"{p}encoder.3.reparam_conv.bias"));

        // Auto-generated names, confirmed via direct graph inspection (see class doc comment)
        // rather than assumed from any naming convention.
        LstmWih = OnnxModel.ToFloat32(model.GetTensor($"{p}/Unsqueeze_7_output_0_subg_96_sub_graph2"));
        LstmWhh = OnnxModel.ToFloat32(model.GetTensor($"{p}/Unsqueeze_8_output_0_subg_96_sub_graph2"));
        LstmBias = OnnxModel.ToFloat32(model.GetTensor($"{p}/Unsqueeze_9_output_0_subg_96_sub_graph2"));

        DecoderWeight = OnnxModel.ToFloat32(model.GetTensor($"{p}decoder.decoder.2.weight"));
        DecoderBias = OnnxModel.ToFloat32(model.GetTensor($"{p}decoder.decoder.2.bias"));
    }
}
