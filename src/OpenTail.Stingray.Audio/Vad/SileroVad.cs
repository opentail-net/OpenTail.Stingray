using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Vad;

/// <summary>
/// Native managed C# port of Silero VAD's real 16kHz forward pass. Real weights are loaded from
/// `models/silero_vad.onnx` via <see cref="SileroVadWeights"/> -- see that class's doc comment
/// for the full real architecture (learned-STFT conv -> magnitude -> 4x fused reparam_conv ->
/// LSTM -> decoder conv -> sigmoid), decoded directly from the real ONNX graph this session
/// (see docs/audio-review-progress.md's Silero VAD section), not guessed or assumed from the
/// model's public description.
///
/// Operates on 512-sample (32ms @ 16kHz) frames, matching Silero's own real chunking
/// convention: `(512 + pad_end=64 - kernel=256) / stride=128 + 1 = 4` STFT frames per chunk.
/// </summary>
public sealed class SileroVad : IVoiceActivityDetector
{
    private const int FrameSize = 512;
    private const int PaddedLen = FrameSize + SileroVadWeights.PadEnd; // 576
    private const int NumStftFrames = (PaddedLen - SileroVadWeights.SttKernel) / SileroVadWeights.SttStride + 1; // 3 -- (576-256)/128+1, floor division

    private readonly SileroVadWeights? _weights;
    private readonly float[] _hState = new float[SileroVadWeights.HiddenDim];
    private readonly float[] _cState = new float[SileroVadWeights.HiddenDim];

    public SileroVad(SileroVadWeights? weights = null)
    {
        _weights = weights;
    }

    /// <summary>Loads real Silero VAD weights directly from `models/silero_vad.onnx` (NOT the messy auto-converted .gguf -- see <see cref="SileroVadWeights"/>'s doc comment for why).</summary>
    public static SileroVad Load(string onnxPath) => new(new SileroVadWeights(onnxPath));

    /// <summary>Evaluates one 512-sample frame and returns real speech probability in [0.0, 1.0]. Requires real weights (no procedural fallback -- see docs/audio-review-progress.md).</summary>
    public float ProcessFrame(ReadOnlySpan<float> frame512)
    {
        if (_weights is null)
            throw new InvalidOperationException("SileroVad.ProcessFrame requires real SileroVadWeights -- construct via SileroVad.Load(onnxPath).");
        var w = _weights;

        // 1. Reflect-pad the END only by 64 samples (NOT symmetric head+tail -- confirmed from
        // the real ONNX Pad node's amount tensor [0,0,0,64]).
        Span<float> padded = stackalloc float[PaddedLen];
        int len = Math.Min(frame512.Length, FrameSize);
        frame512[..len].CopyTo(padded);
        if (len < FrameSize) padded.Slice(len, FrameSize - len).Clear();
        for (int i = 0; i < SileroVadWeights.PadEnd; i++)
        {
            // torch/onnx 'reflect' mode: mirror excluding the boundary sample itself.
            int srcIdx = Math.Max(0, len - 2 - i);
            padded[FrameSize + i] = padded[srcIdx];
        }

        // 2. Learned STFT: Conv1d(padded, StftBasis[258,1,256], stride=128) -> real/imag halves.
        Span<float> real = stackalloc float[SileroVadWeights.NumFreqBins * NumStftFrames];
        Span<float> imag = stackalloc float[SileroVadWeights.NumFreqBins * NumStftFrames];
        for (int f = 0; f < NumStftFrames; f++)
        {
            int start = f * SileroVadWeights.SttStride;
            for (int k = 0; k < SileroVadWeights.NumFreqBins; k++)
            {
                float sumR = 0f, sumI = 0f;
                int rBase = k * SileroVadWeights.SttKernel;
                int iBase = (SileroVadWeights.NumFreqBins + k) * SileroVadWeights.SttKernel;
                for (int n = 0; n < SileroVadWeights.SttKernel; n++)
                {
                    float s = padded[start + n];
                    sumR += s * w.StftBasis[rBase + n];
                    sumI += s * w.StftBasis[iBase + n];
                }
                real[k * NumStftFrames + f] = sumR;
                imag[k * NumStftFrames + f] = sumI;
            }
        }

        // 3. Magnitude spectrogram [129, NumStftFrames].
        Span<float> mag = stackalloc float[SileroVadWeights.NumFreqBins * NumStftFrames];
        for (int i = 0; i < mag.Length; i++)
            mag[i] = MathF.Sqrt(real[i] * real[i] + imag[i] * imag[i]);

        // 4. 4x fused reparam_conv + ReLU (channel-first [C,T] layout throughout, matching the
        // real ONNX Conv1d channel-first convention).
        Span<float> enc0 = stackalloc float[128 * NumStftFrames];
        Conv1dReluSamePad(mag, SileroVadWeights.NumFreqBins, NumStftFrames, w.Encoder0Weight, w.Encoder0Bias, 128, kernel: 3, stride: 1, enc0, out int len0);

        Span<float> enc1 = stackalloc float[64 * ((len0 - 1) / 2 + 1)];
        Conv1dReluSamePad(enc0[..(128 * len0)], 128, len0, w.Encoder1Weight, w.Encoder1Bias, 64, kernel: 3, stride: 2, enc1, out int len1);

        Span<float> enc2 = stackalloc float[64 * ((len1 - 1) / 2 + 1)];
        Conv1dReluSamePad(enc1[..(64 * len1)], 64, len1, w.Encoder2Weight, w.Encoder2Bias, 64, kernel: 3, stride: 2, enc2, out int len2);

        Span<float> enc3 = stackalloc float[128 * len2];
        Conv1dReluSamePad(enc2[..(64 * len2)], 64, len2, w.Encoder3Weight, w.Encoder3Bias, 128, kernel: 3, stride: 1, enc3, out int len3);

        // 5. Standard ONNX LSTM (single layer, hidden=128), one timestep per remaining encoder
        // frame (len3 -- with FrameSize=512 this is always 1, but the loop is general).
        // ONNX LSTM gate order is i,o,f,c (NOT PyTorch's i,f,g,o) -- see SileroVadWeights' doc
        // comment. Bias is Wb+Rb concatenated ([1,1024]=8*128): both halves are summed per gate.
        Span<float> lstmOut = stackalloc float[SileroVadWeights.HiddenDim * len3];
        Span<float> gates = stackalloc float[4 * SileroVadWeights.HiddenDim];
        for (int t = 0; t < len3; t++)
        {
            for (int g = 0; g < 4 * SileroVadWeights.HiddenDim; g++)
            {
                float val = w.LstmBias[g] + w.LstmBias[4 * SileroVadWeights.HiddenDim + g];
                for (int d = 0; d < SileroVadWeights.HiddenDim; d++)
                {
                    val += enc3[d * len3 + t] * w.LstmWih[g * SileroVadWeights.HiddenDim + d];
                    val += _hState[d] * w.LstmWhh[g * SileroVadWeights.HiddenDim + d];
                }
                gates[g] = val;
            }

            int hd = SileroVadWeights.HiddenDim;
            for (int d = 0; d < hd; d++)
            {
                float iGate = Sigmoid(gates[d]);
                float oGate = Sigmoid(gates[hd + d]);
                float fGate = Sigmoid(gates[2 * hd + d]);
                float cGate = MathF.Tanh(gates[3 * hd + d]);

                _cState[d] = fGate * _cState[d] + iGate * cGate;
                _hState[d] = oGate * MathF.Tanh(_cState[d]);
                lstmOut[d * len3 + t] = _hState[d];
            }
        }

        // 6. Decoder: ReLU(lstmOut) -> Conv1d(_, DecoderWeight[1,128,1]) -> Sigmoid -> mean over
        // time. The ReLU (real graph: Unsqueeze(hn) -> Relu_4 -> Conv_5(decoder)) was the one
        // real bug in this port -- found via bisection against a standalone-extracted real
        // onnxruntime run of just the encoder+LSTM (see docs/audio-review-progress.md's Silero
        // VAD section): both the encoder output AND the raw LSTM hidden state matched real
        // onnxruntime output exactly, but the final probability didn't until this ReLU was
        // added -- the LSTM math itself was correct all along.
        float sumProb = 0f;
        for (int t = 0; t < len3; t++)
        {
            float logit = w.DecoderBias[0];
            for (int d = 0; d < SileroVadWeights.HiddenDim; d++)
                logit += MathF.Max(0f, lstmOut[d * len3 + t]) * w.DecoderWeight[d];
            sumProb += Sigmoid(logit);
        }
        return sumProb / len3;
    }

    /// <summary>Channel-first Conv1d with explicit "same"-style padding matching the real ONNX pads=[1,1] convention (pad=1 on both sides for kernel=3), followed by ReLU.</summary>
    private static void Conv1dReluSamePad(
        ReadOnlySpan<float> input, int inChannels, int inLen,
        ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, int outChannels,
        int kernel, int stride, Span<float> output, out int outLen)
    {
        const int pad = 1; // matches the real ONNX Conv nodes' pads=[1,1] for kernel=3
        outLen = (inLen + 2 * pad - kernel) / stride + 1;
        for (int oc = 0; oc < outChannels; oc++)
        {
            float b = bias[oc];
            int wBase = oc * inChannels * kernel;
            for (int ot = 0; ot < outLen; ot++)
            {
                float acc = b;
                int inStart = ot * stride - pad;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    int inIcBase = ic * inLen;
                    for (int k = 0; k < kernel; k++)
                    {
                        int inT = inStart + k;
                        if ((uint)inT < (uint)inLen)
                            acc += input[inIcBase + inT] * weight[wIcBase + k];
                    }
                }
                output[oc * outLen + ot] = MathF.Max(0f, acc);
            }
        }
    }

    /// <summary>Scans an entire audio waveform and returns speech timestamp segments.</summary>
    public IReadOnlyList<VadSpeechSegment> DetectSegments(ReadOnlySpan<float> audio, VadParams? parameters = null)
    {
        parameters ??= new VadParams();
        if (audio.IsEmpty) return [];

        Reset();
        int totalFrames = audio.Length / FrameSize;
        if (totalFrames == 0) return [];

        float[] probs = new float[totalFrames];
        for (int f = 0; f < totalFrames; f++)
        {
            var chunk = audio.Slice(f * FrameSize, FrameSize);
            probs[f] = ProcessFrame(chunk);
        }

        return VadSegmenter.BuildSegments(probs, parameters, FrameSize);
    }

    /// <summary>Resets the internal recurrent LSTM states (hidden and cell state vectors).</summary>
    public void Reset()
    {
        Array.Clear(_hState);
        Array.Clear(_cState);
    }

    private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-Math.Clamp(x, -20f, 20f)));

    public void Dispose()
    {
    }
}
