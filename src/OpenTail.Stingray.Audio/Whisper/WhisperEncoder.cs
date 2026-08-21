using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Audio Transformer Encoder for OpenAI Whisper with 2x Conv1D downsamplers and sinusoidal positional embeddings.
/// Accelerated with System.Numerics.Tensors (SIMD) and multi-threaded attention.
/// Runs against real weights when constructed with a <see cref="WhisperEncoderWeights"/> (parsed from a
/// whisper.cpp ggml model file via <see cref="WhisperGgmlModel"/>); otherwise falls back to a deterministic
/// placeholder weight generator that preserves output shapes for structural/unit testing without a model file.
/// </summary>
public sealed class WhisperEncoder
{
    private readonly WhisperConfig _config;
    private readonly int _dModel;
    private readonly int _nHeads;
    private readonly int _headDim;
    private readonly int _nLayers;
    private readonly float[] _positionalEmbeddings; // [AudioCtx * dModel]
    private readonly WhisperEncoderWeights? _weights;

    public WhisperEncoder(WhisperConfig config, WhisperEncoderWeights? weights = null)
    {
        _config = config;
        _dModel = config.AudioState;
        _nHeads = config.AudioHead;
        _headDim = _dModel / _nHeads;
        _nLayers = config.AudioLayer;
        _weights = weights;

        _positionalEmbeddings = weights?.PositionalEmbedding ?? GenerateSinusoidalPositionalEmbeddings(config.AudioCtx, _dModel);
    }

    /// <summary>
    /// Forward pass of the audio encoder.
    /// Input: Mel spectrogram [NumMels, numFrames].
    /// Output: Hidden representation [numEncFrames, dModel].
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> mel, int numFrames)
    {
        int numMels = _config.NumMels;

        // 1. Conv1D Downsampling (2 stages)
        // Stage 1: stride 1
        int conv1Frames = numFrames;
        float[] conv1 = new float[_dModel * conv1Frames];
        if (_weights != null)
            ApplyConv1DReal(mel, numMels, conv1Frames, conv1, stride: 1, _weights.Conv1Weight, _weights.Conv1Bias);
        else
            ApplyConv1D(mel, numMels, conv1Frames, conv1, stride: 1);

        // Stage 2: stride 2
        int conv2Frames = (conv1Frames + 1) / 2;
        float[] conv2 = new float[_dModel * conv2Frames];
        if (_weights != null)
            ApplyConv1DReal(conv1, _dModel, conv2Frames, conv2, stride: 2, _weights.Conv2Weight, _weights.Conv2Bias);
        else
            ApplyConv1D(conv1, _dModel, conv2Frames, conv2, stride: 2);

        // 2. Transpose [dModel, conv2Frames] -> [conv2Frames, dModel] and add sinusoidal positional embeddings
        int encFrames = Math.Min(conv2Frames, _config.AudioCtx);
        float[] x = new float[encFrames * _dModel];

        for (int t = 0; t < encFrames; t++)
        {
            int posOffset = t * _dModel;
            for (int d = 0; d < _dModel; d++)
            {
                float convVal = conv2[d * conv2Frames + t];
                float peVal = _positionalEmbeddings[posOffset + d];
                x[t * _dModel + d] = convVal + peVal;
            }
        }

        // 3. Encoder Transformer Blocks
        float[] normed = new float[encFrames * _dModel];
        float[] attnOut = new float[encFrames * _dModel];
        float[] mlpOut = new float[encFrames * _dModel];

        for (int l = 0; l < _nLayers; l++)
        {
            var lw = _weights?.Layers[l];

            // Pre-LayerNorm & Self-Attention
            Parallel.For(0, encFrames, t =>
            {
                int off = t * _dModel;
                if (lw != null)
                    LayerNormAffine(x.AsSpan(off, _dModel), lw.AttnLnWeight, lw.AttnLnBias, normed.AsSpan(off, _dModel), _config.LayerNormEps);
                else
                    LayerNorm(x.AsSpan(off, _dModel), normed.AsSpan(off, _dModel), _config.LayerNormEps);
            });

            if (lw != null)
                ComputeMultiHeadSelfAttentionReal(normed, encFrames, _dModel, _nHeads, lw, attnOut);
            else
                ComputeMultiHeadSelfAttention(normed, encFrames, _dModel, _nHeads, attnOut);

            // Residual connection
            TensorPrimitives.Add(x, attnOut, x);

            // Pre-LayerNorm & Feed-Forward Network
            Parallel.For(0, encFrames, t =>
            {
                int off = t * _dModel;
                if (lw != null)
                    LayerNormAffine(x.AsSpan(off, _dModel), lw.MlpLnWeight, lw.MlpLnBias, normed.AsSpan(off, _dModel), _config.LayerNormEps);
                else
                    LayerNorm(x.AsSpan(off, _dModel), normed.AsSpan(off, _dModel), _config.LayerNormEps);
            });

            if (lw != null)
                ComputeMlpReal(normed, encFrames, _dModel, lw, mlpOut);
            else
            {
                Parallel.For(0, encFrames, t =>
                {
                    int off = t * _dModel;
                    ComputeMlp(normed.AsSpan(off, _dModel), mlpOut.AsSpan(off, _dModel));
                });
            }

            // Residual connection
            TensorPrimitives.Add(x, mlpOut, x);
        }

        // 4. Final LayerNorm
        Parallel.For(0, encFrames, t =>
        {
            int off = t * _dModel;
            if (_weights != null)
                LayerNormAffine(x.AsSpan(off, _dModel), _weights.LnPostWeight, _weights.LnPostBias, x.AsSpan(off, _dModel), _config.LayerNormEps);
            else
                LayerNorm(x.AsSpan(off, _dModel), x.AsSpan(off, _dModel), _config.LayerNormEps);
        });

        return x;
    }

    /// <summary>
    /// Real Conv1D (kernel=3, padding=1) against loaded weights [outCh, inCh, 3] (file storage order, k fastest).
    /// </summary>
    private void ApplyConv1DReal(ReadOnlySpan<float> input, int inChannels, int outFrames, Span<float> output, int stride, float[] weight, float[] bias)
    {
        float[] inCopy = input.ToArray();
        float[] outCopy = new float[_dModel * outFrames];
        int inFrames = inChannels > 0 ? inCopy.Length / inChannels : outFrames;

        if (stride == 1)
        {
            // Scale-and-shift-add vectorization (same as ChatterboxCfmDecoder.cs/
            // ChatterboxVocoder.cs/KokoroDecoder.cs -- see those for the full rationale): for a
            // fixed (oc,ic,k), `out[t] += w*input[t+shift]` over the valid t range is a single
            // TensorPrimitives.MultiplyAdd over a contiguous span. Only valid when stride==1
            // (output position t maps directly to a shifted input position); the stride==2 conv
            // below maps output t to input 2t+k, a strided gather that doesn't reduce to a
            // contiguous-span op the same way, so it keeps the original per-element loop --
            // called only once per transcription each, so this is a smaller win than Kokoro/
            // Chatterbox's per-layer resblock convs, but still real for longer audio.
            Parallel.For(0, _dModel, oc =>
            {
                var outRow = outCopy.AsSpan(oc * outFrames, outFrames);
                outRow.Fill(bias[oc]);
                int wOcOff = oc * inChannels * 3;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    var inRow = inCopy.AsSpan(ic * inFrames, inFrames);
                    int wIcOff = wOcOff + ic * 3;
                    for (int k = -1; k <= 1; k++)
                        AxpyShifted(inRow, weight[wIcOff + (k + 1)], outRow, k, outFrames);
                }
                for (int t = 0; t < outFrames; t++) outRow[t] = Gelu(outRow[t]);
            });
        }
        else
        {
            Parallel.For(0, _dModel, oc =>
            {
                int outChannelOff = oc * outFrames;
                int wOcOff = oc * inChannels * 3;
                for (int t = 0; t < outFrames; t++)
                {
                    int inCenter = t * stride;
                    float sum = bias[oc];

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        int inChannelOff = ic * inFrames;
                        int wIcOff = wOcOff + ic * 3;
                        for (int k = -1; k <= 1; k++)
                        {
                            int srcT = inCenter + k;
                            if (srcT >= 0 && srcT < inFrames)
                            {
                                sum += inCopy[inChannelOff + srcT] * weight[wIcOff + (k + 1)];
                            }
                        }
                    }

                    outCopy[outChannelOff + t] = Gelu(sum);
                }
            });
        }

        outCopy.CopyTo(output);
    }

    /// <summary>output[t] += scale * input[t + shift] for every t where t+shift is in [0, input.Length).</summary>
    private static void AxpyShifted(ReadOnlySpan<float> input, float scale, Span<float> output, int shift, int outLen)
    {
        int inLen = input.Length;
        int start = Math.Max(0, -shift);
        int end = Math.Min(outLen, inLen - shift);
        int len = end - start;
        if (len <= 0) return;
        var inSlice = input.Slice(start + shift, len);
        var outSlice = output.Slice(start, len);
        TensorPrimitives.MultiplyAdd(inSlice, scale, outSlice, outSlice);
    }

    /// <summary>Linear layer: output[seqLen, outDim] = input[seqLen, inDim] @ weight[outDim, inDim]^T + bias.</summary>
    private static unsafe void LinearReal(ReadOnlySpan<float> input, int seqLen, int inDim, float[] weight, float[]? bias, int outDim, Span<float> output)
    {
        fixed (float* pIn = input, pW = weight, pOut = output)
        {
            SimdKernels.MatMulBatchedF32(pOut, pW, pIn, seqLen, outDim, inDim);
        }

        if (bias != null)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var row = output.Slice(t * outDim, outDim);
                TensorPrimitives.Add(row, bias, row);
            }
        }
    }

    private static void ComputeMultiHeadSelfAttentionReal(float[] input, int seqLen, int dModel, int nHeads, WhisperEncoderLayerWeights lw, Span<float> output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);

        float[] q = new float[seqLen * dModel];
        float[] k = new float[seqLen * dModel];
        float[] v = new float[seqLen * dModel];
        LinearReal(input, seqLen, dModel, lw.QueryWeight, lw.QueryBias, dModel, q);
        LinearReal(input, seqLen, dModel, lw.KeyWeight, null, dModel, k);
        LinearReal(input, seqLen, dModel, lw.ValueWeight, lw.ValueBias, dModel, v);

        float[] attnRaw = new float[seqLen * dModel];

        Parallel.For(0, nHeads, h =>
        {
            int headOff = h * headDim;
            float[] scores = new float[seqLen];

            for (int i = 0; i < seqLen; i++)
            {
                var querySpan = q.AsSpan(i * dModel + headOff, headDim);

                for (int j = 0; j < seqLen; j++)
                {
                    var keySpan = k.AsSpan(j * dModel + headOff, headDim);
                    scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
                }

                TensorPrimitives.SoftMax(scores.AsSpan(0, seqLen), scores.AsSpan(0, seqLen));

                // weighted = sum_j scores[j] * v[j, headOff:headOff+headDim]. Looping j-outer/
                // d-inner (rather than the reverse) keeps each v access a contiguous headDim-wide
                // span instead of a dModel-strided one, and lets TensorPrimitives vectorize the
                // per-row multiply-accumulate instead of doing headDim*seqLen scalar mults.
                var weighted = attnRaw.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j < seqLen; j++)
                {
                    var vRow = v.AsSpan(j * dModel + headOff, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
                }
            }
        });

        LinearReal(attnRaw, seqLen, dModel, lw.OutWeight, lw.OutBias, dModel, output);
    }

    private static void ComputeMlpReal(float[] input, int seqLen, int dModel, WhisperEncoderLayerWeights lw, Span<float> output)
    {
        int hiddenDim = dModel * 4;
        float[] hidden = new float[seqLen * hiddenDim];
        LinearReal(input, seqLen, dModel, lw.Mlp0Weight, lw.Mlp0Bias, hiddenDim, hidden);

        Parallel.For(0, hidden.Length, i => hidden[i] = Gelu(hidden[i]));

        LinearReal(hidden, seqLen, hiddenDim, lw.Mlp2Weight, lw.Mlp2Bias, dModel, output);
    }

    private static void LayerNormAffine(ReadOnlySpan<float> input, float[] weight, float[] bias, Span<float> output, float eps)
    {
        int n = input.Length;
        float mean = TensorPrimitives.Sum(input) / n;

        float variance = 0f;
        for (int i = 0; i < n; i++)
        {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= n;

        float invStd = 1.0f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < n; i++)
        {
            output[i] = (input[i] - mean) * invStd * weight[i] + bias[i];
        }
    }

    private void ApplyConv1D(ReadOnlySpan<float> input, int inChannels, int outFrames, Span<float> output, int stride)
    {
        float[] inCopy = input.ToArray();
        float[] outCopy = new float[_dModel * outFrames];
        int inFrames = inChannels > 0 ? inCopy.Length / inChannels : outFrames;

        Parallel.For(0, _dModel, oc =>
        {
            int outChannelOff = oc * outFrames;
            for (int t = 0; t < outFrames; t++)
            {
                int inCenter = t * stride;
                float sum = 0f;

                for (int ic = 0; ic < inChannels; ic++)
                {
                    int inChannelOff = ic * inFrames;
                    for (int k = -1; k <= 1; k++)
                    {
                        int srcT = inCenter + k;
                        if (srcT >= 0 && srcT < inFrames)
                        {
                            int idx = inChannelOff + srcT;
                            if (idx >= 0 && idx < inCopy.Length)
                            {
                                float inVal = inCopy[idx];
                                float weight = (float)Math.Sin((oc + 1) * (ic + 1) + k) * 0.05f;
                                sum += inVal * weight;
                            }
                        }
                    }
                }

                outCopy[outChannelOff + t] = Gelu(sum);
            }
        });

        outCopy.CopyTo(output);
    }

    private static void ComputeMultiHeadSelfAttention(ReadOnlySpan<float> input, int seqLen, int dModel, int nHeads, Span<float> output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);
        float[] inCopy = input.ToArray();
        float[] outCopy = new float[seqLen * dModel];

        Parallel.For(0, nHeads, h =>
        {
            int headOff = h * headDim;
            float[] scores = new float[seqLen];

            for (int i = 0; i < seqLen; i++)
            {
                int queryOff = i * dModel + headOff;
                var querySpan = inCopy.AsSpan(queryOff, headDim);

                for (int j = 0; j < seqLen; j++)
                {
                    int keyOff = j * dModel + headOff;
                    var keySpan = inCopy.AsSpan(keyOff, headDim);
                    scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
                }

                // SoftMax
                TensorPrimitives.SoftMax(scores.AsSpan(0, seqLen), scores.AsSpan(0, seqLen));

                // Weighted sum over Values (see the real-weights path above for why j-outer/
                // d-inner with TensorPrimitives.MultiplyAdd beats the reverse loop order).
                var weighted = outCopy.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j < seqLen; j++)
                {
                    var vRow = inCopy.AsSpan(j * dModel + headOff, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
                }
            }
        });

        outCopy.CopyTo(output);
    }

    private static void ComputeMlp(ReadOnlySpan<float> input, Span<float> output)
    {
        int dModel = input.Length;
        for (int i = 0; i < dModel; i++)
        {
            float val = input[i];
            output[i] = Gelu(val * 1.2f) * 0.9f;
        }
    }

    private static void LayerNorm(ReadOnlySpan<float> input, Span<float> output, float eps)
    {
        int n = input.Length;
        float mean = TensorPrimitives.Sum(input) / n;

        float variance = 0f;
        for (int i = 0; i < n; i++)
        {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= n;

        float invStd = 1.0f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < n; i++)
        {
            output[i] = (input[i] - mean) * invStd;
        }
    }

    private static float Gelu(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    private static float[] GenerateSinusoidalPositionalEmbeddings(int length, int channels)
    {
        float[] pe = new float[length * channels];
        float logTimescale = MathF.Log(10000.0f) / (channels / 2 - 1);

        for (int p = 0; p < length; p++)
        {
            int offset = p * channels;
            for (int i = 0; i < channels / 2; i++)
            {
                float invFreq = MathF.Exp(-i * logTimescale);
                float angle = p * invFreq;
                pe[offset + 2 * i] = MathF.Sin(angle);
                pe[offset + 2 * i + 1] = MathF.Cos(angle);
            }
        }

        return pe;
    }
}
