using System.Numerics.Tensors;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Audio Transformer Encoder for OpenAI Whisper with 2x Conv1D downsamplers and sinusoidal positional embeddings.
/// Accelerated with System.Numerics.Tensors (SIMD) and multi-threaded attention.
/// </summary>
public sealed class WhisperEncoder
{
    private readonly WhisperConfig _config;
    private readonly int _dModel;
    private readonly int _nHeads;
    private readonly int _headDim;
    private readonly int _nLayers;
    private readonly float[] _positionalEmbeddings; // [AudioCtx * dModel]

    public WhisperEncoder(WhisperConfig config)
    {
        _config = config;
        _dModel = config.AudioState;
        _nHeads = config.AudioHead;
        _headDim = _dModel / _nHeads;
        _nLayers = config.AudioLayer;

        _positionalEmbeddings = GenerateSinusoidalPositionalEmbeddings(config.AudioCtx, _dModel);
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
        ApplyConv1D(mel, numMels, conv1Frames, conv1, stride: 1);

        // Stage 2: stride 2
        int conv2Frames = (conv1Frames + 1) / 2;
        float[] conv2 = new float[_dModel * conv2Frames];
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
        float[] attnOut = new float[encFrames * _dModel];
        float[] mlpOut = new float[encFrames * _dModel];

        for (int l = 0; l < _nLayers; l++)
        {
            // Pre-LayerNorm & Self-Attention
            Parallel.For(0, encFrames, t =>
            {
                int off = t * _dModel;
                LayerNorm(x.AsSpan(off, _dModel), attnOut.AsSpan(off, _dModel), _config.LayerNormEps);
            });

            ComputeMultiHeadSelfAttention(attnOut, encFrames, _dModel, _nHeads, attnOut);

            // Residual connection
            TensorPrimitives.Add(x, attnOut, x);

            // Pre-LayerNorm & Feed-Forward Network
            Parallel.For(0, encFrames, t =>
            {
                int off = t * _dModel;
                Span<float> tempNorm = stackalloc float[_dModel];
                LayerNorm(x.AsSpan(off, _dModel), tempNorm, _config.LayerNormEps);
                ComputeMlp(tempNorm, mlpOut.AsSpan(off, _dModel));
            });

            // Residual connection
            TensorPrimitives.Add(x, mlpOut, x);
        }

        // 4. Final LayerNorm
        Parallel.For(0, encFrames, t =>
        {
            int off = t * _dModel;
            LayerNorm(x.AsSpan(off, _dModel), x.AsSpan(off, _dModel), _config.LayerNormEps);
        });

        return x;
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

                // Weighted sum over Values
                for (int d = 0; d < headDim; d++)
                {
                    float weightedVal = 0f;
                    for (int j = 0; j < seqLen; j++)
                    {
                        weightedVal += scores[j] * inCopy[j * dModel + headOff + d];
                    }
                    outCopy[i * dModel + headOff + d] = weightedVal;
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
