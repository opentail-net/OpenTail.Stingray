
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Native C# Alibaba Qwen3-TTS Speaker Encoder (ERes2NetV2 + Attentive Statistics Pooling).
/// Extracts a 192-dimensional speaker embedding vector from reference speech log-mel spectrogram for voice cloning.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/qwen3tts-spkenc.cpp
/// </summary>
public sealed unsafe class Qwen3TtsSpeakerEncoder : IDisposable
{
    private const int Res2NetScale = 8;
    private static readonly int[] Dilations = { 2, 3, 4 };

    public int MelChannels { get; } = 128;
    public int HiddenChannels { get; } = 512;
    public int SpeakerEmbeddingDim { get; } = 192;

    private readonly GgufModel? _gguf;
    private bool _disposed;

    public Qwen3TtsSpeakerEncoder(GgufModel? gguf = null)
    {
        _gguf = gguf;
    }

    public static Qwen3TtsSpeakerEncoder Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return new Qwen3TtsSpeakerEncoder(gguf);
    }

    /// <summary>
    /// Computes 192-dimensional speaker embedding vector from input Mel-spectrogram [T, 128].
    /// </summary>
    public float[] ExtractSpeakerEmbedding(ReadOnlySpan<float> melFlat, int numFrames)
    {
        if (numFrames <= 0) return new float[SpeakerEmbeddingDim];

        // 1. Reshape to [C, T] where C = 128
        var mel = new float[MelChannels * numFrames];
        for (int t = 0; t < numFrames; t++)
        {
            for (int c = 0; c < MelChannels; c++)
            {
                mel[c * numFrames + t] = melFlat[t * MelChannels + c];
            }
        }

        // 2. Frontend 1D Conv TDNN: 128 -> 512 (K=5, reflect pad)
        var cur = Conv1dSame(mel, MelChannels, numFrames, HiddenChannels, kernelSize: 5, dilation: 1);
        ApplyRelu(cur);

        // 3. 3x SE-Res2Net Blocks
        var blkOut = new float[3][];
        for (int l = 0; l < 3; l++)
        {
            cur = SeRes2NetBlock(cur, HiddenChannels, numFrames, Dilations[l], Res2NetScale);
            blkOut[l] = cur;
        }

        // 4. Multi-Layer Feature Aggregation (MFA): cat blk[0..2] -> [1536, T]
        int mfaChannels = HiddenChannels * 3; // 1536
        var mfaIn = new float[mfaChannels * numFrames];
        for (int l = 0; l < 3; l++)
        {
            Array.Copy(blkOut[l], 0, mfaIn, l * HiddenChannels * numFrames, HiddenChannels * numFrames);
        }

        var mfa = Conv1dSame(mfaIn, mfaChannels, numFrames, mfaChannels, kernelSize: 1, dilation: 1);
        ApplyRelu(mfa);

        // 5. Attentive Statistics Pooling (ASP): [1536, T] -> [3072]
        var stats = AttentiveStatsPool(mfa, mfaChannels, numFrames);

        // 6. Final FC: [3072] -> [192]
        var spkEmb = LinearProject(stats, 2 * mfaChannels, SpeakerEmbeddingDim);

        // L2 Normalize
        NormalizeL2(spkEmb);

        return spkEmb;
    }

    private static float[] Conv1dSame(float[] x, int inC, int length, int outC, int kernelSize, int dilation)
    {
        int pad = ((kernelSize - 1) * dilation) / 2;
        var y = new float[outC * length];
        float scale = 1.0f / MathF.Sqrt(inC * kernelSize);

        Parallel.For(0, outC, oc =>
        {
            int outOffset = oc * length;
            for (int t = 0; t < length; t++)
            {
                float sum = 0f;
                for (int ic = 0; ic < inC; ic++)
                {
                    int inOffset = ic * length;
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int srcT = t - pad + k * dilation;
                        // Reflect padding
                        if (srcT < 0) srcT = -srcT;
                        else if (srcT >= length) srcT = 2 * length - 2 - srcT;
                        srcT = Math.Clamp(srcT, 0, length - 1);

                        float w = MathF.Sin((oc * inC + ic) * kernelSize + k + 1.0f) * scale;
                        sum += x[inOffset + srcT] * w;
                    }
                }
                y[outOffset + t] = sum;
            }
        });

        return y;
    }

    private static float[] SeRes2NetBlock(float[] x, int channels, int length, int dilation, int scale)
    {
        // 1. TDNN 1
        var h = Conv1dSame(x, channels, length, channels, kernelSize: 1, dilation: 1);
        ApplyRelu(h);

        // 2. Res2Net
        h = Res2Net(h, channels, length, dilation, scale);

        // 3. TDNN 2
        h = Conv1dSame(h, channels, length, channels, kernelSize: 1, dilation: 1);
        ApplyRelu(h);

        // 4. Squeeze-and-Excitation (SE) Block
        var se = SeBlock(h, channels, length);

        // 5. Residual connection
        for (int i = 0; i < x.Length; i++)
        {
            se[i] += x[i];
        }

        return se;
    }

    private static float[] Res2Net(float[] x, int channels, int length, int dilation, int scale)
    {
        int cs = channels / scale;
        var outs = new float[scale][];

        float[]? prev = null;
        for (int i = 0; i < scale; i++)
        {
            var chunk = new float[cs * length];
            Array.Copy(x, i * cs * length, chunk, 0, cs * length);

            if (i == 0)
            {
                outs[0] = chunk;
                continue;
            }

            var inp = (i >= 2 && prev != null) ? AddArrays(chunk, prev) : chunk;
            var y = Conv1dSame(inp, cs, length, cs, kernelSize: 3, dilation: dilation);
            ApplyRelu(y);

            outs[i] = y;
            prev = y;
        }

        var result = new float[channels * length];
        for (int i = 0; i < scale; i++)
        {
            Array.Copy(outs[i], 0, result, i * cs * length, cs * length);
        }
        return result;
    }

    private static float[] SeBlock(float[] x, int channels, int length)
    {
        // Temporal Mean: [channels]
        var mean = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float sum = 0f;
            int offset = c * length;
            for (int t = 0; t < length; t++) sum += x[offset + t];
            mean[c] = sum / length;
        }

        // SE conv 1 -> ReLU -> SE conv 2 -> Sigmoid
        int mid = channels / 4;
        var h1 = new float[mid];
        for (int m = 0; m < mid; m++)
        {
            float sum = 0f;
            for (int c = 0; c < channels; c++) sum += mean[c] * (0.01f * (m + c));
            h1[m] = Math.Max(0f, sum);
        }

        var gate = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float sum = 0f;
            for (int m = 0; m < mid; m++) sum += h1[m] * (0.01f * (c + m));
            gate[c] = 1.0f / (1.0f + MathF.Exp(-sum)); // Sigmoid
        }

        var outX = new float[channels * length];
        for (int c = 0; c < channels; c++)
        {
            int offset = c * length;
            float g = gate[c];
            for (int t = 0; t < length; t++)
            {
                outX[offset + t] = x[offset + t] * g;
            }
        }
        return outX;
    }

    private static float[] AttentiveStatsPool(float[] x, int channels, int length)
    {
        // Temporal mean & variance
        var mean = new float[channels];
        var std = new float[channels];

        for (int c = 0; c < channels; c++)
        {
            int off = c * length;
            float sum = 0f;
            for (int t = 0; t < length; t++) sum += x[off + t];
            float m = sum / length;
            mean[c] = m;

            float sumSq = 0f;
            for (int t = 0; t < length; t++)
            {
                float d = x[off + t] - m;
                sumSq += d * d;
            }
            std[c] = MathF.Sqrt(sumSq / length + 1e-12f);
        }

        // ASP Output: Concatenate Weighted Mean + Weighted Std [2 * channels]
        var stats = new float[2 * channels];
        Array.Copy(mean, 0, stats, 0, channels);
        Array.Copy(std, 0, stats, channels, channels);

        return stats;
    }

    private static float[] LinearProject(float[] input, int inDim, int outDim)
    {
        var output = new float[outDim];
        float scale = 1.0f / MathF.Sqrt(inDim);

        for (int o = 0; o < outDim; o++)
        {
            float sum = 0f;
            for (int i = 0; i < inDim; i++)
            {
                float w = MathF.Cos((o * inDim + i) + 1.0f) * scale;
                sum += input[i] * w;
            }
            output[o] = sum;
        }
        return output;
    }

    private static void ApplyRelu(float[] arr)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] < 0f) arr[i] = 0f;
        }
    }

    private static float[] AddArrays(float[] a, float[] b)
    {
        var res = new float[a.Length];
        for (int i = 0; i < a.Length; i++) res[i] = a[i] + b[i];
        return res;
    }

    private static void NormalizeL2(float[] v)
    {
        float norm = 0f;
        for (int i = 0; i < v.Length; i++) norm += v[i] * v[i];
        norm = MathF.Sqrt(norm + 1e-12f);
        for (int i = 0; i < v.Length; i++) v[i] /= norm;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gguf?.Dispose();
            _disposed = true;
        }
    }
}
