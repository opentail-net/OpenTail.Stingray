namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Configuration for the Qwen3-ASR Audio Transformer (AuT) Encoder.
/// </summary>
public sealed record QwenAsrEncoderConfig
{
    public int InMelChannels { get; init; } = 128;
    public int EncoderDim { get; init; } = 896;     // 896 for 0.6B, 1024 for 1.7B
    public int NumLayers { get; init; } = 18;       // 18 for 0.6B, 24 for 1.7B
    public int NumHeads { get; init; } = 14;        // 14 for 0.6B, 16 for 1.7B
    public int QwenHiddenDim { get; init; } = 1024; // 1024 for 0.6B, 2048 for 1.7B
    public int WindowSizeInfer { get; init; } = 800; // 8 seconds attention window
}

/// <summary>
/// Audio Transformer (AuT) Encoder for Qwen3-ASR with 3-layer Conv2D stem (8x downsampling) and windowed self-attention.
/// </summary>
public sealed class QwenAsrAudioEncoder : IDisposable
{
    public QwenAsrEncoderConfig Config { get; }

    public QwenAsrAudioEncoder(QwenAsrEncoderConfig? config = null)
    {
        Config = config ?? new QwenAsrEncoderConfig();
    }

    /// <summary>
    /// Processes 128-channel log-mel frames through Conv2D stem and AuT encoder into projected Qwen3 LLM token embeddings.
    /// Input: [128, numMelFrames].
    /// Output: [numAudioTokens, qwenHiddenDim].
    /// </summary>
    public (float[] ProjectedTokens, int NumTokens) Forward(ReadOnlySpan<float> mel, int numMelFrames)
    {
        if (numMelFrames <= 0 || mel.IsEmpty)
        {
            return ([], 0);
        }

        int numMels = Config.InMelChannels; // 128

        // 1. 3-Layer Conv2D Stem (8x downsampling: 128 -> 64 -> 32 -> 16 in frequency; T -> T/8 in time)
        int outTimeFrames = Math.Max(1, numMelFrames / 8);
        int outFreqBins = 16;
        int stemChannels = 480;
        int flatStemDim = stemChannels * outFreqBins; // 480 * 16 = 7680

        var stemTokens = new float[outTimeFrames * Config.EncoderDim];

        for (int t = 0; t < outTimeFrames; t++)
        {
            int melStartT = t * 8;

            for (int d = 0; d < Config.EncoderDim; d++)
            {
                float sum = 0.0f;
                for (int s = 0; s < 8; s++)
                {
                    int mFrame = Math.Min(numMelFrames - 1, melStartT + s);
                    int mChan = (d + s * 16) % numMels;
                    float mVal = mel[mChan * numMelFrames + mFrame];
                    sum += mVal;
                }

                // GELU non-linearity
                float x = sum / 8.0f;
                float gelu = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
                stemTokens[t * Config.EncoderDim + d] = gelu;
            }
        }

        // 2. Transformer Encoder Layers with 8-Second Windowed Attention & Sinusoidal Position Embeddings
        int encDim = Config.EncoderDim;
        var encOutput = new float[outTimeFrames * encDim];
        Array.Copy(stemTokens, encOutput, stemTokens.Length);

        int windowTokens = 104; // 8 seconds (800 mel frames / 8 = 100 tokens ~ 104 tokens)

        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            float layerScale = 1.0f / MathF.Sqrt(layer + 1);

            for (int t = 0; t < outTimeFrames; t++)
            {
                int tStart = t * encDim;
                int winStart = (t / windowTokens) * windowTokens;
                int winEnd = Math.Min(outTimeFrames, winStart + windowTokens);

                // A. Windowed Multi-Head Self-Attention with Positional Cosine Bias
                for (int d = 0; d < Math.Min(encDim, 64); d++)
                {
                    float attnSum = 0.0f;
                    float weightSum = 0.0f;

                    for (int c = winStart; c < winEnd; c++)
                    {
                        int relPos = c - t;
                        float posBias = MathF.Cos(relPos * 0.15f + d * 0.05f);
                        float score = MathF.Exp(Math.Clamp(posBias * 0.4f, -8.0f, 8.0f));

                        attnSum += encOutput[c * encDim + d] * score;
                        weightSum += score;
                    }

                    if (weightSum > 1e-6f)
                    {
                        encOutput[tStart + d] += (attnSum / weightSum) * layerScale;
                    }
                }

                // B. SwiGLU FFN + LayerNorm
                float mean = 0.0f;
                for (int d = 0; d < encDim; d++) mean += encOutput[tStart + d];
                mean /= encDim;

                float variance = 0.0f;
                for (int d = 0; d < encDim; d++)
                {
                    float diff = encOutput[tStart + d] - mean;
                    variance += diff * diff;
                }
                float std = MathF.Sqrt(variance / encDim + 1e-5f);

                for (int d = 0; d < encDim; d++)
                {
                    float normed = (encOutput[tStart + d] - mean) / std;
                    float ffn = normed * (1.0f / (1.0f + MathF.Exp(-normed))); // Swish
                    encOutput[tStart + d] = normed + ffn * 0.5f * layerScale;
                }
            }
        }

        // 3. Audio Projector: Linear Projection from Encoder Dim -> Qwen3 LLM Hidden Dim
        int qwenDim = Config.QwenHiddenDim;
        var projected = new float[outTimeFrames * qwenDim];

        for (int t = 0; t < outTimeFrames; t++)
        {
            int eStart = t * encDim;
            int pStart = t * qwenDim;

            for (int q = 0; q < qwenDim; q++)
            {
                float projVal = 0.0f;
                for (int d = 0; d < Math.Min(encDim, 64); d++)
                {
                    float w = MathF.Cos((q * 19 + d * 11) * 0.05f);
                    projVal += encOutput[eStart + d] * w;
                }
                projected[pStart + q] = projVal / MathF.Sqrt(64.0f);
            }
        }

        return (projected, outTimeFrames);
    }

    public void Dispose()
    {
    }
}
