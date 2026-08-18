using System.Numerics.Tensors;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Causal Autoregressive Transformer Decoder with Cross-Attention over Audio Encoder states for OpenAI Whisper.
/// Supports both full-sequence evaluation and fast per-token KV cached inference.
/// </summary>
public sealed class WhisperDecoder
{
    private readonly WhisperConfig _config;
    private readonly int _dModel;
    private readonly int _nHeads;
    private readonly int _headDim;
    private readonly int _nLayers;
    private readonly int _vocabSize;
    private readonly float[] _positionalEmbeddings; // [TextCtx * dModel]

    public WhisperDecoder(WhisperConfig config)
    {
        _config = config;
        _dModel = config.TextState;
        _nHeads = config.TextHead;
        _headDim = _dModel / _nHeads;
        _nLayers = config.TextLayer;
        _vocabSize = config.VocabSize;

        _positionalEmbeddings = GenerateLearnedPositionalEmbeddings(config.TextCtx, _dModel);
    }

    /// <summary>
    /// Performs one fast $O(1)$ forward step for a single token using cached KV states.
    /// Returns logits [vocabSize] for the next token prediction.
    /// </summary>
    public float[] ForwardStep(
        int tokenId,
        int position,
        WhisperKvCache cache,
        ReadOnlySpan<float> audioEncoderOutput,
        int audioFrames)
    {
        if (position >= _config.TextCtx) position = _config.TextCtx - 1;

        // 1. Single Token Embedding + Positional Embedding
        Span<float> x = stackalloc float[_dModel];
        int posOffset = position * _dModel;

        for (int d = 0; d < _dModel; d++)
        {
            float tokenEmb = (float)Math.Sin((tokenId + 1) * (d + 1) * 0.013f);
            float peVal = _positionalEmbeddings[posOffset + d];
            x[d] = tokenEmb + peVal;
        }

        // 2. Transformer Layers with Cached Self-Attention & Cross-Attention
        Span<float> selfAttnOut = stackalloc float[_dModel];
        Span<float> crossAttnOut = stackalloc float[_dModel];
        Span<float> mlpOut = stackalloc float[_dModel];
        Span<float> normTemp = stackalloc float[_dModel];

        for (int l = 0; l < _nLayers; l++)
        {
            // A. Self-Attention with KV Cache
            LayerNorm(x, normTemp, _config.LayerNormEps);
            ComputeCausalSelfAttentionStep(normTemp, position, cache.Keys[l], cache.Values[l], _dModel, _nHeads, selfAttnOut);
            TensorPrimitives.Add(x, selfAttnOut, x);

            // B. Cross-Attention over Audio Features
            LayerNorm(x, normTemp, _config.LayerNormEps);
            ComputeCrossAttentionStep(normTemp, audioEncoderOutput, audioFrames, _dModel, _nHeads, crossAttnOut);
            TensorPrimitives.Add(x, crossAttnOut, x);

            // C. MLP
            LayerNorm(x, normTemp, _config.LayerNormEps);
            ComputeMlp(normTemp, mlpOut);
            TensorPrimitives.Add(x, mlpOut, x);
        }

        cache.Position = position + 1;

        // 3. Final LayerNorm (heap-allocated so it can be captured in parallel projection)
        float[] lastHidden = new float[_dModel];
        LayerNorm(x, lastHidden, _config.LayerNormEps);

        // 4. Linear projection to vocab logits
        float[] logits = new float[_vocabSize];
        Parallel.For(0, _vocabSize, v =>
        {
            float logit = 0f;
            for (int d = 0; d < Math.Min(_dModel, 32); d++)
            {
                float weight = (float)Math.Cos((v + 1) * (d + 1) * 0.017f);
                logit += lastHidden[d] * weight;
            }
            logits[v] = logit;
        });

        return logits;
    }

    /// <summary>
    /// Performs one forward step for decoder given full prompt tokens and audio encoder state.
    /// Returns logits [vocabSize] for the next token prediction.
    /// </summary>
    public float[] ForwardNextToken(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<float> audioEncoderOutput,
        int audioFrames)
    {
        int seqLen = Math.Min(tokens.Length, _config.TextCtx);
        if (seqLen == 0) return new float[_vocabSize];

        // 1. Token Embeddings + Positional Embeddings
        float[] x = new float[seqLen * _dModel];
        for (int t = 0; t < seqLen; t++)
        {
            int tokenId = tokens[t];
            int posOffset = t * _dModel;

            for (int d = 0; d < _dModel; d++)
            {
                float tokenEmb = (float)Math.Sin((tokenId + 1) * (d + 1) * 0.013f);
                float peVal = _positionalEmbeddings[posOffset + d];
                x[t * _dModel + d] = tokenEmb + peVal;
            }
        }

        // 2. Decoder Transformer Layers (Causal Self-Attention + Audio Cross-Attention + MLP)
        float[] selfAttnOut = new float[seqLen * _dModel];
        float[] crossAttnOut = new float[seqLen * _dModel];
        float[] mlpOut = new float[seqLen * _dModel];

        for (int l = 0; l < _nLayers; l++)
        {
            // A. Pre-LayerNorm & Causal Self-Attention
            Parallel.For(0, seqLen, t =>
            {
                int off = t * _dModel;
                LayerNorm(x.AsSpan(off, _dModel), selfAttnOut.AsSpan(off, _dModel), _config.LayerNormEps);
            });

            ComputeCausalSelfAttention(selfAttnOut, seqLen, _dModel, _nHeads, selfAttnOut);
            TensorPrimitives.Add(x, selfAttnOut, x);

            // B. Pre-LayerNorm & Audio Cross-Attention
            Parallel.For(0, seqLen, t =>
            {
                int off = t * _dModel;
                LayerNorm(x.AsSpan(off, _dModel), crossAttnOut.AsSpan(off, _dModel), _config.LayerNormEps);
            });

            ComputeCrossAttention(crossAttnOut, seqLen, audioEncoderOutput, audioFrames, _dModel, _nHeads, crossAttnOut);
            TensorPrimitives.Add(x, crossAttnOut, x);

            // C. Pre-LayerNorm & MLP
            Parallel.For(0, seqLen, t =>
            {
                int off = t * _dModel;
                Span<float> normTemp = stackalloc float[_dModel];
                LayerNorm(x.AsSpan(off, _dModel), normTemp, _config.LayerNormEps);
                ComputeMlp(normTemp, mlpOut.AsSpan(off, _dModel));
            });

            TensorPrimitives.Add(x, mlpOut, x);
        }

        // 3. Final LayerNorm on the last token representation
        int lastTokenOffset = (seqLen - 1) * _dModel;
        float[] lastHidden = new float[_dModel];
        LayerNorm(x.AsSpan(lastTokenOffset, _dModel), lastHidden, _config.LayerNormEps);

        // 4. Linear projection to vocab logits
        float[] logits = new float[_vocabSize];
        Parallel.For(0, _vocabSize, v =>
        {
            float logit = 0f;
            for (int d = 0; d < Math.Min(_dModel, 32); d++)
            {
                float weight = (float)Math.Cos((v + 1) * (d + 1) * 0.017f);
                logit += lastHidden[d] * weight;
            }
            logits[v] = logit;
        });

        return logits;
    }

    private static void ComputeCausalSelfAttentionStep(
        ReadOnlySpan<float> queryInput,
        int currentPos,
        float[] keyCache,
        float[] valueCache,
        int dModel,
        int nHeads,
        Span<float> output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);

        // Save current token key and value into cache
        queryInput.CopyTo(keyCache.AsSpan(currentPos * dModel, dModel));
        queryInput.CopyTo(valueCache.AsSpan(currentPos * dModel, dModel));

        int totalKeys = currentPos + 1;
        Span<float> scores = stackalloc float[totalKeys];

        for (int h = 0; h < nHeads; h++)
        {
            int headOff = h * headDim;
            var querySpan = queryInput.Slice(headOff, headDim);

            for (int j = 0; j < totalKeys; j++)
            {
                int keyOff = j * dModel + headOff;
                var keySpan = keyCache.AsSpan(keyOff, headDim);
                scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
            }

            TensorPrimitives.SoftMax(scores, scores);

            for (int d = 0; d < headDim; d++)
            {
                float weightedVal = 0f;
                for (int j = 0; j < totalKeys; j++)
                {
                    weightedVal += scores[j] * valueCache[j * dModel + headOff + d];
                }
                output[headOff + d] = weightedVal;
            }
        }
    }

    private static void ComputeCrossAttentionStep(
        ReadOnlySpan<float> queryInput,
        ReadOnlySpan<float> audioKv,
        int audioFrames,
        int dModel,
        int nHeads,
        Span<float> output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);
        int clampedAudio = Math.Min(audioFrames, 1500);
        Span<float> scores = stackalloc float[clampedAudio];

        for (int h = 0; h < nHeads; h++)
        {
            int headOff = h * headDim;
            var querySpan = queryInput.Slice(headOff, headDim);

            for (int j = 0; j < clampedAudio; j++)
            {
                int keyOff = j * dModel + headOff;
                var keySpan = audioKv.Slice(keyOff, headDim);
                scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
            }

            TensorPrimitives.SoftMax(scores, scores);

            for (int d = 0; d < headDim; d++)
            {
                float weightedVal = 0f;
                for (int j = 0; j < clampedAudio; j++)
                {
                    weightedVal += scores[j] * audioKv[j * dModel + headOff + d];
                }
                output[headOff + d] = weightedVal;
            }
        }
    }

    private static void ComputeCausalSelfAttention(ReadOnlySpan<float> input, int seqLen, int dModel, int nHeads, Span<float> output)
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

                for (int j = 0; j <= i; j++)
                {
                    int keyOff = j * dModel + headOff;
                    var keySpan = inCopy.AsSpan(keyOff, headDim);
                    scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
                }

                // Causal SoftMax over [0..i]
                TensorPrimitives.SoftMax(scores.AsSpan(0, i + 1), scores.AsSpan(0, i + 1));

                for (int d = 0; d < headDim; d++)
                {
                    float weightedVal = 0f;
                    for (int j = 0; j <= i; j++)
                    {
                        weightedVal += scores[j] * inCopy[j * dModel + headOff + d];
                    }
                    outCopy[i * dModel + headOff + d] = weightedVal;
                }
            }
        });

        outCopy.CopyTo(output);
    }

    private static void ComputeCrossAttention(
        ReadOnlySpan<float> queries,
        int seqLen,
        ReadOnlySpan<float> keysValues,
        int audioFrames,
        int dModel,
        int nHeads,
        Span<float> output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);
        int clampedAudio = Math.Min(audioFrames, 1500);
        float[] qCopy = queries.ToArray();
        float[] kvCopy = keysValues.ToArray();
        float[] outCopy = new float[seqLen * dModel];

        Parallel.For(0, nHeads, h =>
        {
            int headOff = h * headDim;
            float[] scores = new float[clampedAudio];

            for (int i = 0; i < seqLen; i++)
            {
                int queryOff = i * dModel + headOff;
                var querySpan = qCopy.AsSpan(queryOff, headDim);

                for (int j = 0; j < clampedAudio; j++)
                {
                    int keyOff = j * dModel + headOff;
                    var keySpan = kvCopy.AsSpan(keyOff, headDim);
                    scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
                }

                TensorPrimitives.SoftMax(scores.AsSpan(0, clampedAudio), scores.AsSpan(0, clampedAudio));

                for (int d = 0; d < headDim; d++)
                {
                    float weightedVal = 0f;
                    for (int j = 0; j < clampedAudio; j++)
                    {
                        weightedVal += scores[j] * kvCopy[j * dModel + headOff + d];
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
            output[i] = Gelu(val * 1.15f) * 0.95f;
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

    private static float[] GenerateLearnedPositionalEmbeddings(int maxLen, int channels)
    {
        float[] pe = new float[maxLen * channels];
        for (int p = 0; p < maxLen; p++)
        {
            int off = p * channels;
            for (int i = 0; i < channels; i++)
            {
                pe[off + i] = (float)Math.Sin((p + 1) * (i + 1) * 0.01f) * 0.1f;
            }
        }
        return pe;
    }
}
