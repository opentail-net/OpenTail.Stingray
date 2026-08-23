using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Causal Autoregressive Transformer Decoder with Cross-Attention over Audio Encoder states for OpenAI Whisper.
/// Supports both full-sequence evaluation and fast per-token KV cached inference.
/// Runs against real weights when constructed with a <see cref="WhisperDecoderWeights"/> (parsed from a
/// whisper.cpp ggml model file via <see cref="WhisperGgmlModel"/>); otherwise falls back to a deterministic
/// placeholder weight generator that preserves output shapes for structural/unit testing without a model file.
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
    private readonly WhisperDecoderWeights? _weights;

    public WhisperDecoder(WhisperConfig config, WhisperDecoderWeights? weights = null)
    {
        _config = config;
        _dModel = config.TextState;
        _nHeads = config.TextHead;
        _headDim = _dModel / _nHeads;
        _nLayers = config.TextLayer;
        _vocabSize = config.VocabSize;
        _weights = weights;

        _positionalEmbeddings = weights?.PositionalEmbedding ?? GenerateLearnedPositionalEmbeddings(config.TextCtx, _dModel);
    }

    /// <summary>
    /// Projects the audio encoder output into per-layer cross-attention Keys/Values once and stores
    /// them on the cache, so <see cref="ForwardStep"/> doesn't re-project all audio frames every token.
    /// No-op on the placeholder (no real weights) path. Call once per utterance before decoding.
    /// </summary>
    public void PrimeCrossAttention(WhisperKvCache cache, ReadOnlySpan<float> encoderOutput, int audioFrames)
    {
        if (_weights is null) return;

        int frames = Math.Min(audioFrames, 1500);
        var crossKeys = new float[_nLayers][];
        var crossValues = new float[_nLayers][];

        for (int l = 0; l < _nLayers; l++)
        {
            var lw = _weights.Layers[l];
            crossKeys[l] = new float[frames * _dModel];
            crossValues[l] = new float[frames * _dModel];
            LinearReal(encoderOutput[..(frames * _dModel)], frames, _dModel, lw.CrossKeyWeight, null, _dModel, crossKeys[l]);
            LinearReal(encoderOutput[..(frames * _dModel)], frames, _dModel, lw.CrossValueWeight, lw.CrossValueBias, _dModel, crossValues[l]);
        }

        cache.SetCrossAttentionCache(crossKeys, crossValues, frames);
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

        if (_weights != null)
            return ForwardStepReal(tokenId, position, cache, _weights);

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

    private float[] ForwardStepReal(int tokenId, int position, WhisperKvCache cache, WhisperDecoderWeights w)
    {
        float[] x = new float[_dModel];
        int posOffset = position * _dModel;
        int tokOffset = tokenId * _dModel;
        for (int d = 0; d < _dModel; d++)
        {
            x[d] = w.TokenEmbeddingWeight[tokOffset + d] + w.PositionalEmbedding[posOffset + d];
        }

        float[] normTemp = new float[_dModel];
        float[] q = new float[_dModel];
        float[] attnRaw = new float[_dModel];
        float[] attnOut = new float[_dModel];

        for (int l = 0; l < _nLayers; l++)
        {
            var lw = w.Layers[l];

            // A. Self-Attention with KV Cache (project Q/K/V, store K/V into cache at this position)
            LayerNormAffine(x, lw.AttnLnWeight, lw.AttnLnBias, normTemp, _config.LayerNormEps);
            LinearReal(normTemp, 1, _dModel, lw.QueryWeight, lw.QueryBias, _dModel, q);
            LinearReal(normTemp, 1, _dModel, lw.KeyWeight, null, _dModel, cache.Keys[l].AsSpan(position * _dModel, _dModel));
            LinearReal(normTemp, 1, _dModel, lw.ValueWeight, lw.ValueBias, _dModel, cache.Values[l].AsSpan(position * _dModel, _dModel));

            ComputeAttentionStepFromProjected(q, position + 1, cache.Keys[l], cache.Values[l], _dModel, _nHeads, attnRaw);
            LinearReal(attnRaw, 1, _dModel, lw.OutWeight, lw.OutBias, _dModel, attnOut);
            TensorPrimitives.Add(x, attnOut, x);

            // B. Cross-Attention over precomputed audio Keys/Values
            LayerNormAffine(x, lw.CrossAttnLnWeight, lw.CrossAttnLnBias, normTemp, _config.LayerNormEps);
            LinearReal(normTemp, 1, _dModel, lw.CrossQueryWeight, lw.CrossQueryBias, _dModel, q);
            ComputeAttentionStepFromProjected(q, cache.CrossFrames, cache.CrossKeys![l], cache.CrossValues![l], _dModel, _nHeads, attnRaw);
            LinearReal(attnRaw, 1, _dModel, lw.CrossOutWeight, lw.CrossOutBias, _dModel, attnOut);
            TensorPrimitives.Add(x, attnOut, x);

            // C. MLP
            LayerNormAffine(x, lw.MlpLnWeight, lw.MlpLnBias, normTemp, _config.LayerNormEps);
            ComputeMlpReal(normTemp, lw, attnOut);
            TensorPrimitives.Add(x, attnOut, x);
        }

        cache.Position = position + 1;

        float[] lastHidden = new float[_dModel];
        LayerNormAffine(x, w.LnWeight, w.LnBias, lastHidden, _config.LayerNormEps);

        // Tied LM head: logits = lastHidden @ TokenEmbeddingWeight^T (no bias).
        float[] logits = new float[_vocabSize];
        LinearReal(lastHidden, 1, _dModel, w.TokenEmbeddingWeight, null, _vocabSize, logits);
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

        if (_weights != null)
            return ForwardNextTokenReal(tokens, seqLen, audioEncoderOutput, audioFrames, _weights);

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

    private float[] ForwardNextTokenReal(ReadOnlySpan<int> tokens, int seqLen, ReadOnlySpan<float> audioEncoderOutput, int audioFrames, WhisperDecoderWeights w)
    {
        int audioLen = Math.Min(audioFrames, 1500);

        float[] x = new float[seqLen * _dModel];
        for (int t = 0; t < seqLen; t++)
        {
            int tokOffset = tokens[t] * _dModel;
            int posOffset = t * _dModel;
            for (int d = 0; d < _dModel; d++)
                x[t * _dModel + d] = w.TokenEmbeddingWeight[tokOffset + d] + w.PositionalEmbedding[posOffset + d];
        }

        float[] normed = new float[seqLen * _dModel];
        float[] q = new float[seqLen * _dModel];
        float[] k = new float[seqLen * _dModel];
        float[] v = new float[seqLen * _dModel];
        float[] attnRaw = new float[seqLen * _dModel];
        float[] attnOut = new float[seqLen * _dModel];
        float[] crossK = new float[audioLen * _dModel];
        float[] crossV = new float[audioLen * _dModel];

        for (int l = 0; l < _nLayers; l++)
        {
            var lw = w.Layers[l];

            // A. Causal Self-Attention
            Parallel.For(0, seqLen, t => LayerNormAffine(x.AsSpan(t * _dModel, _dModel), lw.AttnLnWeight, lw.AttnLnBias, normed.AsSpan(t * _dModel, _dModel), _config.LayerNormEps));
            LinearReal(normed, seqLen, _dModel, lw.QueryWeight, lw.QueryBias, _dModel, q);
            LinearReal(normed, seqLen, _dModel, lw.KeyWeight, null, _dModel, k);
            LinearReal(normed, seqLen, _dModel, lw.ValueWeight, lw.ValueBias, _dModel, v);
            ComputeCausalSelfAttentionFromProjected(q, k, v, seqLen, _dModel, _nHeads, attnRaw);
            LinearReal(attnRaw, seqLen, _dModel, lw.OutWeight, lw.OutBias, _dModel, attnOut);
            TensorPrimitives.Add(x, attnOut, x);

            // B. Cross-Attention (project audio K/V fresh for this layer, shared across all seqLen queries)
            Parallel.For(0, seqLen, t => LayerNormAffine(x.AsSpan(t * _dModel, _dModel), lw.CrossAttnLnWeight, lw.CrossAttnLnBias, normed.AsSpan(t * _dModel, _dModel), _config.LayerNormEps));
            LinearReal(normed, seqLen, _dModel, lw.CrossQueryWeight, lw.CrossQueryBias, _dModel, q);
            LinearReal(audioEncoderOutput[..(audioLen * _dModel)], audioLen, _dModel, lw.CrossKeyWeight, null, _dModel, crossK);
            LinearReal(audioEncoderOutput[..(audioLen * _dModel)], audioLen, _dModel, lw.CrossValueWeight, lw.CrossValueBias, _dModel, crossV);
            ComputeCrossAttentionFromProjected(q, crossK, crossV, seqLen, audioLen, _dModel, _nHeads, attnRaw);
            LinearReal(attnRaw, seqLen, _dModel, lw.CrossOutWeight, lw.CrossOutBias, _dModel, attnOut);
            TensorPrimitives.Add(x, attnOut, x);

            // C. MLP
            Parallel.For(0, seqLen, t => LayerNormAffine(x.AsSpan(t * _dModel, _dModel), lw.MlpLnWeight, lw.MlpLnBias, normed.AsSpan(t * _dModel, _dModel), _config.LayerNormEps));
            ComputeMlpRealBatched(normed, seqLen, lw, attnOut);
            TensorPrimitives.Add(x, attnOut, x);
        }

        int lastOff = (seqLen - 1) * _dModel;
        float[] lastHidden = new float[_dModel];
        LayerNormAffine(x.AsSpan(lastOff, _dModel), w.LnWeight, w.LnBias, lastHidden, _config.LayerNormEps);

        float[] logits = new float[_vocabSize];
        LinearReal(lastHidden, 1, _dModel, w.TokenEmbeddingWeight, null, _vocabSize, logits);
        return logits;
    }

    /// <summary>Linear layer: output[seqLen, outDim] = input[seqLen, inDim] @ weight[outDim, inDim]^T + bias.</summary>
    private static unsafe void LinearReal(ReadOnlySpan<float> input, int seqLen, int inDim, float[] weight, float[]? bias, int outDim, Span<float> output)
    {
        // See WhisperEncoder.LinearReal's comment: MicroGemmKernel's core batches the weight
        // matrix's memory traffic across rows instead of re-streaming it once per row like
        // SimdKernels.MatMulBatchedF32 does. Matters here for PrimeCrossAttention (seqLen up to
        // 1500 audio frames) and ForwardNextTokenReal (seqLen == prompt length); ForwardStepReal's
        // seqLen==1 incremental decode calls are unaffected either way.
        fixed (float* pIn = input, pW = weight, pOut = output)
        {
            MicroGemmKernel.MatMulF32CoreOrFallback(pIn, pW, pOut, seqLen, inDim, outDim);
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

    private static void ComputeAttentionStepFromProjected(float[] query, int totalKeys, float[] keyCache, float[] valueCache, int dModel, int nHeads, Span<float> output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);
        Span<float> scores = totalKeys <= 4096 ? stackalloc float[totalKeys] : new float[totalKeys];

        for (int h = 0; h < nHeads; h++)
        {
            int headOff = h * headDim;
            var querySpan = query.AsSpan(headOff, headDim);

            for (int j = 0; j < totalKeys; j++)
            {
                var keySpan = keyCache.AsSpan(j * dModel + headOff, headDim);
                scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
            }

            TensorPrimitives.SoftMax(scores, scores);

            // j-outer/d-inner (contiguous valueCache row + TensorPrimitives.MultiplyAdd) instead
            // of d-outer/j-inner (dModel-strided scalar reads) -- see WhisperEncoder.cs's real
            // attention path for the full rationale. This is the incremental decode-step path,
            // called once per generated token, so it's the hottest of the attention variants.
            var weighted = output.Slice(headOff, headDim);
            weighted.Clear();
            for (int j = 0; j < totalKeys; j++)
            {
                var vRow = valueCache.AsSpan(j * dModel + headOff, headDim);
                TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
            }
        }
    }

    private static void ComputeCausalSelfAttentionFromProjected(float[] q, float[] k, float[] v, int seqLen, int dModel, int nHeads, float[] output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);

        Parallel.For(0, nHeads, h =>
        {
            int headOff = h * headDim;
            float[] scores = new float[seqLen];

            for (int i = 0; i < seqLen; i++)
            {
                var querySpan = q.AsSpan(i * dModel + headOff, headDim);

                for (int j = 0; j <= i; j++)
                {
                    var keySpan = k.AsSpan(j * dModel + headOff, headDim);
                    scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
                }

                TensorPrimitives.SoftMax(scores.AsSpan(0, i + 1), scores.AsSpan(0, i + 1));

                var weighted = output.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j <= i; j++)
                {
                    var vRow = v.AsSpan(j * dModel + headOff, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
                }
            }
        });
    }

    private static void ComputeCrossAttentionFromProjected(float[] q, float[] k, float[] v, int seqLen, int audioLen, int dModel, int nHeads, float[] output)
    {
        int headDim = dModel / nHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);

        Parallel.For(0, nHeads, h =>
        {
            int headOff = h * headDim;
            float[] scores = new float[audioLen];

            for (int i = 0; i < seqLen; i++)
            {
                var querySpan = q.AsSpan(i * dModel + headOff, headDim);

                for (int j = 0; j < audioLen; j++)
                {
                    var keySpan = k.AsSpan(j * dModel + headOff, headDim);
                    scores[j] = TensorPrimitives.Dot(querySpan, keySpan) * scale;
                }

                TensorPrimitives.SoftMax(scores.AsSpan(0, audioLen), scores.AsSpan(0, audioLen));

                var weighted = output.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j < audioLen; j++)
                {
                    var vRow = v.AsSpan(j * dModel + headOff, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
                }
            }
        });
    }

    private static void ComputeMlpReal(float[] input, WhisperDecoderLayerWeights lw, float[] output)
    {
        int dModel = input.Length;
        int hiddenDim = dModel * 4;
        float[] hidden = new float[hiddenDim];
        LinearReal(input, 1, dModel, lw.Mlp0Weight, lw.Mlp0Bias, hiddenDim, hidden);
        for (int i = 0; i < hidden.Length; i++) hidden[i] = Gelu(hidden[i]);
        LinearReal(hidden, 1, hiddenDim, lw.Mlp2Weight, lw.Mlp2Bias, dModel, output);
    }

    private void ComputeMlpRealBatched(float[] input, int seqLen, WhisperDecoderLayerWeights lw, float[] output)
    {
        int hiddenDim = _dModel * 4;
        float[] hidden = new float[seqLen * hiddenDim];
        LinearReal(input, seqLen, _dModel, lw.Mlp0Weight, lw.Mlp0Bias, hiddenDim, hidden);
        Parallel.For(0, hidden.Length, i => hidden[i] = Gelu(hidden[i]));
        LinearReal(hidden, seqLen, hiddenDim, lw.Mlp2Weight, lw.Mlp2Bias, _dModel, output);
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

            var weighted = output.Slice(headOff, headDim);
            weighted.Clear();
            for (int j = 0; j < totalKeys; j++)
            {
                var vRow = valueCache.AsSpan(j * dModel + headOff, headDim);
                TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
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

            var weighted = output.Slice(headOff, headDim);
            weighted.Clear();
            for (int j = 0; j < clampedAudio; j++)
            {
                var vRow = audioKv.Slice(j * dModel + headOff, headDim);
                TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
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

                var weighted = outCopy.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j <= i; j++)
                {
                    var vRow = inCopy.AsSpan(j * dModel + headOff, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, scores[j], weighted, weighted);
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

                var weighted = outCopy.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j < clampedAudio; j++)
                {
                    var vRow = kvCopy.AsSpan(j * dModel + headOff, headDim);
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
