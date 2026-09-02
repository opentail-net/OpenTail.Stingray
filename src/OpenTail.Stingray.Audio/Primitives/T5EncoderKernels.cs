
namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared, dimension-parameterized real T5 encoder forward pass (non-gated `T5DenseActDense`
/// FFN variant: `wo(relu(wi(x)))`, one `wi` matrix -- NOT the gated `wi_0`/`wi_1` variant
/// <see cref="OpenTail.Stingray.Audio.Parler.T5Encoder"/> implements for Parler-TTS's
/// flan-t5-large). Extracted 2026-09-02 as a genuine DRY pass once TWO real, independently
/// verified callers existed with the identical non-gated algorithm and only differing dims:
/// MusicGen's bundled `t5-base` (see <see cref="OpenTail.Stingray.Audio.MusicGen.MusicGenTextEncoder"/>)
/// and AudioGen's external, frozen `t5-large` conditioner -- per CLAUDE.md rule 7, this kind of
/// extraction happens once duplication is real and verified, not speculatively ahead of a second
/// caller.
///
/// <para>Real T5-specific quirks (confirmed from the real `transformers` `modeling_t5.py`, same
/// as every other T5 port in this codebase): NO attention `1/sqrt(headDim)` scaling; `T5LayerNorm`
/// is pure RMSNorm (no bias, no mean-subtraction); the relative position bias is computed ONCE
/// from block 0's own bias table and reused/added into every subsequent layer's attention
/// scores, never recomputed per layer.</para>
/// </summary>
public sealed record T5EncoderDims(
    int DModel, int DFf, int DKv, int NumLayers, int NumHeads,
    int RelativeAttentionNumBuckets, int RelativeAttentionMaxDistance, float LayerNormEps = 1e-6f);

public sealed class NonGatedT5EncoderWeights
{
    public required float[] SharedEmbedding { get; init; } // [vocab, DModel]
    public required T5EncoderLayerWeights[] Layers { get; init; }
    public required float[] FinalLayerNormWeight { get; init; }
    public required float[] RelativeAttentionBias { get; init; } // [RelativeAttentionNumBuckets, NumHeads], block 0 only
}

public sealed class T5EncoderLayerWeights
{
    public required CfmLinearWeight SelfAttnQWeight { get; init; }
    public required CfmLinearWeight SelfAttnKWeight { get; init; }
    public required CfmLinearWeight SelfAttnVWeight { get; init; }
    public required CfmLinearWeight SelfAttnOWeight { get; init; }
    public required float[] SelfAttnLayerNormWeight { get; init; }
    public required CfmLinearWeight FfnWiWeight { get; init; }
    public required CfmLinearWeight FfnWoWeight { get; init; }
    public required float[] FfnLayerNormWeight { get; init; }
}

public static class T5EncoderKernels
{
    /// <summary>Loads a real, stock (non-fine-tuned) T5 encoder's weights from a safetensors file, optionally under a name prefix (e.g. "text_encoder." for a bundled checkpoint, "" for a stock standalone checkpoint like t5-large).</summary>
    public static NonGatedT5EncoderWeights Load(SafetensorsLoader loader, T5EncoderDims dims, string prefix)
    {
        var layers = new T5EncoderLayerWeights[dims.NumLayers];
        int qkvDim = dims.NumHeads * dims.DKv;
        for (int i = 0; i < dims.NumLayers; i++)
        {
            string p = $"{prefix}encoder.block.{i}";
            layers[i] = new T5EncoderLayerWeights
            {
                SelfAttnQWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.q.weight"), outDim: qkvDim, inDim: dims.DModel),
                SelfAttnKWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.k.weight"), outDim: qkvDim, inDim: dims.DModel),
                SelfAttnVWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.v.weight"), outDim: qkvDim, inDim: dims.DModel),
                SelfAttnOWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.o.weight"), outDim: dims.DModel, inDim: qkvDim),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.layer.0.layer_norm.weight"),
                FfnWiWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.1.DenseReluDense.wi.weight"), outDim: dims.DFf, inDim: dims.DModel),
                FfnWoWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.1.DenseReluDense.wo.weight"), outDim: dims.DModel, inDim: dims.DFf),
                FfnLayerNormWeight = loader.ReadF32($"{p}.layer.1.layer_norm.weight"),
            };
        }

        return new NonGatedT5EncoderWeights
        {
            SharedEmbedding = loader.ReadF32($"{prefix}shared.weight"),
            FinalLayerNormWeight = loader.ReadF32($"{prefix}encoder.final_layer_norm.weight"),
            RelativeAttentionBias = loader.ReadF32($"{prefix}encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            Layers = layers,
        };
    }

    /// <summary>Runs the full T5 encoder. `tokenIds` -&gt; embed -&gt; N x T5 layer (self-attn + plain ReLU FFN) -&gt; final RMSNorm. Returns [t][DModel].</summary>
    public static float[][] Forward(T5EncoderDims dims, NonGatedT5EncoderWeights w, int[] tokenIds)
    {
        int t = tokenIds.Length;
        int dim = dims.DModel;
        var x = new float[t * dim];
        for (int i = 0; i < t; i++)
            Array.Copy(w.SharedEmbedding, (long)tokenIds[i] * dim, x, (long)i * dim, dim);

        var positionBias = ComputeRelativePositionBias(dims, w, t);

        foreach (var layer in w.Layers)
            x = Layer(dims, x, layer, t, positionBias);

        var flatOut = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), w.FinalLayerNormWeight, flatOut.AsSpan(i * dim, dim), dims.LayerNormEps));

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            output[i] = new float[dim];
            Array.Copy(flatOut, i * dim, output[i], 0, dim);
        }
        return output;
    }

    private static float[] Layer(T5EncoderDims dims, float[] x, T5EncoderLayerWeights lw, int t, float[][,] positionBias)
    {
        int dim = dims.DModel;
        var normed1 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), lw.SelfAttnLayerNormWeight, normed1.AsSpan(i * dim, dim), dims.LayerNormEps));

        var attnOut = SelfAttention(dims, normed1, lw, t, positionBias);

        var afterAttn = new float[t * dim];
        TensorPrimitives.Add(x, attnOut, afterAttn);

        var normed2 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(afterAttn.AsSpan(i * dim, dim), lw.FfnLayerNormWeight, normed2.AsSpan(i * dim, dim), dims.LayerNormEps));

        var ffnOut = Ffn(dims, normed2, lw, t);

        var output = new float[t * dim];
        TensorPrimitives.Add(afterAttn, ffnOut, output);
        return output;
    }

    private static unsafe float[] SelfAttention(T5EncoderDims dims, float[] x, T5EncoderLayerWeights lw, int t, float[][,] positionBias)
    {
        int dim = dims.DModel;
        int nHeads = dims.NumHeads;
        int dKv = dims.DKv;
        int qkvDim = nHeads * dKv;

        var q = new float[t * qkvDim];
        var k = new float[t * qkvDim];
        var v = new float[t * qkvDim];
        fixed (float* xp = x, qp = q, kp = k, vp = v)
        {
            lw.SelfAttnQWeight.MatMul(xp, t, qp);
            lw.SelfAttnKWeight.MatMul(xp, t, kp);
            lw.SelfAttnVWeight.MatMul(xp, t, vp);
        }

        var context = new float[t * qkvDim];
        Parallel.For(0, nHeads, h =>
        {
            int off = h * dKv;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dKv; d++) dot += q[i * qkvDim + off + d] * k[j * qkvDim + off + d];
                    scores[j] = dot + positionBias[h][i, j]; // NO scaling -- real T5 quirk
                }
                SoftmaxInPlace(scores);

                var ctxSpan = context.AsSpan(i * qkvDim + off, dKv);
                for (int j = 0; j < t; j++)
                {
                    float s = scores[j];
                    var vSpan = v.AsSpan(j * qkvDim + off, dKv);
                    for (int d = 0; d < dKv; d++) ctxSpan[d] += s * vSpan[d];
                }
            }
        });

        var output = new float[t * dim];
        fixed (float* cp = context, op = output)
            lw.SelfAttnOWeight.MatMul(cp, t, op);
        return output;
    }

    /// <summary>Real T5DenseActDense: `wo(relu(wi(x)))`, no biases anywhere.</summary>
    private static unsafe float[] Ffn(T5EncoderDims dims, float[] x, T5EncoderLayerWeights lw, int t)
    {
        int dim = dims.DModel;
        int ff = dims.DFf;
        var hidden = new float[t * ff];
        fixed (float* xp = x, hp = hidden)
            lw.FfnWiWeight.MatMul(xp, t, hp);

        for (int i = 0; i < hidden.Length; i++)
            hidden[i] = MathF.Max(0f, hidden[i]);

        var output = new float[t * dim];
        fixed (float* hp = hidden, op = output)
            lw.FfnWoWeight.MatMul(hp, t, op);
        return output;
    }

    private static float[][,] ComputeRelativePositionBias(T5EncoderDims dims, NonGatedT5EncoderWeights w, int t)
    {
        var bias = new float[dims.NumHeads][,];
        for (int h = 0; h < dims.NumHeads; h++) bias[h] = new float[t, t];

        for (int qi = 0; qi < t; qi++)
        {
            for (int kj = 0; kj < t; kj++)
            {
                int relPos = kj - qi;
                int bucket = RelativePositionBucket(relPos, bidirectional: true, dims.RelativeAttentionNumBuckets, dims.RelativeAttentionMaxDistance);
                for (int h = 0; h < dims.NumHeads; h++)
                    bias[h][qi, kj] = w.RelativeAttentionBias[bucket * dims.NumHeads + h];
            }
        }
        return bias;
    }

    /// <summary>Real `_relative_position_bucket`, transcribed exactly from `transformers/models/t5/modeling_t5.py`.</summary>
    private static int RelativePositionBucket(int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int relativeBuckets = 0;
        if (bidirectional)
        {
            numBuckets /= 2;
            relativeBuckets += relativePosition > 0 ? numBuckets : 0;
            relativePosition = Math.Abs(relativePosition);
        }
        else
        {
            relativePosition = -Math.Min(relativePosition, 0);
        }

        int maxExact = numBuckets / 2;
        bool isSmall = relativePosition < maxExact;

        int relativePositionIfLarge = maxExact + (int)(
            MathF.Log(relativePosition / (float)maxExact)
            / MathF.Log(maxDistance / (float)maxExact)
            * (numBuckets - maxExact));
        relativePositionIfLarge = Math.Min(relativePositionIfLarge, numBuckets - 1);

        relativeBuckets += isSmall ? relativePosition : relativePositionIfLarge;
        return relativeBuckets;
    }

    private static void T5LayerNorm(ReadOnlySpan<float> x, float[] weight, Span<float> output, float eps)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }
}
