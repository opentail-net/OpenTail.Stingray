
namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Real Fish Speech S2 Pro fast-AR (per-codebook expansion transformer), transcribed directly
/// from `examples/s2.cpp/src/s2_model.cpp`'s real `fast_decode` forward pass (lines ~1100-1255,
/// see docs/audio-review-progress.md's Fish Speech section for the full derivation -- do not
/// re-derive).
///
/// <para><b>Genuinely different from the slow-AR trunk, not reusable via `ForwardPass`</b>:
/// `fast_head_dim=128` (not `2560/32=80` -- a real, distinct per-head width), no QK-norm
/// (`fast_attention_qk_norm=false`, unlike the slow-AR's `true`), a real CAUSAL mask across
/// codebook positions (unlike the slow-AR/Orpheus's single-embedding-per-step, non-causal-
/// within-composition pattern), and a SEPARATE, NOT-tied output head
/// (`fast_tie_word_embeddings=false`). Re-run from scratch on every call (no persistent KV
/// cache) -- cheap since the sequence is at most 1 + num_codebooks-1 = 10 tokens over 4 layers.
/// </para>
///
/// <para><b>Input construction</b>: position 0 = the slow-AR's own per-position hidden state,
/// used AS-IS (`fast_project_in=false` for this checkpoint -- no projection). Positions
/// 1..len(prefix) = <see cref="FishSpeechWeights.FastEmbeddings"/>[value] for each codebook value
/// already decided THIS timestep (plain lookup by raw value, NOT offset by codebook index --
/// unlike the slow-AR's `codebook_embeddings` table).</para>
///
/// <para><b>Output</b>: final RMSNorm -> take ONLY the last position's hidden -> project through
/// `FastOutputWeight` -> `CodebookSize` logits for the NEXT codebook.</para>
/// </summary>
/// <summary>
/// Real self-attention KV cache for one slow-AR timestep's fast-AR codebook expansion. The real
/// fast-AR is causal self-attention only (no cross-attention) over at most
/// <c>1 + (NumCodebooks-1) = 10</c> positions -- growing this cache by one position per codebook
/// draw and reusing it across the 9 real per-timestep calls avoids the O(t²) full re-run
/// <see cref="FishSpeechFastAr.Forward"/> did on every single call (measured this session's
/// performance-pass baseline as the single largest real cost in the whole Fish Speech pipeline --
/// see docs/audio-review-progress.md). Must be <see cref="Reset"/> at the start of every new
/// slow-AR timestep (the fast-AR has no persistent state across timesteps in the real reference).
/// </summary>
public sealed class FishSpeechFastArCache
{
    public readonly int NumLayers;
    public readonly int KvSize;
    public readonly int MaxPositions;

    // Per-layer cached K and V: [NumLayers][MaxPositions][KvSize]
    public readonly float[][][] K;
    public readonly float[][][] V;
    public readonly int[] Counts;

    // Reusable per-step scratch workspace buffers
    public readonly float[] Normed;
    public readonly float[] Qkv;
    public readonly float[] Q;
    public readonly float[] Context;
    public readonly float[] AttnOut;
    public readonly float[] H1;
    public readonly float[] FfnNormed;
    public readonly float[] Gate;
    public readonly float[] Up;
    public readonly float[] FfnOut;
    public readonly float[] Output;
    public readonly float[] Scores;
    public readonly float[] Logits;

    public FishSpeechFastArCache(int numLayers, int dim, int qSize, int kvSize, int ffnDim, int codebookSize, int maxPositions = 16)
    {
        NumLayers = numLayers;
        KvSize = kvSize;
        MaxPositions = maxPositions;
        Counts = new int[numLayers];

        K = new float[numLayers][][];
        V = new float[numLayers][][];
        for (int l = 0; l < numLayers; l++)
        {
            K[l] = new float[maxPositions][];
            V[l] = new float[maxPositions][];
            for (int p = 0; p < maxPositions; p++)
            {
                K[l][p] = new float[kvSize];
                V[l][p] = new float[kvSize];
            }
        }

        Normed = new float[dim];
        Qkv = new float[qSize + 2 * kvSize];
        Q = new float[qSize];
        Context = new float[qSize];
        AttnOut = new float[dim];
        H1 = new float[dim];
        FfnNormed = new float[dim];
        Gate = new float[ffnDim];
        Up = new float[ffnDim];
        FfnOut = new float[dim];
        Output = new float[dim];
        Scores = new float[maxPositions];
        Logits = new float[codebookSize];
    }

    public FishSpeechFastArCache(FishSpeechWeights w, int maxPositions = 16)
        : this(w.FastBlockCount, w.FastEmbeddingDim, w.FastHeadCount * w.FastHeadDim, w.FastHeadCountKv * w.FastHeadDim, w.FastLayers[0].FfnDim, w.CodebookSize, maxPositions)
    {
    }

    public void Reset()
    {
        Array.Clear(Counts, 0, Counts.Length);
    }
}

public static class FishSpeechFastAr
{
    /// <summary>Predicts logits for the next codebook, given the slow-AR's hidden state for this timestep and the codebook values already decided so far this timestep.</summary>
    public static float[] Forward(FishSpeechWeights w, float[] slowArHidden, int[] prefixCodebookValues)
    {
        int t = 1 + prefixCodebookValues.Length;
        int dim = w.FastEmbeddingDim;

        var x = new float[t][];
        x[0] = (float[])slowArHidden.Clone();
        for (int i = 0; i < prefixCodebookValues.Length; i++)
        {
            var row = new float[dim];
            Array.Copy(w.FastEmbeddings, (long)prefixCodebookValues[i] * dim, row, 0, dim);
            x[i + 1] = row;
        }

        foreach (var layer in w.FastLayers)
            x = Layer(x, layer, w);

        var normedLast = RmsNorm(x[t - 1], w.FastNormWeight, w.FastRmsNormEps);
        return LinearQ8_0(normedLast, w.FastOutputWeight, dim, w.CodebookSize);
    }

    /// <summary>Real embedding lookup for a codebook value fed as the fast-AR's next input position (plain lookup, NOT offset by codebook index -- zero allocation span view).</summary>
    public static ReadOnlySpan<float> EmbedFastToken(FishSpeechWeights w, int value)
    {
        int dim = w.FastEmbeddingDim;
        return new ReadOnlySpan<float>(w.FastEmbeddings, value * dim, dim);
    }

    /// <summary>
    /// KV-cached single-position step, mathematically equivalent to calling
    /// <see cref="Forward"/> with a prefix one token longer each time (see
    /// <see cref="FishSpeechFastArCache"/>'s doc comment), but without redoing attention work
    /// for already-processed positions. Zero GC heap allocations via pre-allocated scratch workspace.
    /// </summary>
    public static ReadOnlySpan<float> ForwardStep(FishSpeechWeights w, FishSpeechFastArCache cache, ReadOnlySpan<float> inputVec)
    {
        ReadOnlySpan<float> x = inputVec;
        for (int i = 0; i < w.FastLayers.Length; i++)
        {
            LayerStep(x, w.FastLayers[i], w, cache, i);
            x = cache.Output;
        }

        RmsNormInPlace(cache.Output, w.FastNormWeight, w.FastRmsNormEps, cache.Normed);
        LinearQ8_0(cache.Normed, w.FastOutputWeight, w.FastEmbeddingDim, w.CodebookSize, cache.Logits);
        return cache.Logits;
    }

    private static void LayerStep(ReadOnlySpan<float> x, FishSpeechFastLayerWeights lw, FishSpeechWeights w, FishSpeechFastArCache cache, int layerIdx)
    {
        int dim = w.FastEmbeddingDim;
        int nHead = w.FastHeadCount;
        int nHeadKv = w.FastHeadCountKv;
        int headDim = w.FastHeadDim;
        int qSize = nHead * headDim;
        int kvSize = nHeadKv * headDim;

        RmsNormInPlace(x, lw.AttentionNormWeight, w.FastRmsNormEps, cache.Normed);
        LinearQ8_0(cache.Normed, lw.WqkvWeight, dim, qSize + 2 * kvSize, cache.Qkv);

        int pos = cache.Counts[layerIdx];
        var kSlot = cache.K[layerIdx][pos];
        var vSlot = cache.V[layerIdx][pos];

        Array.Copy(cache.Qkv, 0, cache.Q, 0, qSize);
        Array.Copy(cache.Qkv, qSize, kSlot, 0, kvSize);
        Array.Copy(cache.Qkv, qSize + kvSize, vSlot, 0, kvSize);

        ApplyRope(cache.Q, nHead, headDim, pos, w.FastRopeCos, w.FastRopeSin);
        ApplyRope(kSlot, nHeadKv, headDim, pos, w.FastRopeCos, w.FastRopeSin);

        cache.Counts[layerIdx]++;
        int t = cache.Counts[layerIdx];

        Array.Clear(cache.Context, 0, qSize);
        int groupSize = nHead / nHeadKv;
        float scale = 1f / MathF.Sqrt(headDim);

        var kLayer = cache.K[layerIdx];
        var vLayer = cache.V[layerIdx];

        for (int h = 0; h < nHead; h++)
        {
            int qOff = h * headDim;
            int kvOff = (h / groupSize) * headDim;
            for (int j = 0; j < t; j++)
            {
                float dot = 0f;
                var kj = kLayer[j];
                for (int d = 0; d < headDim; d++) dot += cache.Q[qOff + d] * kj[kvOff + d];
                cache.Scores[j] = dot * scale;
            }
            SoftmaxInPlace(cache.Scores, t);

            for (int j = 0; j < t; j++)
            {
                float s = cache.Scores[j];
                var vj = vLayer[j];
                for (int d = 0; d < headDim; d++)
                    cache.Context[qOff + d] += s * vj[kvOff + d];
            }
        }

        LinearQ8_0(cache.Context, lw.WoWeight, qSize, dim, cache.AttnOut);
        for (int d = 0; d < dim; d++) cache.H1[d] = x[d] + cache.AttnOut[d];

        RmsNormInPlace(cache.H1, lw.FfnNormWeight, w.FastRmsNormEps, cache.FfnNormed);
        int ffnDim = lw.FfnDim;
        LinearQ8_0(cache.FfnNormed, lw.W1Weight, dim, ffnDim, cache.Gate);
        LinearQ8_0(cache.FfnNormed, lw.W3Weight, dim, ffnDim, cache.Up);

        // Fused SwiGLU activation
        for (int d = 0; d < ffnDim; d++)
        {
            float g = cache.Gate[d];
            cache.Gate[d] = (g / (1f + MathF.Exp(-g))) * cache.Up[d];
        }

        LinearQ8_0(cache.Gate, lw.W2Weight, ffnDim, dim, cache.FfnOut);

        for (int d = 0; d < dim; d++) cache.Output[d] = cache.H1[d] + cache.FfnOut[d];
    }

    private static float[][] Layer(float[][] x, FishSpeechFastLayerWeights lw, FishSpeechWeights w)
    {
        int t = x.Length;
        int dim = w.FastEmbeddingDim;
        int nHead = w.FastHeadCount;
        int nHeadKv = w.FastHeadCountKv;
        int headDim = w.FastHeadDim;
        int qSize = nHead * headDim;
        int kvSize = nHeadKv * headDim;

        var normed = new float[t][];
        for (int i = 0; i < t; i++) normed[i] = RmsNorm(x[i], lw.AttentionNormWeight, w.FastRmsNormEps);

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var qkv = LinearQ8_0(normed[i], lw.WqkvWeight, dim, qSize + 2 * kvSize);
            q[i] = qkv.AsSpan(0, qSize).ToArray();
            k[i] = qkv.AsSpan(qSize, kvSize).ToArray();
            v[i] = qkv.AsSpan(qSize + kvSize, kvSize).ToArray();
        }

        // Real RoPE, own freq_base/context_length -- no QK-norm for this checkpoint (fast_attention_qk_norm=false).
        for (int i = 0; i < t; i++)
        {
            ApplyRope(q[i], nHead, headDim, i, w.FastRopeFreqBase);
            ApplyRope(k[i], nHeadKv, headDim, i, w.FastRopeFreqBase);
        }

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[qSize];

        int groupSize = nHead / nHeadKv;
        Parallel.For(0, nHead, h =>
        {
            int qOff = h * headDim;
            int kvOff = (h / groupSize) * headDim;
            float scale = 1f / MathF.Sqrt(headDim);
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                // Real causal mask: position i only attends to positions <= i.
                for (int j = 0; j <= i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][qOff + d] * k[j][kvOff + d];
                    scores[j] = dot * scale;
                }
                SoftmaxInPlace(scores, i + 1);

                var ctxSpan = context[i].AsSpan(qOff, headDim);
                for (int j = 0; j <= i; j++)
                    for (int d = 0; d < headDim; d++) ctxSpan[d] += scores[j] * v[j][kvOff + d];
            }
        });

        var attnOut = new float[t][];
        for (int i = 0; i < t; i++) attnOut[i] = LinearQ8_0(context[i], lw.WoWeight, qSize, dim);

        var h1 = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d];
            h1[i] = row;
        }

        var ffnNormed = new float[t][];
        for (int i = 0; i < t; i++) ffnNormed[i] = RmsNorm(h1[i], lw.FfnNormWeight, w.FastRmsNormEps);

        int ffnDim = lw.FfnDim;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var gate = LinearQ8_0(ffnNormed[i], lw.W1Weight, dim, ffnDim);
            var up = LinearQ8_0(ffnNormed[i], lw.W3Weight, dim, ffnDim);
            for (int d = 0; d < ffnDim; d++) gate[d] = Silu(gate[d]) * up[d];
            var ffnOut = LinearQ8_0(gate, lw.W2Weight, ffnDim, dim);

            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = h1[i][d] + ffnOut[d];
            output[i] = row;
        }
        return output;
    }

    /// <summary>
    /// Real RoPE convention, confirmed from `fish_speech/models/text2semantic/llama.py`'s real
    /// `apply_rotary_emb`/`precompute_freqs_cis` (fetched from the real `fishaudio/fish-speech`
    /// GitHub repo) AND independently corroborated by `s2_model.cpp`'s own `ggml_rope_ext(...,
    /// mode=0, ...)` call (ggml's `GGML_ROPE_TYPE_NORM`, NOT `GGML_ROPE_TYPE_NEOX`/mode=2) --
    /// INTERLEAVED CONSECUTIVE PAIRS `(x[2i], x[2i+1])`, the classic original-Llama/GPT-J
    /// rotation, NOT the split-half `(x[i], x[i+headDim/2])` convention (confirmed via this
    /// project's own `ModelHyperparams.IsNeoxRope`, which defaults unlisted architectures --
    /// "fish-speech" is not in the NEOX list -- to this same NORM/interleaved convention,
    /// meaning the slow-AR's `ForwardPass` reuse was ALREADY correct here without any fix
    /// needed; this fast-AR module had the wrong convention and is fixed here to match).
    /// </summary>
    /// <summary>Vectorized RoPE application using precomputed cos/sin tables (zero transcendental function calls).</summary>
    internal static void ApplyRope(float[] vec, int nHeads, int headDim, int position, float[,] cosTable, float[,] sinTable)
    {
        int half = headDim / 2;
        for (int h = 0; h < nHeads; h++)
        {
            int off = h * headDim;
            for (int i = 0; i < half; i++)
            {
                float cos = cosTable[position, i];
                float sin = sinTable[position, i];
                int idx0 = off + 2 * i;
                int idx1 = off + 2 * i + 1;
                float a = vec[idx0], b = vec[idx1];
                vec[idx0] = a * cos - b * sin;
                vec[idx1] = a * sin + b * cos;
            }
        }
    }

    /// <summary>Shared with <see cref="FishSpeechCodec"/>'s quantizer post_module transformer, same interleaved RoPE convention.</summary>
    internal static void ApplyRope(float[] vec, int nHeads, int headDim, int position, float freqBase)
    {
        int half = headDim / 2;
        for (int h = 0; h < nHeads; h++)
        {
            int off = h * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq = 1f / MathF.Pow(freqBase, 2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                int idx0 = off + 2 * i;
                int idx1 = off + 2 * i + 1;
                float a = vec[idx0], b = vec[idx1];
                vec[idx0] = a * cos - b * sin;
                vec[idx1] = a * sin + b * cos;
            }
        }
    }

    internal static float Silu(float x) => x / (1f + MathF.Exp(-x));

    internal static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    /// <summary>Real Q8_0 fused mat-vec (zero allocation into pre-allocated destination buffer).</summary>
    private static unsafe void LinearQ8_0(float[] input, byte[] weight, int inDim, int outDim, float[] output)
    {
        fixed (byte* wp = weight)
        fixed (float* xp = input, op = output)
        {
            SimdKernels.MatVecQ8_0(op, wp, xp, outDim, inDim);
        }
    }

    private static unsafe float[] LinearQ8_0(float[] input, byte[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (byte* wp = weight)
        fixed (float* xp = input, op = output)
        {
            SimdKernels.MatVecQ8_0(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    private static void RmsNormInPlace(ReadOnlySpan<float> x, float[] weight, float eps, float[] output)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    internal static float[] RmsNorm(float[] x, float[] weight, float eps)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        var output = new float[n];
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

    private static void SoftmaxInPlace(float[] scores, int count)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < count; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < count; i++) scores[i] *= invSum;
    }
}
