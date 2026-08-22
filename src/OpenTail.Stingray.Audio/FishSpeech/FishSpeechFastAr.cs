using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

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
        return LinearNoBias(normedLast, w.FastOutputWeight, dim, w.CodebookSize);
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
            var qkv = LinearNoBias(normed[i], lw.WqkvWeight, dim, qSize + 2 * kvSize);
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
        for (int i = 0; i < t; i++) attnOut[i] = LinearNoBias(context[i], lw.WoWeight, qSize, dim);

        var h1 = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d];
            h1[i] = row;
        }

        var ffnNormed = new float[t][];
        for (int i = 0; i < t; i++) ffnNormed[i] = RmsNorm(h1[i], lw.FfnNormWeight, w.FastRmsNormEps);

        int ffnDim = lw.W1Weight.Length / dim;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var gate = LinearNoBias(ffnNormed[i], lw.W1Weight, dim, ffnDim);
            var up = LinearNoBias(ffnNormed[i], lw.W3Weight, dim, ffnDim);
            for (int d = 0; d < ffnDim; d++) gate[d] = Silu(gate[d]) * up[d];
            var ffnOut = LinearNoBias(gate, lw.W2Weight, ffnDim, dim);

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
    private static void ApplyRope(float[] vec, int nHeads, int headDim, int position, float freqBase)
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

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    private static float[] RmsNorm(float[] x, float[] weight, float eps)
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
