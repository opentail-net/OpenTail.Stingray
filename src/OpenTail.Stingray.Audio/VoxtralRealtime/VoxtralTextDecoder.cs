
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real Voxtral Realtime text-decoder forward pass -- see
/// <see cref="VoxtralTextDecoderWeights"/>'s doc comment for the architecture derivation. A
/// standard Mistral GQA decoder (RoPE-NEOX, no QK-norm, SwiGLU, RMSNorm) plus a real
/// AdaLN-Zero-style delay-conditioning gate on the MLP branch, and a real ADDITIVE (not
/// row-replacement) audio-embedding splice.
/// </summary>
public static class VoxtralTextDecoder
{
    /// <summary>Runs a full (non-streaming) prefill over <paramref name="tokenIds"/> with
    /// <paramref name="audioEmbeddings"/> ([tokens][3072]) added into the first
    /// <c>audioEmbeddings.Length</c> positions, and returns per-position logits
    /// [tokenIds.Length][VocabSize].</summary>
    public static float[][] Forward(VoxtralTextDecoderWeights w, int[] tokenIds, float[][] audioEmbeddings, int numDelayTokens)
    {
        int hidden = VoxtralTextDecoderWeights.HiddenSize;
        int n = tokenIds.Length;
        var x = new float[n][];
        for (int t = 0; t < n; t++)
        {
            var row = new float[hidden];
            int embedBase = tokenIds[t] * hidden;
            for (int d = 0; d < hidden; d++) row[d] = w.EmbedTokensWeight[embedBase + d];
            if (t < audioEmbeddings.Length)
            {
                var audio = audioEmbeddings[t];
                for (int d = 0; d < hidden; d++) row[d] += audio[d];
            }
            x[t] = row;
        }

        var tCond = TimeEmbedding(numDelayTokens, hidden);

        foreach (var layer in w.Layers)
        {
            var attnIn = RmsNormRows(x, hidden, layer.InputNorm);
            var attn = SelfAttention(attnIn, hidden, layer);
            AddRows(x, attn);

            var mlpIn = RmsNormRows(x, hidden, layer.PostNorm);
            var scale = AdaGate(tCond, layer);
            for (int t = 0; t < n; t++)
                for (int d = 0; d < hidden; d++)
                    mlpIn[t][d] *= scale[d];
            var mlp = Mlp(mlpIn, hidden, layer);
            AddRows(x, mlp);
        }

        x = RmsNormRows(x, hidden, w.NormWeight);

        var logits = new float[n][];
        for (int t = 0; t < n; t++)
            logits[t] = LinearNoBias(x[t], w.EmbedTokensWeightQ8_0, hidden, VoxtralTextDecoderWeights.VocabSize);
        return logits;
    }

    /// <summary>Real sinusoidal timestep embedding, cos/sin halves, base 10000 -- matches the
    /// reference's own `time_embedding` exactly, computed once per request (not per-position).
    /// </summary>
    private static float[] TimeEmbedding(int numDelayTokens, int hidden)
    {
        var values = new float[hidden];
        int half = hidden / 2;
        for (int i = 0; i < half; i++)
        {
            double inv = Math.Exp(-Math.Log(10000.0) * i / half);
            double value = numDelayTokens * inv;
            values[i] = (float)Math.Cos(value);
            values[i + half] = (float)Math.Sin(value);
        }
        return values;
    }

    /// <summary>Real AdaLN-Zero gate: `1 + Linear(32->hidden, no bias)(GELU(Linear(hidden->32,
    /// no bias)(tCond)))`, broadcast identically across every position.</summary>
    private static float[] AdaGate(float[] tCond, VoxtralTextLayerWeights layer)
    {
        int hidden = VoxtralTextDecoderWeights.HiddenSize;
        int adaDim = VoxtralTextDecoderWeights.AdaHiddenDim;
        var ada = LinearNoBias(tCond, layer.Ada1Weight, hidden, adaDim);
        for (int i = 0; i < ada.Length; i++) ada[i] = GeluErf(ada[i]);
        var gate = LinearNoBias(ada, layer.Ada2Weight, adaDim, hidden);
        for (int i = 0; i < gate.Length; i++) gate[i] += 1f;
        return gate;
    }

    private static float[] SelfAttention(float[][] xRows, int hidden, VoxtralTextLayerWeights l)
    {
        int frames = xRows.Length;
        int heads = VoxtralTextDecoderWeights.NumHeads;
        int kvHeads = VoxtralTextDecoderWeights.NumKvHeads;
        int headDim = VoxtralTextDecoderWeights.HeadDim;
        int kvRepeat = heads / kvHeads;
        float invSqrtD = 1f / MathF.Sqrt(headDim);

        var q = new float[frames][];
        var k = new float[frames][];
        var v = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            q[t] = LinearNoBias(xRows[t], l.QWeight, hidden, heads * headDim);
            k[t] = LinearNoBias(xRows[t], l.KWeight, hidden, kvHeads * headDim);
            v[t] = LinearNoBias(xRows[t], l.VWeight, hidden, kvHeads * headDim);
            RopeNeoxInPlace(q[t], heads, headDim, t);
            RopeNeoxInPlace(k[t], kvHeads, headDim, t);
        }

        var contextFlat = new float[frames][];
        for (int t = 0; t < frames; t++) contextFlat[t] = new float[heads * headDim];

        var scores = new float[frames];
        for (int h = 0; h < heads; h++)
        {
            int qBase = h * headDim;
            int kvBase = (h / kvRepeat) * headDim;
            for (int i = 0; i < frames; i++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j <= i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][qBase + d] * k[j][kvBase + d];
                    dot *= invSqrtD;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = 0; j <= i; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j <= i; j++) acc += scores[j] * invSum * v[j][kvBase + d];
                    contextFlat[i][qBase + d] = acc;
                }
            }
        }

        var output = new float[frames][];
        for (int t = 0; t < frames; t++) output[t] = LinearNoBias(contextFlat[t], l.OWeight, heads * headDim, hidden);
        var flatOut = new float[frames * hidden];
        for (int t = 0; t < frames; t++) Array.Copy(output[t], 0, flatOut, t * hidden, hidden);
        return flatOut;
    }

    /// <summary>Real GGML `GGML_ROPE_TYPE_NEOX` convention -- half-split pair rotation, same as
    /// <see cref="VoxtralAudioEncoder"/>'s own RoPE.</summary>
    private static void RopeNeoxInPlace(float[] qkv, int heads, int headDim, int position)
    {
        int half = headDim / 2;
        for (int h = 0; h < heads; h++)
        {
            int baseIdx = h * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Pow(VoxtralTextDecoderWeights.RopeTheta, -2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                float a = qkv[baseIdx + i];
                float b = qkv[baseIdx + half + i];
                qkv[baseIdx + i] = a * cos - b * sin;
                qkv[baseIdx + half + i] = a * sin + b * cos;
            }
        }
    }

    private static float[] Mlp(float[][] xRows, int hidden, VoxtralTextLayerWeights l)
    {
        int frames = xRows.Length;
        int inter = VoxtralTextDecoderWeights.IntermediateSize;
        var flatOut = new float[frames * hidden];
        for (int t = 0; t < frames; t++)
        {
            var gate = LinearNoBias(xRows[t], l.GateWeight, hidden, inter);
            var up = LinearNoBias(xRows[t], l.UpWeight, hidden, inter);
            for (int i = 0; i < inter; i++) gate[i] = Silu(gate[i]) * up[i];
            var down = LinearNoBias(gate, l.DownWeight, inter, hidden);
            Array.Copy(down, 0, flatOut, t * hidden, hidden);
        }
        return flatOut;
    }

    private static void AddRows(float[][] x, float[] flatDelta)
    {
        int hidden = x[0].Length;
        for (int t = 0; t < x.Length; t++)
            for (int c = 0; c < hidden; c++)
                x[t][c] += flatDelta[t * hidden + c];
    }

    private static float[][] RmsNormRows(float[][] xRows, int hidden, float[] weight)
    {
        var output = new float[xRows.Length][];
        for (int t = 0; t < xRows.Length; t++)
        {
            var row = xRows[t];
            double sumSq = 0;
            for (int c = 0; c < hidden; c++) sumSq += (double)row[c] * row[c];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / hidden + VoxtralTextDecoderWeights.RmsNormEps));
            var outRow = new float[hidden];
            for (int c = 0; c < hidden; c++) outRow[c] = row[c] * invRms * weight[c];
            output[t] = outRow;
        }
        return output;
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static float GeluErf(float x) => 0.5f * x * (1f + Erf(x * 0.70710678f));

    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    private static unsafe float[] LinearNoBias(float[] input, byte[] weightQ8_0, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* pOut = output, pIn = input)
        fixed (byte* pW = weightQ8_0)
            OpenTail.Stingray.Cpu.SimdKernels.MatVecQ8_0(pOut, pW, pIn, outDim, inDim);
        return output;
    }

    private static unsafe void LinearQKV_GQA(float[] input, byte[] wQ, byte[] wK, byte[] wV,
        float[] outQ, float[] outK, float[] outV, int hidden, int qDim, int kvDim)
    {
        int bytesPerRow = (hidden / 32) * 34;
        int scratchBytes = OpenTail.Stingray.Cpu.SimdKernels.Q8_0ScratchBytes(hidden);
        byte* scratch = stackalloc byte[scratchBytes];
        fixed (float* pIn = input)
            OpenTail.Stingray.Cpu.SimdKernels.QuantizeRowToQ8_0(pIn, hidden, scratch);

        fixed (byte* pQ = wQ, pK = wK, pV = wV)
        fixed (float* pq = outQ, pk = outK, pv = outV)
        {
            byte* localQ = pQ, localK = pK, localV = pV;
            float* localOutQ = pq, localOutK = pk, localOutV = pv;
            byte* localScratch = scratch;
            if (qDim >= 64)
            {
                Parallel.For(0, qDim, i =>
                {
                    localOutQ[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(localQ + (long)i * bytesPerRow, localScratch, hidden);
                });
            }
            else
            {
                for (int i = 0; i < qDim; i++)
                    pq[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pQ + (long)i * bytesPerRow, scratch, hidden);
            }

            if (kvDim >= 64)
            {
                Parallel.For(0, kvDim, i =>
                {
                    localOutK[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(localK + (long)i * bytesPerRow, localScratch, hidden);
                    localOutV[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(localV + (long)i * bytesPerRow, localScratch, hidden);
                });
            }
            else
            {
                for (int i = 0; i < kvDim; i++)
                {
                    pk[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pK + (long)i * bytesPerRow, scratch, hidden);
                    pv[i] = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(pV + (long)i * bytesPerRow, scratch, hidden);
                }
            }
        }
    }

    /// <summary>Per-layer KV cache for incremental (one-token-at-a-time) decoding -- real KV-cache
    /// reuse across autoregressive steps, matching the reference's <c>DecodeStepGraph</c> mechanism
    /// (same computation as <see cref="Forward"/>'s full self-attention, just incremental).</summary>
    public sealed class KvCache
    {
        public required List<float[]>[] K { get; init; } // [layer][position] -> kvHeads*headDim
        public required List<float[]>[] V { get; init; }

        /// <summary>Per-layer AdaLN-Zero gate, cached because <c>numDelayTokens</c> (and hence
        /// <see cref="TimeEmbedding"/> and every layer's <see cref="AdaGate"/> output) is constant
        /// for an entire generation -- recomputing it from scratch on every decoded token was pure
        /// waste (perf-sweep Phase 1.3, see docs/perf-sweep-plan.md).</summary>
        internal float[][]? CachedAdaScale { get; set; }
        internal int CachedNumDelayTokens { get; set; }
    }

    private static float[][] AdaScalePerLayer(VoxtralTextDecoderWeights w, KvCache cache, int numDelayTokens)
    {
        if (cache.CachedAdaScale is null || cache.CachedNumDelayTokens != numDelayTokens)
        {
            int hidden = VoxtralTextDecoderWeights.HiddenSize;
            var tCond = TimeEmbedding(numDelayTokens, hidden);
            var scales = new float[w.Layers.Length][];
            for (int li = 0; li < w.Layers.Length; li++) scales[li] = AdaGate(tCond, w.Layers[li]);
            cache.CachedAdaScale = scales;
            cache.CachedNumDelayTokens = numDelayTokens;
        }
        return cache.CachedAdaScale;
    }

    /// <summary>Runs prefill exactly like <see cref="Forward"/> but also populates a
    /// <see cref="KvCache"/> so subsequent single-token steps can reuse the computed K/V instead of
    /// recomputing the full prefix every step. Returns per-position logits (same as
    /// <see cref="Forward"/>) plus the populated cache.</summary>
    public static (float[][] Logits, KvCache Cache) PrefillWithCache(VoxtralTextDecoderWeights w, int[] tokenIds, float[][] audioEmbeddings, int numDelayTokens)
    {
        int hidden = VoxtralTextDecoderWeights.HiddenSize;
        int n = tokenIds.Length;
        var x = new float[n][];
        for (int t = 0; t < n; t++)
        {
            var row = new float[hidden];
            int embedBase = tokenIds[t] * hidden;
            for (int d = 0; d < hidden; d++) row[d] = w.EmbedTokensWeight[embedBase + d];
            if (t < audioEmbeddings.Length)
            {
                var audio = audioEmbeddings[t];
                for (int d = 0; d < hidden; d++) row[d] += audio[d];
            }
            x[t] = row;
        }

        int numLayers = w.Layers.Length;
        var cache = new KvCache
        {
            K = new List<float[]>[numLayers],
            V = new List<float[]>[numLayers],
        };
        for (int l = 0; l < numLayers; l++) { cache.K[l] = new List<float[]>(n + 128); cache.V[l] = new List<float[]>(n + 128); }

        var adaScales = AdaScalePerLayer(w, cache, numDelayTokens);

        for (int li = 0; li < numLayers; li++)
        {
            var layer = w.Layers[li];
            var attnIn = RmsNormRows(x, hidden, layer.InputNorm);
            var attn = SelfAttentionCaching(attnIn, hidden, layer, cache.K[li], cache.V[li]);
            AddRows(x, attn);

            var mlpIn = RmsNormRows(x, hidden, layer.PostNorm);
            var scale = adaScales[li];
            for (int t = 0; t < n; t++)
                for (int d = 0; d < hidden; d++)
                    mlpIn[t][d] *= scale[d];
            var mlp = Mlp(mlpIn, hidden, layer);
            AddRows(x, mlp);
        }

        var normed = RmsNormRows(x, hidden, w.NormWeight);
        var logits = new float[n][];
        for (int t = 0; t < n; t++)
            logits[t] = LinearNoBias(normed[t], w.EmbedTokensWeightQ8_0, hidden, VoxtralTextDecoderWeights.VocabSize);
        return (logits, cache);
    }

    /// <summary>Decodes exactly one new token at <paramref name="position"/>, reusing/extending
    /// <paramref name="cache"/>. <paramref name="audioEmbedding"/> is the audio row to add at this
    /// position, or null once the audio-embedding sequence is exhausted. Returns logits for the
    /// new position.</summary>
    public static float[] Step(VoxtralTextDecoderWeights w, KvCache cache, int tokenId, float[]? audioEmbedding, int position, int numDelayTokens)
    {
        int hidden = VoxtralTextDecoderWeights.HiddenSize;
        var x = new float[hidden];
        int embedBase = tokenId * hidden;
        for (int d = 0; d < hidden; d++) x[d] = w.EmbedTokensWeight[embedBase + d];
        if (audioEmbedding is not null)
            for (int d = 0; d < hidden; d++) x[d] += audioEmbedding[d];

        var adaScales = AdaScalePerLayer(w, cache, numDelayTokens);

        for (int li = 0; li < w.Layers.Length; li++)
        {
            var layer = w.Layers[li];
            var attnIn = RmsNormRow(x, hidden, layer.InputNorm);
            var attn = SelfAttentionStep(attnIn, hidden, layer, cache.K[li], cache.V[li], position);
            for (int d = 0; d < hidden; d++) x[d] += attn[d];

            var mlpIn = RmsNormRow(x, hidden, layer.PostNorm);
            var scale = adaScales[li];
            for (int d = 0; d < hidden; d++) mlpIn[d] *= scale[d];
            var mlp = MlpRow(mlpIn, hidden, layer);
            for (int d = 0; d < hidden; d++) x[d] += mlp[d];
        }

        var normed = RmsNormRow(x, hidden, w.NormWeight);
        return LinearNoBias(normed, w.EmbedTokensWeightQ8_0, hidden, VoxtralTextDecoderWeights.VocabSize);
    }

    private static float[] SelfAttentionCaching(float[][] xRows, int hidden, VoxtralTextLayerWeights l, List<float[]> kCache, List<float[]> vCache)
    {
        int frames = xRows.Length;
        int heads = VoxtralTextDecoderWeights.NumHeads;
        int kvHeads = VoxtralTextDecoderWeights.NumKvHeads;
        int headDim = VoxtralTextDecoderWeights.HeadDim;
        int kvRepeat = heads / kvHeads;
        float invSqrtD = 1f / MathF.Sqrt(headDim);
        int qDim = heads * headDim;
        int kvDim = kvHeads * headDim;

        var q = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var qt = new float[qDim];
            var kt = new float[kvDim];
            var vt = new float[kvDim];
            LinearQKV_GQA(xRows[t], l.QWeight, l.KWeight, l.VWeight, qt, kt, vt, hidden, qDim, kvDim);
            RopeNeoxInPlace(qt, heads, headDim, t);
            RopeNeoxInPlace(kt, kvHeads, headDim, t);
            q[t] = qt;
            kCache.Add(kt);
            vCache.Add(vt);
        }

        var contextFlat = new float[frames][];
        for (int t = 0; t < frames; t++) contextFlat[t] = new float[heads * headDim];

        var scores = new float[frames];
        for (int h = 0; h < heads; h++)
        {
            int qBase = h * headDim;
            int kvBase = (h / kvRepeat) * headDim;
            for (int i = 0; i < frames; i++)
            {
                float maxScore = float.NegativeInfinity;
                var qiSpan = q[i].AsSpan(qBase, headDim);
                for (int j = 0; j <= i; j++)
                {
                    float dot = System.Numerics.Tensors.TensorPrimitives.Dot(qiSpan, kCache[j].AsSpan(kvBase, headDim)) * invSqrtD;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = 0; j <= i; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;
                var ctxSpan = contextFlat[i].AsSpan(qBase, headDim);
                for (int j = 0; j <= i; j++)
                {
                    float s = scores[j] * invSum;
                    System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(
                        vCache[j].AsSpan(kvBase, headDim), s, ctxSpan, ctxSpan);
                }
            }
        }

        var output = new float[frames][];
        for (int t = 0; t < frames; t++) output[t] = LinearNoBias(contextFlat[t], l.OWeight, heads * headDim, hidden);
        var flatOut = new float[frames * hidden];
        for (int t = 0; t < frames; t++) Array.Copy(output[t], 0, flatOut, t * hidden, hidden);
        return flatOut;
    }

    private static float[] SelfAttentionStep(float[] xRow, int hidden, VoxtralTextLayerWeights l, List<float[]> kCache, List<float[]> vCache, int position)
    {
        int heads = VoxtralTextDecoderWeights.NumHeads;
        int kvHeads = VoxtralTextDecoderWeights.NumKvHeads;
        int headDim = VoxtralTextDecoderWeights.HeadDim;
        int kvRepeat = heads / kvHeads;
        float invSqrtD = 1f / MathF.Sqrt(headDim);
        int qDim = heads * headDim;
        int kvDim = kvHeads * headDim;

        var q = new float[qDim];
        var k = new float[kvDim];
        var v = new float[kvDim];
        LinearQKV_GQA(xRow, l.QWeight, l.KWeight, l.VWeight, q, k, v, hidden, qDim, kvDim);

        RopeNeoxInPlace(q, heads, headDim, position);
        RopeNeoxInPlace(k, kvHeads, headDim, position);
        kCache.Add(k);
        vCache.Add(v);

        int total = kCache.Count; // includes this new position
        var scores = new float[total];
        var context = new float[heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            int qBase = h * headDim;
            int kvBase = (h / kvRepeat) * headDim;
            float maxScore = float.NegativeInfinity;
            var qSpan = q.AsSpan(qBase, headDim);
            for (int j = 0; j < total; j++)
            {
                float dot = System.Numerics.Tensors.TensorPrimitives.Dot(qSpan, kCache[j].AsSpan(kvBase, headDim)) * invSqrtD;
                scores[j] = dot;
                if (dot > maxScore) maxScore = dot;
            }
            float sum = 0f;
            for (int j = 0; j < total; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
            float invSum = 1f / sum;
            var ctxSpan = context.AsSpan(qBase, headDim);
            for (int j = 0; j < total; j++)
            {
                float s = scores[j] * invSum;
                System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(
                    vCache[j].AsSpan(kvBase, headDim), s, ctxSpan, ctxSpan);
            }
        }

        return LinearNoBias(context, l.OWeight, heads * headDim, hidden);
    }

    private static unsafe float[] MlpRow(float[] xRow, int hidden, VoxtralTextLayerWeights l)
    {
        int inter = VoxtralTextDecoderWeights.IntermediateSize;
        int inScratchBytes = OpenTail.Stingray.Cpu.SimdKernels.Q8_0ScratchBytes(hidden);
        byte* inScratch = stackalloc byte[inScratchBytes];
        fixed (float* pIn = xRow)
            OpenTail.Stingray.Cpu.SimdKernels.QuantizeRowToQ8_0(pIn, hidden, inScratch);

        var gate = new float[inter];
        int inBytesPerRow = (hidden / 32) * 34;
        fixed (byte* pGW = l.GateWeight, pUW = l.UpWeight)
        fixed (float* pGate = gate)
        {
            byte* localGW = pGW, localUW = pUW;
            float* localGate = pGate;
            byte* localScratch = inScratch;
            Parallel.For(0, inter, i =>
            {
                long off = (long)i * inBytesPerRow;
                float g = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(localGW + off, localScratch, hidden);
                float u = OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0_Q8_0(localUW + off, localScratch, hidden);
                localGate[i] = Silu(g) * u;
            });
        }

        return LinearNoBias(gate, l.DownWeight, inter, hidden);
    }

    private static float[] RmsNormRow(float[] row, int hidden, float[] weight)
    {
        float sumSq = System.Numerics.Tensors.TensorPrimitives.SumOfSquares(row.AsSpan(0, hidden));
        float invRms = (float)(1.0 / Math.Sqrt((double)sumSq / hidden + VoxtralTextDecoderWeights.RmsNormEps));
        var outRow = new float[hidden];
        System.Numerics.Tensors.TensorPrimitives.Multiply(row.AsSpan(0, hidden), weight.AsSpan(0, hidden), outRow);
        System.Numerics.Tensors.TensorPrimitives.Multiply(outRow, invRms, outRow);
        return outRow;
    }
}
