namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// One input row for MOSS-TTS-Nano's global transformer: a text-token id plus this frame's 16 RVQ
/// audio-codebook token ids (or <see cref="MossTtsGlobalTransformerWeights.AudioPadTokenId"/> for a
/// codebook not present at this row -- e.g. every row of the text-only prompt). Mirrors the packed
/// `row_width = n_vq + 1` layout `global_transformer.cpp`'s `Graph::run` reads from.
/// </summary>
public readonly struct MossTtsGlobalRow(int textId, int[] audioIds)
{
    public int TextId { get; } = textId;
    public int[] AudioIds { get; } = audioIds; // length == NumCodebooks
}

/// <summary>
/// Pre-allocated key-value cache for MOSS-TTS-Nano's 12-layer global transformer.
/// </summary>
public sealed class MossTtsGlobalKvCache
{
    public int Length { get; internal set; }
    public float[][][] Keys { get; }
    public float[][][] Values { get; }

    public MossTtsGlobalKvCache(int maxCapacity = 512, int numLayers = MossTtsGlobalTransformerWeights.NumLayers)
    {
        Keys = new float[numLayers][][];
        Values = new float[numLayers][][];
        for (int l = 0; l < numLayers; l++)
        {
            Keys[l] = new float[maxCapacity][];
            Values[l] = new float[maxCapacity][];
            for (int t = 0; t < maxCapacity; t++)
            {
                Keys[l][t] = new float[MossTtsGlobalTransformerWeights.HiddenDim];
                Values[l][t] = new float[MossTtsGlobalTransformerWeights.HiddenDim];
            }
        }
        Length = 0;
    }
}

/// <summary>
/// Native C# port of MOSS-TTS-Nano's global transformer forward pass, from
/// `examples/audio.cpp/src/models/moss/moss_tts_nano/global_transformer.cpp`'s `transformer_layer`/
/// `Graph` constructor (not guessed). Real per-row input embedding: the text embedding for that
/// row's text token, PLUS the sum of all 16 audio-codebook embeddings whose id is not the pad
/// sentinel at that row (matches the reference's `emb * mask` accumulation into `hidden`) -- text
/// and audio tokens for the same row are summed, not concatenated, matching the "text row" vs.
/// "audio frame row" structure MOSS-TTS-Nano's real prompt layout uses. Real per-layer math:
/// pre-LN GPT2 block with fused QKV (`c_attn`), adjacent-pair RoPE (`GGML_ROPE_TYPE_NORMAL`) on Q/K
/// only (not V), scaled dot-product causal self-attention, `c_proj` output projection, a second
/// pre-LN MLP block (`fc_in` -&gt; `gelu_new` -&gt; `fc_out`), both with residual adds. Only the LAST
/// row's post-final-LayerNorm hidden state is needed by callers (matches the reference's
/// `Graph::run`, which returns only the final row).
/// </summary>
public static class MossTtsGlobalTransformer
{
    /// <summary>Embeds a single input row (text embedding + sum of present audio codebook embeddings).</summary>
    public static float[] EmbedInputRow(MossTtsGlobalTransformerWeights w, MossTtsGlobalRow row)
    {
        var h = EmbedRow(w.TextEmbedding, row.TextId, MossTtsGlobalTransformerWeights.HiddenDim);
        for (int q = 0; q < MossTtsGlobalTransformerWeights.NumCodebooks; q++)
        {
            int id = row.AudioIds[q];
            if (id == MossTtsGlobalTransformerWeights.AudioPadTokenId) continue;
            var emb = EmbedRow(w.AudioEmbeddings[q], id, MossTtsGlobalTransformerWeights.HiddenDim);
            TensorPrimitives.Add(h, emb, h);
        }
        return h;
    }

    /// <summary>
    /// Processes initial input rows in full, populates the KV cache for all layers, and returns the last row's hidden state.
    /// </summary>
    public static float[] ForwardPrefillRows(
        float[][] inputRows,
        IReadOnlyList<MossTtsGlobalTransformerLayerWeights> layers,
        float[] finalNormWeight,
        float[] finalNormBias,
        MossTtsGlobalKvCache cache)
    {
        int n = inputRows.Length;
        const int dim = MossTtsGlobalTransformerWeights.HiddenDim;
        const int headDim = MossTtsGlobalTransformerWeights.HeadDim;
        const int numHeads = MossTtsGlobalTransformerWeights.NumHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var current = new float[n][];
        for (int i = 0; i < n; i++) current[i] = (float[])inputRows[i].Clone();

        for (int l = 0; l < layers.Count; l++)
        {
            var layer = layers[l];
            var normed = new float[n][];
            for (int i = 0; i < n; i++) normed[i] = LayerNorm(current[i], layer.Ln1Weight, layer.Ln1Bias);
            var qkvAll = LinearBatched(normed, layer.CAttnWeight, layer.CAttnBias, dim, dim * 3);

            var q = new float[n][];
            for (int i = 0; i < n; i++)
            {
                q[i] = new float[dim];
                var k = cache.Keys[l][i];
                var v = cache.Values[l][i];
                Array.Copy(qkvAll[i], 0, q[i], 0, dim);
                Array.Copy(qkvAll[i], dim, k, 0, dim);
                Array.Copy(qkvAll[i], dim * 2, v, 0, dim);
                ApplyRopeAdjacentPairs(q[i], numHeads, headDim, position: i);
                ApplyRopeAdjacentPairs(k, numHeads, headDim, position: i);
            }

            var contexts = new float[n][];
            Parallel.For(0, n, i =>
            {
                var context = new float[dim];
                int availableKeys = i + 1;
                var scores = new float[availableKeys];
                unsafe
                {
                    fixed (float* qp = q[i])
                    {
                        for (int hOffIdx = 0; hOffIdx < numHeads; hOffIdx++)
                        {
                            int hOff = hOffIdx * headDim;
                            for (int t = 0; t < availableKeys; t++)
                            {
                                fixed (float* kp = cache.Keys[l][t])
                                    scores[t] = SimdKernels.DotF32(qp + hOff, kp + hOff, headDim) * scale;
                            }
                            SoftmaxInPlace(scores);
                            for (int t = 0; t < availableKeys; t++)
                            {
                                var vt = cache.Values[l][t];
                                float p = scores[t];
                                for (int d = 0; d < headDim; d++) context[hOff + d] += p * vt[hOff + d];
                            }
                        }
                    }
                }
                contexts[i] = context;
            });

            var attnOut = LinearBatched(contexts, layer.CProjWeight, layer.CProjBias, dim, dim);
            for (int i = 0; i < n; i++) TensorPrimitives.Add(current[i], attnOut[i], current[i]);

            var ffnNormed = new float[n][];
            for (int i = 0; i < n; i++) ffnNormed[i] = LayerNorm(current[i], layer.Ln2Weight, layer.Ln2Bias);
            var fcAll = LinearBatched(ffnNormed, layer.FcInWeight, layer.FcInBias, dim, MossTtsGlobalTransformerWeights.IntermediateDim);
            for (int i = 0; i < n; i++) GeluNewInPlace(fcAll[i]);
            var projAll = LinearBatched(fcAll, layer.FcOutWeight, layer.FcOutBias, MossTtsGlobalTransformerWeights.IntermediateDim, dim);
            for (int i = 0; i < n; i++) TensorPrimitives.Add(current[i], projAll[i], current[i]);
        }

        cache.Length = n;
        return LayerNorm(current[^1], finalNormWeight, finalNormBias);
    }

    /// <summary>
    /// Processes initial prompt rows in full, populates the KV cache for all layers, and returns the last row's hidden state.
    /// </summary>
    public static float[] ForwardPrefill(
        MossTtsGlobalTransformerWeights w,
        IReadOnlyList<MossTtsGlobalRow> prompt,
        MossTtsGlobalKvCache cache)
    {
        var inputRows = new float[prompt.Count][];
        for (int i = 0; i < prompt.Count; i++)
            inputRows[i] = EmbedInputRow(w, prompt[i]);
        return ForwardPrefillRows(inputRows, w.Layers, w.FinalNormWeight, w.FinalNormBias, cache);
    }

    /// <summary>
    /// Evaluates only the single newly appended row, updating the KV cache and returning its hidden state.
    /// </summary>
    public static float[] ForwardStepRow(
        float[] inputRow,
        IReadOnlyList<MossTtsGlobalTransformerLayerWeights> layers,
        float[] finalNormWeight,
        float[] finalNormBias,
        MossTtsGlobalKvCache cache)
    {
        int p = cache.Length;
        const int dim = MossTtsGlobalTransformerWeights.HiddenDim;
        const int headDim = MossTtsGlobalTransformerWeights.HeadDim;
        const int numHeads = MossTtsGlobalTransformerWeights.NumHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var current = (float[])inputRow.Clone();

        for (int l = 0; l < layers.Count; l++)
        {
            var layer = layers[l];
            var normed = LayerNorm(current, layer.Ln1Weight, layer.Ln1Bias);
            var qkv = Linear(normed, layer.CAttnWeight, layer.CAttnBias, dim, dim * 3);

            var q = new float[dim];
            var k = cache.Keys[l][p];
            var v = cache.Values[l][p];
            Array.Copy(qkv, 0, q, 0, dim);
            Array.Copy(qkv, dim, k, 0, dim);
            Array.Copy(qkv, dim * 2, v, 0, dim);
            ApplyRopeAdjacentPairs(q, numHeads, headDim, position: p);
            ApplyRopeAdjacentPairs(k, numHeads, headDim, position: p);

            var context = new float[dim];
            int availableKeys = p + 1;
            var scores = new float[availableKeys];
            unsafe
            {
                fixed (float* qp = q)
                {
                    for (int hOffIdx = 0; hOffIdx < numHeads; hOffIdx++)
                    {
                        int hOff = hOffIdx * headDim;
                        for (int t = 0; t < availableKeys; t++)
                        {
                            fixed (float* kp = cache.Keys[l][t])
                                scores[t] = SimdKernels.DotF32(qp + hOff, kp + hOff, headDim) * scale;
                        }
                        SoftmaxInPlace(scores);
                        for (int t = 0; t < availableKeys; t++)
                        {
                            var vt = cache.Values[l][t];
                            float prob = scores[t];
                            for (int d = 0; d < headDim; d++) context[hOff + d] += prob * vt[hOff + d];
                        }
                    }
                }
            }

            var attnOut = Linear(context, layer.CProjWeight, layer.CProjBias, dim, dim);
            TensorPrimitives.Add(current, attnOut, current);

            var ffnNormed = LayerNorm(current, layer.Ln2Weight, layer.Ln2Bias);
            var fcAll = Linear(ffnNormed, layer.FcInWeight, layer.FcInBias, dim, MossTtsGlobalTransformerWeights.IntermediateDim);
            GeluNewInPlace(fcAll);
            var projAll = Linear(fcAll, layer.FcOutWeight, layer.FcOutBias, MossTtsGlobalTransformerWeights.IntermediateDim, dim);
            TensorPrimitives.Add(current, projAll, current);
        }

        cache.Length = p + 1;
        return LayerNorm(current, finalNormWeight, finalNormBias);
    }

    /// <summary>
    /// Evaluates only the single newly appended row, updating the KV cache and returning its hidden state.
    /// </summary>
    public static float[] ForwardStep(
        MossTtsGlobalTransformerWeights w,
        MossTtsGlobalRow newRow,
        MossTtsGlobalKvCache cache)
    {
        var inputRow = EmbedInputRow(w, newRow);
        return ForwardStepRow(inputRow, w.Layers, w.FinalNormWeight, w.FinalNormBias, cache);
    }

    /// <summary>
    /// Runs the full causal sequence and returns every row's post-final-LayerNorm hidden state
    /// (not just the last one -- useful for prefill where multiple rows' hidden states are needed,
    /// e.g. to drive the local transformer for each already-known prompt frame).
    /// </summary>
    public static float[][] ForwardAll(MossTtsGlobalTransformerWeights w, IReadOnlyList<MossTtsGlobalRow> rows)
    {
        int n = rows.Count;
        var hidden = new float[n][];
        for (int i = 0; i < n; i++)
            hidden[i] = EmbedInputRow(w, rows[i]);

        return RunTransformerStack(hidden, w.Layers, w.FinalNormWeight, w.FinalNormBias);
    }

    /// <summary>
    /// Shared GPT2+RoPE transformer-stack forward pass: given N already-embedded input rows (row 0
    /// gets RoPE position 0, row 1 position 1, etc.) and a stack of layers with matching
    /// <see cref="MossTtsGlobalTransformerWeights.HiddenDim"/>/<see cref="MossTtsGlobalTransformerWeights.NumHeads"/>,
    /// runs the full causal sequence and returns every row's post-final-LayerNorm hidden state.
    /// Used by both the global transformer (<see cref="ForwardAll"/>) and the local frame decoder
    /// (<see cref="MossTtsLocalFrameDecoder"/>) -- same real per-layer math in both, per
    /// `global_transformer.cpp`/`local_frame_decoder.cpp`'s identical `transformer_layer` helper.
    /// </summary>
    internal static float[][] RunTransformerStack(
        float[][] hidden,
        IReadOnlyList<MossTtsGlobalTransformerLayerWeights> layers,
        float[] finalNormWeight,
        float[] finalNormBias)
    {
        int n = hidden.Length;
        const int dim = MossTtsGlobalTransformerWeights.HiddenDim;
        const int headDim = MossTtsGlobalTransformerWeights.HeadDim;
        const int numHeads = MossTtsGlobalTransformerWeights.NumHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var current = new float[n][];
        for (int i = 0; i < n; i++) current[i] = (float[])hidden[i].Clone();

        foreach (var layer in layers)
        {
            var normed = new float[n][];
            for (int i = 0; i < n; i++) normed[i] = LayerNorm(current[i], layer.Ln1Weight, layer.Ln1Bias);
            var qkvAll = LinearBatched(normed, layer.CAttnWeight, layer.CAttnBias, dim, dim * 3);

            var q = new float[n][];
            var k = new float[n][];
            var v = new float[n][];
            for (int i = 0; i < n; i++)
            {
                q[i] = new float[dim];
                k[i] = new float[dim];
                v[i] = new float[dim];
                Array.Copy(qkvAll[i], 0, q[i], 0, dim);
                Array.Copy(qkvAll[i], dim, k[i], 0, dim);
                Array.Copy(qkvAll[i], dim * 2, v[i], 0, dim);
                ApplyRopeAdjacentPairs(q[i], numHeads, headDim, position: i);
                ApplyRopeAdjacentPairs(k[i], numHeads, headDim, position: i);
            }

            var contexts = new float[n][];
            Parallel.For(0, n, i =>
            {
                var context = new float[dim];
                int availableKeys = i + 1;
                var scores = new float[availableKeys];
                unsafe
                {
                    fixed (float* qp = q[i])
                    {
                        for (int hOffIdx = 0; hOffIdx < numHeads; hOffIdx++)
                        {
                            int hOff = hOffIdx * headDim;
                            for (int t = 0; t < availableKeys; t++)
                            {
                                fixed (float* kp = k[t])
                                    scores[t] = SimdKernels.DotF32(qp + hOff, kp + hOff, headDim) * scale;
                            }
                            SoftmaxInPlace(scores);
                            for (int t = 0; t < availableKeys; t++)
                            {
                                var vt = v[t];
                                float p = scores[t];
                                for (int d = 0; d < headDim; d++) context[hOff + d] += p * vt[hOff + d];
                            }
                        }
                    }
                }
                contexts[i] = context;
            });

            var attnOut = LinearBatched(contexts, layer.CProjWeight, layer.CProjBias, dim, dim);
            for (int i = 0; i < n; i++) TensorPrimitives.Add(current[i], attnOut[i], current[i]);

            var ffnNormed = new float[n][];
            for (int i = 0; i < n; i++) ffnNormed[i] = LayerNorm(current[i], layer.Ln2Weight, layer.Ln2Bias);
            var fcAll = LinearBatched(ffnNormed, layer.FcInWeight, layer.FcInBias, dim, MossTtsGlobalTransformerWeights.IntermediateDim);
            for (int i = 0; i < n; i++) GeluNewInPlace(fcAll[i]);
            var projAll = LinearBatched(fcAll, layer.FcOutWeight, layer.FcOutBias, MossTtsGlobalTransformerWeights.IntermediateDim, dim);
            for (int i = 0; i < n; i++) TensorPrimitives.Add(current[i], projAll[i], current[i]);
        }

        var result = new float[n][];
        for (int i = 0; i < n; i++) result[i] = LayerNorm(current[i], finalNormWeight, finalNormBias);
        return result;
    }

    /// <summary>Convenience wrapper: runs the full sequence and returns only the last row's hidden state.</summary>
    public static float[] ForwardLastHidden(MossTtsGlobalTransformerWeights w, IReadOnlyList<MossTtsGlobalRow> rows)
        => ForwardAll(w, rows)[^1];

    /// <summary>Projects a hidden state through the text LM head (tied-embedding weight, no bias).</summary>
    public static float[] TextLogits(MossTtsGlobalTransformerWeights w, float[] hidden)
        => Linear(hidden, w.TextLmHeadWeight, bias: [], MossTtsGlobalTransformerWeights.HiddenDim, MossTtsGlobalTransformerWeights.VocabSize);

    // -----------------------------------------------------------------------
    // Real GGML_ROPE_TYPE_NORMAL (adjacent-pair) rotary embedding, applied per-head in place.
    // -----------------------------------------------------------------------
    internal static void ApplyRopeAdjacentPairs(float[] x, int numHeads, int headDim, int position)
    {
        int half = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < half; i++)
            {
                double freq = 1.0 / Math.Pow(MossTtsGlobalTransformerWeights.RopeBase, (2.0 * i) / headDim);
                double theta = position * freq;
                float cos = (float)Math.Cos(theta);
                float sin = (float)Math.Sin(theta);
                int idx0 = hOff + 2 * i;
                int idx1 = idx0 + 1;
                float x0 = x[idx0];
                float x1 = x[idx1];
                x[idx0] = x0 * cos - x1 * sin;
                x[idx1] = x0 * sin + x1 * cos;
            }
        }
    }

    internal static float[] EmbedRow(float[] table, int index, int dim)
    {
        var row = new float[dim];
        Array.Copy(table, (long)index * dim, row, 0, dim);
        return row;
    }

    internal static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, x = input, y = output)
        {
            SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        }
        if (bias is { Length: > 0 })
            for (int o = 0; o < outDim; o++) output[o] += bias[o];
        return output;
    }

    internal static unsafe float[][] LinearBatched(float[][] inputs, float[] weight, float[] bias, int inDim, int outDim)
    {
        int n = inputs.Length;
        var outputs = new float[n][];
        for (int i = 0; i < n; i++) outputs[i] = new float[outDim];

        Parallel.For(0, n, i =>
        {
            fixed (float* w = weight, x = inputs[i], y = outputs[i], b = bias)
            {
                SimdKernels.MatVecF32(y, w, x, outDim, inDim);
                for (int o = 0; o < outDim; o++) y[o] += b[o];
            }
        });

        return outputs;
    }

    internal static unsafe float[] LayerNorm(float[] x, float[] weight, float[] bias)
    {
        var output = new float[x.Length];
        fixed (float* xp = x, wp = weight, bp = bias, op = output)
        {
            SimdKernels.LayerNorm(op, xp, wp, bp, x.Length, MossTtsGlobalTransformerWeights.LayerNormEps);
        }
        return output;
    }

    internal static void SoftmaxInPlace(Span<float> scores)
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

    private static void GeluNewInPlace(float[] x)
    {
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            float inner = c * (v + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
    }
}
