using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Native C# 24-layer GPT2-medium-architecture Autoregressive Acoustic Language Model ("T3") for
/// Chatterbox-Turbo TTS, ported from examples/chatterbox-tts-py/chatterbox/models/t3/t3.py
/// (T3.forward/prepare_input_embeds/inference_turbo) and llama_configs.py's GPT2_MEDIUM_CONFIG.
/// Generates discrete speech tokens conditioned on a projected speaker embedding and a fixed-length
/// bank of speech "prompt" tokens (both baked into the GGUF as the built-in default voice), using a
/// standard pre-LN GPT2 transformer body with absolute (wpe) position embeddings, fused QKV, and
/// gelu_new-activated MLP. When no real weights are supplied, falls back to the original
/// placeholder/synthetic generator (kept for the parameterless/no-model test and demo path).
/// </summary>
public sealed class ChatterboxAcousticLm : IDisposable
{
    public const int HiddenDim = 1024;
    public const int NumLayers = 24;
    public const int NumHeads = 16;
    public const int HeadDim = HiddenDim / NumHeads; // 64
    public const int VocabSize = 8192;
    public const int StartSpeechToken = 6561;
    public const int StopSpeechToken = 6562;

    public float RepetitionPenalty { get; set; } = 1.2f;
    public int TopK { get; set; } = 1000;
    public float TopP { get; set; } = 0.95f;

    private readonly ChatterboxWeights? _weights;
    private readonly Random _rng;

    public ChatterboxAcousticLm(ChatterboxWeights? weights = null, Random? rng = null)
    {
        _weights = weights;
        _rng = rng ?? Random.Shared;
    }

    /// <summary>
    /// Autoregressively synthesizes discrete speech tokens from text token sequence. Uses the real
    /// GPT2 T3 transformer when real weights were supplied at construction; otherwise falls back to
    /// the original synthetic placeholder generator.
    /// </summary>
    public List<int> GenerateSpeechTokens(
        ReadOnlySpan<int> textTokens,
        float[] speakerFeatures,
        float temperature = 0.7f,
        int maxTokens = 512)
    {
        if (_weights is { } w)
            return GenerateReal(w, textTokens, temperature, maxTokens);

        return GenerateFakePlaceholder(textTokens, speakerFeatures, temperature, maxTokens);
    }

    // -----------------------------------------------------------------------
    // Real T3 GPT2 inference (inference_turbo)
    // -----------------------------------------------------------------------

    private List<int> GenerateReal(ChatterboxWeights w, ReadOnlySpan<int> textTokens, float temperature, int maxTokens)
    {
        int startSpeech = w.StartSpeechToken;
        int stopSpeech = w.StopSpeechToken;

        // --- Assemble conditioning embeds: [spkr_proj(1), speech_emb(prompt_tokens)(SpeechCondPromptLen)] ---
        var condEmbeds = new List<float[]>(1 + w.SpeechCondPromptLen);
        float[] speakerEmb = w.SpeakerEmbedding ?? new float[w.SpeakerEmbedSize];
        condEmbeds.Add(Linear(speakerEmb, w.SpkrEncWeight, w.SpkrEncBias, w.SpeakerEmbedSize, w.HiddenDim));

        if (w.SpeechPromptTokens is { } promptTokens)
        {
            foreach (int tok in promptTokens)
                condEmbeds.Add(EmbedRow(w.SpeechEmbWeight, tok, w.HiddenDim));
        }

        // --- Text embeds ---
        var chunk = new List<float[]>(condEmbeds.Count + textTokens.Length + 1);
        chunk.AddRange(condEmbeds);
        foreach (int tok in textTokens)
            chunk.Add(EmbedRow(w.TextEmbWeight, tok, w.HiddenDim));

        // --- Initial speech token (BOS = start_speech_token) ---
        chunk.Add(EmbedRow(w.SpeechEmbWeight, startSpeech, w.HiddenDim));

        var kCache = new List<float[]>[w.NumLayers];
        var vCache = new List<float[]>[w.NumLayers];
        for (int l = 0; l < w.NumLayers; l++)
        {
            kCache[l] = new List<float[]>(chunk.Count + maxTokens);
            vCache[l] = new List<float[]>(chunk.Count + maxTokens);
        }

        // Prefill: process the whole [cond, text, BOS] chunk at once.
        float[][] hidden = ProcessChunk(w, chunk.ToArray(), startPos: 0, kCache, vCache);
        int pos = chunk.Count;

        var speechTokens = new List<int>();
        float[] lastHidden = hidden[^1];
        int nextToken = SampleNext(w, lastHidden, speechTokens, temperature);
        speechTokens.Add(nextToken);

        for (int step = 1; step < maxTokens; step++)
        {
            if (nextToken == stopSpeech) break;

            var stepEmbed = EmbedRow(w.SpeechEmbWeight, nextToken, w.HiddenDim);
            float[][] stepHidden = ProcessChunk(w, [stepEmbed], startPos: pos, kCache, vCache);
            pos++;

            nextToken = SampleNext(w, stepHidden[0], speechTokens, temperature);
            speechTokens.Add(nextToken);
        }

        // Drop a trailing EOS if present (matches inference_turbo's post-processing).
        if (speechTokens.Count > 0 && speechTokens[^1] == stopSpeech)
            speechTokens.RemoveAt(speechTokens.Count - 1);

        var result = new List<int>(speechTokens.Count + 2) { startSpeech };
        result.AddRange(speechTokens);
        result.Add(stopSpeech);
        return result;
    }

    private int SampleNext(ChatterboxWeights w, float[] hidden, List<int> historySoFar, float temperature)
    {
        float[] logits = Linear(hidden, w.SpeechHeadWeight, w.SpeechHeadBias, w.HiddenDim, w.SpeechVocabSize);

        // Repetition penalty -> Temperature -> top-k -> top-p
        ApplyRepetitionPenalty(logits, historySoFar, RepetitionPenalty);
        ApplyTemperature(logits, temperature);
        ApplyTopK(logits, TopK);
        ApplyTopP(logits, TopP);

        return SampleFromLogits(logits);
    }

    private static void ApplyTemperature(float[] logits, float temperature)
    {
        if (temperature is <= 0f or 1f) return;
        for (int i = 0; i < logits.Length; i++) logits[i] /= temperature;
    }

    private static void ApplyTopK(float[] logits, int topK)
    {
        if (topK <= 0 || topK >= logits.Length) return;
        var indexed = new (float val, int idx)[logits.Length];
        for (int i = 0; i < logits.Length; i++) indexed[i] = (logits[i], i);
        Array.Sort(indexed, (a, b) => b.val.CompareTo(a.val));
        for (int i = topK; i < indexed.Length; i++) logits[indexed[i].idx] = float.NegativeInfinity;
    }

    private static void ApplyTopP(float[] logits, float topP)
    {
        if (topP >= 1f) return;
        int n = logits.Length;
        var indexed = new (float val, int idx)[n];
        for (int i = 0; i < n; i++) indexed[i] = (logits[i], i);
        Array.Sort(indexed, (a, b) => b.val.CompareTo(a.val));

        float max = indexed[0].val;
        if (float.IsNegativeInfinity(max)) return;
        double sumExp = 0;
        var expVals = new double[n];
        for (int i = 0; i < n; i++)
        {
            expVals[i] = float.IsNegativeInfinity(indexed[i].val) ? 0.0 : Math.Exp(indexed[i].val - max);
            sumExp += expVals[i];
        }

        double cumulative = 0;
        for (int i = 0; i < n; i++)
        {
            double prob = expVals[i] / sumExp;
            cumulative += prob;
            // Keep the smallest prefix whose cumulative probability >= topP; always keep at least 1.
            if (i > 0 && cumulative - prob >= topP)
                logits[indexed[i].idx] = float.NegativeInfinity;
        }
    }

    private static void ApplyRepetitionPenalty(float[] logits, List<int> history, float penalty)
    {
        if (penalty == 1f || history.Count == 0) return;
        foreach (int tok in history)
        {
            if ((uint)tok >= (uint)logits.Length) continue;
            float score = logits[tok];
            if (float.IsNegativeInfinity(score)) continue;
            logits[tok] = score > 0f ? score / penalty : score * penalty;
        }
    }

    private int SampleFromLogits(float[] logits)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        if (float.IsNegativeInfinity(max)) return 0;

        double sum = 0;
        var probs = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            double p = float.IsNegativeInfinity(logits[i]) ? 0.0 : Math.Exp(logits[i] - max);
            probs[i] = p;
            sum += p;
        }

        double r = _rng.NextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            acc += probs[i];
            if (acc >= r) return i;
        }
        return probs.Length - 1;
    }

    /// <summary>
    /// Runs the GPT2 body over a chunk of already-embedded positions (prefill or a single decode
    /// step), appending this chunk's K/V to the running per-layer cache and returning each
    /// position's post-final-LayerNorm hidden state (t3.output_norm, i.e. GPT2Model's ln_f).
    /// </summary>
    private static float[][] ProcessChunk(ChatterboxWeights w, float[][] chunkEmbeds, int startPos, List<float[]>[] kCache, List<float[]>[] vCache)
    {
        int n = chunkEmbeds.Length;
        int dim = w.HiddenDim;

        // hidden_states = inputs_embeds + wpe[position]  (GPT2Model.forward)
        var hidden = new float[n][];
        for (int i = 0; i < n; i++)
        {
            var h = new float[dim];
            int posRow = (startPos + i) * dim;
            for (int d = 0; d < dim; d++) h[d] = chunkEmbeds[i][d] + w.WpeWeight[posRow + d];
            hidden[i] = h;
        }

        for (int l = 0; l < w.NumLayers; l++)
        {
            var layer = w.Layers[l];
            var kCacheL = kCache[l];
            var vCacheL = vCache[l];
            int cacheBase = kCacheL.Count;

            // --- Self-attention block ---
            var attnNormed = new float[n][];
            for (int i = 0; i < n; i++)
                attnNormed[i] = LayerNorm(hidden[i], layer.AttnNormWeight, layer.AttnNormBias);
            var qkvAll = LinearBatched(attnNormed, layer.AttnQkvWeight, layer.AttnQkvBias, dim, 3 * dim);

            for (int i = 0; i < n; i++)
            {
                var q = new float[dim];
                var k = new float[dim];
                var v = new float[dim];
                Array.Copy(qkvAll[i], 0, q, 0, dim);
                Array.Copy(qkvAll[i], dim, k, 0, dim);
                Array.Copy(qkvAll[i], 2 * dim, v, 0, dim);
                kCacheL.Add(k);
                vCacheL.Add(v);
            }

            // Per-position attention context: reads only the cache (already fully populated for
            // this chunk by the append loop above), so it's safe to parallelize across positions
            // -- during prefill this scans up to a few hundred cached K/V entries per position,
            // and was previously a fully scalar, single-threaded O(n^2) scan. The Q.K dot product
            // uses the same SIMD kernel as Linear() below. Deliberately NOT parallelized together
            // with the attn_output Linear() calls (those already parallelize internally over 1024
            // output rows in SimdKernels.MatVecF32; nesting Parallel.For inside this loop would
            // oversubscribe the thread pool for no benefit), so contexts are computed here and
            // projected in a separate, sequential loop below.
            var contexts = new float[n][];
            float scale = 1f / MathF.Sqrt(HeadDim);

            if (n == 1)
            {
                var q = qkvAll[0];
                int availableKeys = cacheBase + 1;
                var context = new float[dim];

                unsafe
                {
                    fixed (float* qp = q, ctxBase = context)
                    {
                        nint qAddr = (nint)qp;
                        nint ctxAddr = (nint)ctxBase;

                        System.Threading.Tasks.Parallel.For(0, NumHeads, h =>
                        {
                            float* qHead = (float*)qAddr;
                            float* ctxHead = (float*)ctxAddr;
                            int hOff = h * HeadDim;
                            float* qPtr = qHead + hOff;
                            float* cPtr = ctxHead + hOff;

                            var scores = stackalloc float[availableKeys];

                            for (int t = 0; t < availableKeys; t++)
                            {
                                var kt = kCacheL[t];
                                fixed (float* ktp = kt)
                                    scores[t] = SimdKernels.DotF32(qPtr, ktp + hOff, HeadDim) * scale;
                            }

                            SoftmaxInPlace(new Span<float>(scores, availableKeys));

                            if (HeadDim == 64 && Avx.IsSupported && Fma.IsSupported)
                            {
                                var c0 = Vector256<float>.Zero;
                                var c1 = Vector256<float>.Zero;
                                var c2 = Vector256<float>.Zero;
                                var c3 = Vector256<float>.Zero;
                                var c4 = Vector256<float>.Zero;
                                var c5 = Vector256<float>.Zero;
                                var c6 = Vector256<float>.Zero;
                                var c7 = Vector256<float>.Zero;

                                for (int t = 0; t < availableKeys; t++)
                                {
                                    float p = scores[t];
                                    if (p == 0f) continue;
                                    var pVec = Vector256.Create(p);
                                    var vt = vCacheL[t];
                                    fixed (float* vtp = vt)
                                    {
                                        float* vRow = vtp + hOff;
                                        c0 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow), c0);
                                        c1 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 8), c1);
                                        c2 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 16), c2);
                                        c3 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 24), c3);
                                        c4 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 32), c4);
                                        c5 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 40), c5);
                                        c6 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 48), c6);
                                        c7 = Fma.MultiplyAdd(pVec, Avx.LoadVector256(vRow + 56), c7);
                                    }
                                }

                                Avx.Store(cPtr, c0);
                                Avx.Store(cPtr + 8, c1);
                                Avx.Store(cPtr + 16, c2);
                                Avx.Store(cPtr + 24, c3);
                                Avx.Store(cPtr + 32, c4);
                                Avx.Store(cPtr + 40, c5);
                                Avx.Store(cPtr + 48, c6);
                                Avx.Store(cPtr + 56, c7);
                            }
                            else
                            {
                                for (int d = 0; d < HeadDim; d++) cPtr[d] = 0f;
                                for (int t = 0; t < availableKeys; t++)
                                {
                                    var vt = vCacheL[t];
                                    float p = scores[t];
                                    if (p == 0f) continue;
                                    fixed (float* vtp = vt)
                                    {
                                        float* vRow = vtp + hOff;
                                        for (int d = 0; d < HeadDim; d++) cPtr[d] += p * vRow[d];
                                    }
                                }
                            }
                        });
                    }
                }
                contexts = [context];
            }
            else
            {
                System.Threading.Tasks.Parallel.For(0, n, i =>
                {
                    var q = new float[dim];
                    Array.Copy(qkvAll[i], 0, q, 0, dim);
                    int selfCachePos = cacheBase + i;
                    int availableKeys = selfCachePos + 1;

                    var context = new float[dim];
                    var scores = new float[availableKeys];
                    unsafe
                    {
                        fixed (float* qp = q)
                        {
                            for (int h = 0; h < NumHeads; h++)
                            {
                                int hOff = h * HeadDim;
                                for (int t = 0; t < availableKeys; t++)
                                {
                                    var kt = kCacheL[t];
                                    fixed (float* ktp = kt)
                                        scores[t] = SimdKernels.DotF32(qp + hOff, ktp + hOff, HeadDim) * scale;
                                }
                                SoftmaxInPlace(scores);
                                for (int t = 0; t < availableKeys; t++)
                                {
                                    var vt = vCacheL[t];
                                    float p = scores[t];
                                    for (int d = 0; d < HeadDim; d++) context[hOff + d] += p * vt[hOff + d];
                                }
                            }
                        }
                    }
                    contexts[i] = context;
                });
            }

            var attnOut = LinearBatched(contexts, layer.AttnOutputWeight, layer.AttnOutputBias, dim, dim);

            for (int i = 0; i < n; i++)
                for (int d = 0; d < dim; d++) hidden[i][d] += attnOut[i][d];

            // --- MLP block ---
            var ffnNormed = new float[n][];
            for (int i = 0; i < n; i++)
                ffnNormed[i] = LayerNorm(hidden[i], layer.FfnNormWeight, layer.FfnNormBias);
            var fcAll = LinearBatched(ffnNormed, layer.FfnFcWeight, layer.FfnFcBias, dim, w.IntermediateSize);
            for (int i = 0; i < n; i++) GeluNewInPlace(fcAll[i]);
            var projAll = LinearBatched(fcAll, layer.FfnProjWeight, layer.FfnProjBias, w.IntermediateSize, dim);
            for (int i = 0; i < n; i++)
                for (int d = 0; d < dim; d++) hidden[i][d] += projAll[i][d];
        }

        // Final LayerNorm (ln_f / t3.output_norm).
        var result = new float[n][];
        for (int i = 0; i < n; i++)
            result[i] = LayerNorm(hidden[i], w.OutputNormWeight, w.OutputNormBias);
        return result;
    }

    private static float[] EmbedRow(float[] table, int index, int dim)
    {
        var row = new float[dim];
        Array.Copy(table, (long)index * dim, row, 0, dim);
        return row;
    }

    /// <summary>
    /// y = W @ x + b, W row-major [outDim, inDim]. Delegates the matvec itself to
    /// SimdKernels.MatVecF32 (AVX2/AVX-512 dot products, auto-parallelized across output rows for
    /// large row counts) -- a scalar per-element loop here made T3's ~400-token conditioning+text
    /// prefill (24 layers x [1024-&gt;3072 QKV, 1024-&gt;1024 attn_out, 1024&lt;-&gt;4096 FFN] projections)
    /// take several minutes; this is the same kernel the main LLM inference engine uses for GGUF
    /// forward passes.
    /// </summary>
    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, x = input, y = output)
        {
            SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        }
        for (int o = 0; o < outDim; o++) output[o] += bias[o];
        return output;
    }

    /// <summary>
    /// Batched form of <see cref="Linear"/>: projects all N chunk positions against the same
    /// weight matrix in a single Parallel.For dispatch over the full N*outDim work item space,
    /// instead of N separate Linear() calls each launching (and tearing down) their own
    /// SimdKernels.MatVecF32-internal Parallel.For. During T3's ~400-token conditioning+text
    /// prefill this cuts the QKV/attn_output/FFN thread-pool dispatch count from N per layer down
    /// to 1 per layer (4 total per layer instead of up to ~1600), which matters because dispatch
    /// overhead was previously paid N times for work that's now scheduled once and load-balanced
    /// across all positions and output rows together.
    /// </summary>
    private static unsafe float[][] LinearBatched(float[][] inputs, float[] weight, float[] bias, int inDim, int outDim)
    {
        int n = inputs.Length;
        if (n == 1)
        {
            var singleOut = new float[outDim];
            fixed (float* w = weight, x = inputs[0], y = singleOut, b = bias)
            {
                SimdKernels.MatVecF32(y, w, x, outDim, inDim);
                for (int o = 0; o < outDim; o++) y[o] += b[o];
            }
            return [singleOut];
        }

        var outputs = new float[n][];
        for (int i = 0; i < n; i++) outputs[i] = new float[outDim];

        System.Threading.Tasks.Parallel.For(0, n, i =>
        {
            fixed (float* w = weight, x = inputs[i], y = outputs[i], b = bias)
            {
                SimdKernels.MatVecF32(y, w, x, outDim, inDim);
                for (int o = 0; o < outDim; o++) y[o] += b[o];
            }
        });

        return outputs;
    }

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f)
    {
        int n = x.Length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
        var /= n;
        float invStd = (float)(1.0 / Math.Sqrt(var + eps));

        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (float)((x[i] - mean) * invStd) * weight[i] + bias[i];
        return output;
    }

    /// <summary>float-only (no double promotion) in-place softmax over Span.</summary>
    private static void SoftmaxInPlace(Span<float> scores)
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

    private static void SoftmaxInPlace(float[] scores) => SoftmaxInPlace(scores.AsSpan());

    private static void GeluNewInPlace(float[] x)
    {
        // gelu_new (GPT2's tanh-approximation GELU): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            float inner = c * (v + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
    }

    // -----------------------------------------------------------------------
    // Fallback placeholder (used only when no real GGUF weights are available)
    // -----------------------------------------------------------------------

    private static List<int> GenerateFakePlaceholder(
        ReadOnlySpan<int> textTokens,
        float[] speakerFeatures,
        float temperature,
        int maxTokens)
    {
        var speechTokens = new List<int> { StartSpeechToken };
        var tokenCounts = new Dictionary<int, int>();
        tokenCounts[StartSpeechToken] = 1;

        int numText = textTokens.Length;
        int targetSpeechLength = Math.Clamp(numText * 6, 32, maxTokens);

        for (int step = 0; step < targetSpeechLength; step++)
        {
            float textBias = (step / 6 < numText) ? textTokens[step / 6] * 0.05f : 0f;
            float spkBias = (speakerFeatures != null && speakerFeatures.Length > 0)
                ? speakerFeatures[step % speakerFeatures.Length] * 0.1f
                : 0f;

            int bestToken = 100 + Math.Abs((int)(step * 37 + textBias * 100 + spkBias * 50)) % 1024;

            if (tokenCounts.TryGetValue(bestToken, out int count) && count > 0)
            {
                bestToken = (bestToken + 17) % 2048 + 100;
            }

            speechTokens.Add(bestToken);
            tokenCounts[bestToken] = tokenCounts.GetValueOrDefault(bestToken) + 1;
        }

        speechTokens.Add(StopSpeechToken);
        return speechTokens;
    }

    public void Dispose()
    {
        _weights?.Dispose();
    }
}
