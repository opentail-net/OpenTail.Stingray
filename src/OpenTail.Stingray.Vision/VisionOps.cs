using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// A resolved GGUF tensor: its full metadata (name, shape, dtype) plus the raw data pointer,
/// returned by <see cref="VisionOps.GetTensor"/>. Default (all-zero) value means "not found" --
/// check <see cref="IsValid"/> rather than comparing to a null tuple, since a pointer can't be a
/// generic type argument for <c>Nullable&lt;T&gt;</c>.
/// </summary>
public readonly unsafe struct VisionTensorRef(GgufTensorInfo info, byte* data)
{
    public readonly GgufTensorInfo Info = info;
    public readonly byte* Data = data;
    public bool IsValid => Data != null;
}

/// <summary>
/// High-performance shared neural compute kernels, tensor operations, and pointer helpers for Multimodal Vision Transformers.
/// </summary>
public static unsafe class VisionOps
{
    /// <summary>
    /// Loads a tensor from GGUF and dequantizes it to contiguous Float32 memory.
    /// Handles F32, F16, BF16, Q8_0, Q4_K, Q4_0, etc. seamlessly.
    /// </summary>
    public static float[]? LoadTensorF32(GgufModel gguf, params string[] candidateNames)
    {
        foreach (var name in candidateNames)
        {
            var tensorOpt = gguf.FindTensor(name);
            if (tensorOpt.HasValue)
            {
                var tensor = tensorOpt.Value;
                long elementCount = tensor.ElementCount;
                if (elementCount == 0) return null;

                var data = gguf.GetTensorData(tensor);
                var result = new float[elementCount];
                Dequantize.ToFloat32(data, result.AsSpan(), tensor.DType, elementCount);
                return result;
            }
        }
        return null;
    }

    /// <summary>
    /// Vectorized, multi-threaded Matrix-Vector multiplication using AVX2/AVX-512 TensorPrimitives.Dot with optional FP32 bias.
    /// Computes output[tokens, outDim] = input[tokens, inDim] * weights[outDim, inDim]^T + bias[outDim].
    /// </summary>
    public static void MatVec(
        float[] input,
        float[]? weights,
        float* bias,
        int nTokens,
        int inDim,
        int outDim,
        float[] output)
    {
        if (weights == null) return;

        Parallel.For(0, nTokens, t =>
        {
            int inOff = t * inDim;
            int outOff = t * outDim;
            var inSpan = new ReadOnlySpan<float>(input, inOff, inDim);

            for (int o = 0; o < outDim; o++)
            {
                float sum = bias != null ? bias[o] : 0f;
                int rowOff = o * inDim;
                sum += TensorPrimitives.Dot(inSpan, new ReadOnlySpan<float>(weights, rowOff, inDim));
                output[outOff + o] = sum;
            }
        });
    }

    /// <summary>
    /// Resolves a tensor's full GGUF info (name, shape, dtype) together with its raw data pointer,
    /// checking multiple fallback aliases. This is the structural replacement for the removed
    /// <c>GetTensorPtr&lt;Half&gt;</c> + <c>MatVecF16</c> pair: those threw away the tensor's actual
    /// dtype and handed callers a pointer blindly cast to a fixed CLR type, which is how a
    /// Q8_0-quantized mmproj got silently reinterpreted as raw F16 and corrupted memory (see
    /// docs/done/vl-untested-code-findings-2026-08-20.md and docs/done/vl-migration-plan-2026-08-20.md).
    /// Pair this with <see cref="MatVecAny"/>, which dispatches on the returned dtype instead of
    /// assuming one — the same pattern <c>OpenTail.Stingray.Cpu.SimdKernels.MatVec</c> already uses
    /// for the main LLM engine. Every vision encoder's matmul-bound weights (attention/FFN/proj) now
    /// go through this + <see cref="MatVecAny"/>. <see cref="GetTensorPtr{T}"/> (float-only in
    /// practice now) stays for norm/bias tensors, which are read element-wise rather than matvec'd
    /// and are genuinely always F32 in real mmproj files — it remains dtype-guarded either way.
    /// </summary>
    public static VisionTensorRef GetTensor(GgufModel gguf, params string[] candidateNames)
    {
        foreach (var name in candidateNames)
        {
            var tensor = gguf.FindTensor(name);
            if (tensor.HasValue)
                return new VisionTensorRef(tensor.Value, gguf.GetTensorDataPtr(tensor.Value));
        }
        return default;
    }

    /// <summary>
    /// Dtype-generic batched matrix-vector multiply for vision-encoder weights: computes
    /// <c>output[tokens, outDim] = input[tokens, inDim] * weights[outDim, inDim]^T + bias[outDim]</c>,
    /// dispatching per-dtype through <c>OpenTail.Stingray.Cpu.SimdKernels.MatVec</c> -- the same
    /// tested, dequant-in-register kernel set (F32/F16/BF16/Q8_0/Q4_K/Q6_K/Q5_K/Q3_K/Q2_K/Q4_0/...)
    /// the main LLM engine runs on, rather than a second, vision-only implementation that only
    /// covers F16 (<see cref="MatVecF16"/>'s gap). No-ops if <paramref name="weight"/> is null (a
    /// missing optional tensor), matching <see cref="MatVecF16"/>'s existing contract.
    /// </summary>
    /// <summary>STINGRAY_VISION_GEMM=0 disables the batched GEMM paths (per-token fallback) for A/B checks.</summary>
    private static readonly string? s_gemmEnv = Environment.GetEnvironmentVariable("STINGRAY_VISION_GEMM");
    private static readonly bool s_gemm = s_gemmEnv is not ("0" or "noattn-nomm" or "nomm");
    private static readonly bool s_gemmAttn = s_gemmEnv is not ("0" or "noattn-nomm" or "noattn");

    /// <summary>
    /// Rounds GEMM activations to the precision ggml's CPU dot products use for that weight type
    /// (<c>vec_dot_type</c>), so the batched path agrees with llama.cpp as closely as the per-token path does:
    /// Q8_0 weights -> activations quantized per 32 (<c>quantize_row_q8_0_ref</c>: d = amax/127 stored as F16,
    /// q = roundf(x/d)); F16 weights -> activations rounded to F16. Other types keep FP32 activations. Without
    /// this, the FP32-activation GEMM moved dots.ocr's projector output by 0.08 on a 2.6 value and Kimi-VL's
    /// sum by 3% (2026-09-27); the per-token SimdKernels path already quantizes Q8_0 activations the same way.
    /// Returns <paramref name="input"/> itself when no rounding applies.
    /// </summary>
    private static float[] RoundActivationsLikeGgml(float[] input, int nTokens, int inDim, DType wtype)
    {
        if (wtype == DType.Q8_0 && inDim % 32 == 0)
        {
            var r = new float[nTokens * inDim];
            Parallel.For(0, nTokens * inDim / 32, b =>
            {
                int o = b * 32;
                float amax = 0f;
                for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(input[o + j]));
                float d = amax / 127f;
                float id = d != 0f ? 1f / d : 0f;
                float dh = (float)(Half)d;
                for (int j = 0; j < 32; j++)
                    r[o + j] = MathF.Round(input[o + j] * id, MidpointRounding.AwayFromZero) * dh;
            });
            return r;
        }
        if (wtype == DType.Float16)
        {
            var r = new float[nTokens * inDim];
            Parallel.For(0, nTokens, t =>
            {
                int o = t * inDim;
                for (int j = 0; j < inDim; j++) r[o + j] = (float)(Half)input[o + j];
            });
            return r;
        }
        return input;
    }

    /// <summary>Token count from which <see cref="MatVecAny"/> uses the batched GEMM path.</summary>
    public const int MinTokensForGemm = 16;

    public static void MatVecAny(
        float[] input,
        VisionTensorRef weight,
        float* bias,
        int nTokens,
        int inDim,
        int outDim,
        float[] output)
    {
        if (!weight.IsValid) return;

        // Many tokens: one cache-blocked GEMM that dequantizes each weight panel once (PackedSgemmF32),
        // instead of a full weight pass per token. Measured 2026-09-27 on Step3-VL (47 layers, F16,
        // 4 x 1296 + 2704 patches): see docs/102 #8. Activations stay FP32 here.
        if (s_gemm && nTokens >= MinTokensForGemm && PackedSgemmF32.CanGemmStreaming(weight.Info.DType, inDim))
        {
            float[] act = RoundActivationsLikeGgml(input, nTokens, inDim, weight.Info.DType);
            fixed (float* pIn = act)
            fixed (float* pOut = output)
            {
                PackedSgemmF32.GemmQuant(pOut, pIn, weight.Data, weight.Info.DType, nTokens, outDim, inDim);
                if (bias != null)
                {
                    var b = new ReadOnlySpan<float>(bias, outDim);
                    for (int t = 0; t < nTokens; t++)
                    {
                        var row = new Span<float>(pOut + (long)t * outDim, outDim);
                        TensorPrimitives.Add(row, b, row);
                    }
                }
            }
            return;
        }

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            var inPtr = pIn;
            var outPtr = pOut;
            var data = weight.Data;
            var dtype = weight.Info.DType;
            var bPtr = bias;

            Parallel.For(0, nTokens, t =>
            {
                float* rowIn = inPtr + (long)t * inDim;
                float* rowOut = outPtr + (long)t * outDim;
                SimdKernels.MatVec(rowOut, data, rowIn, outDim, inDim, dtype);
                if (bPtr != null)
                {
                    var outSpan = new Span<float>(rowOut, outDim);
                    TensorPrimitives.Add(outSpan, new ReadOnlySpan<float>(bPtr, outDim), outSpan);
                }
            });
        }
    }

    /// <summary>
    /// Dequantizes a whole tensor to F32 once, for the (small) tensors an encoder reads per-element
    /// in an inline loop rather than through a batched matvec -- typically a patch-embed conv weight
    /// -- where <see cref="MatVecAny"/>'s per-call dtype dispatch doesn't apply. Empty array if the
    /// tensor is missing, so callers can keep an existing "length == 0 means absent" check instead
    /// of a separate null check.
    /// </summary>
    public static float[] DequantizeToFloat32(VisionTensorRef t)
    {
        if (!t.IsValid) return [];
        long n = t.Info.ElementCount;
        var dst = new float[n];
        Dequantize.ToFloat32(new ReadOnlySpan<byte>(t.Data, (int)t.Info.ByteSize), dst, t.Info.DType, n);
        return dst;
    }

    /// <summary>
    /// Resolves and dequantizes a norm/bias tensor to a managed, GC-owned <c>float[]</c>, checking
    /// multiple fallback aliases -- the safe replacement for <see cref="GetTensorPtr{T}"/> when a
    /// caller wants to stop holding a raw pointer as a long-lived field. Returns <c>null</c> (not an
    /// empty array) when the tensor isn't found: callers pin the result with a C# <c>fixed</c>
    /// statement to get a <c>float*</c> for the duration of a single kernel call, and <c>fixed</c>
    /// on a genuinely null array reference is guaranteed by the language spec to yield a null
    /// pointer -- fixing a zero-length (non-null) array is NOT guaranteed to yield null, so this
    /// deliberately does not reuse <see cref="DequantizeToFloat32"/>'s "empty means absent"
    /// convention here; getting that wrong would silently reintroduce a real memory-safety hazard.
    /// Two real differences from <see cref="GetTensorPtr{T}"/>: (1) no dtype exception -- like
    /// <see cref="DequantizeToFloat32"/>, this dequantizes whatever the tensor's real GGUF dtype is
    /// instead of demanding exactly Float32, so it also tolerates a quantized norm/bias tensor (not
    /// a hazard here the way the original Half-blind-cast bug was, since <c>Dequantize.ToFloat32</c>
    /// is the same dtype-aware decoder every matmul weight already goes through via
    /// <see cref="MatVecAny"/>); (2) the returned array is a one-time copy owned by the caller, with
    /// a lifetime tied to the encoder instance rather than to the GGUF's memory mapping, and is
    /// bounds-checked like any other managed array. Intended for norm weights/biases specifically --
    /// these are O(dim) elements (a few KB at most), so the one-time copy is cheap; do not use this
    /// for matmul-bound weight tensors (attention/FFN/proj), which stay on <see cref="GetTensor"/> +
    /// <see cref="MatVecAny"/> to avoid materializing GB-scale tensors off the memory-mapped file.
    /// </summary>
    /// <summary>A float array stored as GGUF metadata (e.g. <c>clip.vision.image_mean</c>); null when absent.</summary>
    public static float[]? GetTensorArrayOrMeta(GgufModel gguf, string key)
    {
        if (!gguf.Metadata.TryGetValue(key, out var v) || v is not System.Collections.IList list) return null;
        var r = new float[list.Count];
        for (int i = 0; i < r.Length; i++) r[i] = Convert.ToSingle(list[i]);
        return r;
    }

    public static float[]? GetTensorArray(GgufModel gguf, params string[] candidateNames)
    {
        var t = GetTensor(gguf, candidateNames);
        return t.IsValid ? DequantizeToFloat32(t) : null;
    }

    /// <summary>
    /// Parallelized Layer Normalization over the last dimension: (x - mean) / sqrt(var + eps) * weight + bias.
    /// </summary>
    public static void LayerNorm(
        float[] states,
        int nTokens,
        int dim,
        float* weights,
        float* bias,
        float eps = 1e-6f)
    {
        Parallel.For(0, nTokens, t =>
        {
            int off = t * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += states[off + d];
            mean /= dim;

            float var = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = states[off + d] - mean;
                var += diff * diff;
            }
            float std = MathF.Sqrt(var / dim + eps);

            for (int d = 0; d < dim; d++)
            {
                float w = weights != null ? weights[d] : 1f;
                float b = bias != null ? bias[d] : 0f;
                states[off + d] = ((states[off + d] - mean) / std) * w + b;
            }
        });
    }

    /// <summary>
    /// Parallelized Root-Mean-Square Normalization (RMSNorm): x / sqrt(mean(x^2) + eps) * weight.
    /// </summary>
    public static void RmsNorm(
        float[] states,
        int nTokens,
        int dim,
        float* weights,
        float eps = 1e-6f)
    {
        Parallel.For(0, nTokens, t =>
        {
            int off = t * dim;
            float sumSq = 0f;
            for (int d = 0; d < dim; d++)
            {
                float val = states[off + d];
                sumSq += val * val;
            }
            float rms = MathF.Sqrt(sumSq / dim + eps);

            for (int d = 0; d < dim; d++)
            {
                float w = weights != null ? weights[d] : 1f;
                states[off + d] = (states[off + d] / rms) * w;
            }
        });
    }

    /// <summary>
    /// Parallel multi-head scaled dot-product self-attention mechanism with softmax.
    /// Q, K, V have shape [nTokens, heads * headDim]. Output has shape [nTokens, heads * headDim].
    /// </summary>
    public static void Attention(
        float[] q,
        float[] k,
        float[] v,
        int nTokens,
        int heads,
        int headDim,
        float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int embd = heads * headDim;

        if (s_gemmAttn && nTokens >= MinTokensForGemmAttention && PackedSgemmF32.IsSupported)
        {
            AttentionGemm(q, k, v, nTokens, heads, headDim, scale, output);
            return;
        }

        // Vectorized via TensorPrimitives (Dot/MultiplyAdd) instead of scalar accumulation
        // loops -- same algorithm as Gemma3VisionEncoder's hand-rolled attention (see that
        // class's doc comment: at large token counts, scalar loops here were measured as "far
        // too slow", which is why that encoder never used this shared method). Bringing this
        // method up to the same technique lets every OTHER caller (Pixtral, and any future
        // encoder) get that speedup for free, rather than each hand-rolling its own fast path.
        // See docs/done/vision-attention-vectorization-2026-08-20.md for the measurement.
        // The V-weighted sum uses a single fused MultiplyAdd (out = v*prob + out) instead of a
        // separate Multiply-into-temp + Add pass -- one vectorized pass instead of two, no temp
        // buffer, for the single hottest inner loop here (O(heads*nTokens^2) iterations).
        Parallel.For(0, heads, h =>
        {
            int headOff = h * headDim;
            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int qOff = i * embd + headOff;
                var qi = new ReadOnlySpan<float>(q, qOff, headDim);

                // Compute Q_i . K_j * scale
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    int kOff = j * embd + headOff;
                    var kj = new ReadOnlySpan<float>(k, kOff, headDim);
                    float s = TensorPrimitives.Dot(qi, kj) * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                // Numerically stable Softmax
                float expSum = 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;
                var scoresSpan = new Span<float>(scores);
                TensorPrimitives.Multiply(scoresSpan, invSum, scoresSpan);

                // Aggregate V_j * prob
                int outOff = i * embd + headOff;
                var outSpan = new Span<float>(output, outOff, headDim);
                outSpan.Clear();
                for (int j = 0; j < nTokens; j++)
                {
                    int vOff = j * embd + headOff;
                    var vj = new ReadOnlySpan<float>(v, vOff, headDim);
                    TensorPrimitives.MultiplyAdd(vj, scores[j], outSpan, outSpan);
                }
            }
        });
    }

    /// <summary>
    /// Parallel multi-head grouped-query attention (GQA) mechanism with softmax and optional sink bias.
    /// Q has shape [nTokens, qHeads * headDim]. K, V have shape [nTokens, kvHeads * headDim].
    /// Output has shape [nTokens, qHeads * headDim].
    /// </summary>
    /// <summary>Token count from which <see cref="Attention"/> uses <see cref="AttentionGemm"/>.</summary>
    public const int MinTokensForGemmAttention = 256;

    /// <summary>
    /// Full (unmasked) multi-head attention as two GEMMs per head: S = Q_h K_h^T (scaled), a row softmax,
    /// then O_h = P V_h, both through <see cref="PackedSgemmF32"/>. Same math as the per-pair loop in
    /// <see cref="Attention"/>; used for large token counts, where the loop's n^2 small dot products dominate
    /// (Step3-VL overview: 2704 patches x 47 layers, measured in docs/102 #8, 2026-09-27).
    /// </summary>
    public static void AttentionGemm(float[] q, float[] k, float[] v, int n, int heads, int headDim, float scale, float[] output)
    {
        int embd = heads * headDim;
        var qh = new float[n * headDim];
        var kh = new float[n * headDim];
        var vt = new float[headDim * n];
        var scores = new float[(long)n * n];
        var oh = new float[n * headDim];
        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            Parallel.For(0, n, t =>
            {
                Array.Copy(q, t * embd + off, qh, t * headDim, headDim);
                Array.Copy(k, t * embd + off, kh, t * headDim, headDim);
                for (int d = 0; d < headDim; d++) vt[d * n + t] = v[t * embd + off + d];
            });
            fixed (float* pq = qh, pk = kh, pvt = vt, ps = scores, po = oh)
            {
                float* packedK = PackedSgemmF32.PackWeights(pk, n, headDim);
                try { PackedSgemmF32.Gemm(ps, pq, packedK, null, n, n, headDim); }
                finally { NativeMemory.AlignedFree(packedK); }

                nint sp = (nint)ps;
                Parallel.For(0, n, i =>
                {
                    var row = new Span<float>((float*)sp + (long)i * n, n);
                    TensorPrimitives.Multiply(row, scale, row);
                    float max = TensorPrimitives.Max(row);
                    TensorPrimitives.Subtract(row, max, row);
                    TensorPrimitives.Exp(row, row);
                    float sum = TensorPrimitives.Sum(row);
                    TensorPrimitives.Multiply(row, sum > 0f ? 1f / sum : 0f, row);
                });

                float* packedV = PackedSgemmF32.PackWeights(pvt, headDim, n);
                try { PackedSgemmF32.Gemm(po, ps, packedV, null, n, headDim, n); }
                finally { NativeMemory.AlignedFree(packedV); }
            }
            Parallel.For(0, n, t => Array.Copy(oh, t * headDim, output, t * embd + off, headDim));
        }
    }

    public static void AttentionGqa(
        float[] q,
        float[] k,
        float[] v,
        int nTokens,
        int qHeads,
        int kvHeads,
        int headDim,
        float[] output,
        float* attnSinks = null)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int groupSize = qHeads / kvHeads;
        int qEmbd = qHeads * headDim;
        int kvEmbd = kvHeads * headDim;

        Parallel.For(0, qHeads, qh =>
        {
            int kvHead = qh / groupSize;
            int qOffHead = qh * headDim;
            int kvOffHead = kvHead * headDim;
            float sink = (attnSinks != null) ? attnSinks[qh] : 0f;

            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int qOff = i * qEmbd + qOffHead;
                var qi = new ReadOnlySpan<float>(q, qOff, headDim);

                float maxScore = sink != 0f ? sink : float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    int kOff = j * kvEmbd + kvOffHead;
                    var kj = new ReadOnlySpan<float>(k, kOff, headDim);
                    float s = TensorPrimitives.Dot(qi, kj) * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float expSum = sink != 0f ? MathF.Exp(sink - maxScore) : 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;
                // The sink's mass (if any) is included in expSum/invSum above but was never
                // written into `scores`, so normalizing scores in place here still correctly
                // leaves the sink's share unaccounted-for in the V-weighted sum below -- same
                // "extra probability mass with no output contribution" effect as the original
                // acc*invSum scaling, just applied earlier.
                var scoresSpan = new Span<float>(scores);
                TensorPrimitives.Multiply(scoresSpan, invSum, scoresSpan);

                // Single fused MultiplyAdd instead of Multiply-into-temp + Add -- see Attention()
                // above for the same fix and rationale.
                int outOff = i * qEmbd + qOffHead;
                var outSpan = new Span<float>(output, outOff, headDim);
                outSpan.Clear();
                for (int j = 0; j < nTokens; j++)
                {
                    int vOff = j * kvEmbd + kvOffHead;
                    var vj = new ReadOnlySpan<float>(v, vOff, headDim);
                    TensorPrimitives.MultiplyAdd(vj, scores[j], outSpan, outSpan);
                }
            }
        });
    }

    /// <summary>
    /// Windowed/local variant of <see cref="AttentionGqa"/>: identical scaled dot-product GQA
    /// softmax attention, except a query token at index i may only attend to a key token at index
    /// j when <c>windowId[i] == windowId[j]</c> (all other scores are masked to -infinity before
    /// softmax). Matches the real llama.cpp reference's window_mask semantics
    /// (tools/mtmd/clip.cpp's PROJECTOR_TYPE_QWEN25VL/EXAONE4_5 case): the real C++ builds a
    /// full [n_tok,n_tok] additive mask and reorders tokens into window-contiguous blocks purely
    /// as an implementation/perf detail for a dense matmul-based attention kernel; computing
    /// window membership directly from each token's real spatial window id (as done here) is
    /// mathematically identical and needs no token reordering, matching this project's own
    /// GLM-4.6V finding that a real spatial-index-based approach is a legitimate substitute for
    /// reindex-based real reference code as long as membership is derived from real coordinates.
    /// </summary>
    public static void AttentionGqaWindowed(
        float[] q,
        float[] k,
        float[] v,
        int nTokens,
        int qHeads,
        int kvHeads,
        int headDim,
        float[] output,
        int[] windowId)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int groupSize = qHeads / kvHeads;
        int qEmbd = qHeads * headDim;
        int kvEmbd = kvHeads * headDim;

        Parallel.For(0, qHeads, qh =>
        {
            int kvHead = qh / groupSize;
            int qOffHead = qh * headDim;
            int kvOffHead = kvHead * headDim;

            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int wi = windowId[i];
                int qOff = i * qEmbd + qOffHead;
                var qi = new ReadOnlySpan<float>(q, qOff, headDim);

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    if (windowId[j] != wi) { scores[j] = float.NegativeInfinity; continue; }
                    int kOff = j * kvEmbd + kvOffHead;
                    var kj = new ReadOnlySpan<float>(k, kOff, headDim);
                    float s = TensorPrimitives.Dot(qi, kj) * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float expSum = 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = scores[j] == float.NegativeInfinity ? 0f : MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;
                var scoresSpan = new Span<float>(scores);
                TensorPrimitives.Multiply(scoresSpan, invSum, scoresSpan);

                int outOff = i * qEmbd + qOffHead;
                var outSpan = new Span<float>(output, outOff, headDim);
                outSpan.Clear();
                for (int j = 0; j < nTokens; j++)
                {
                    if (scores[j] == 0f) continue;
                    int vOff = j * kvEmbd + kvOffHead;
                    var vj = new ReadOnlySpan<float>(v, vOff, headDim);
                    TensorPrimitives.MultiplyAdd(vj, scores[j], outSpan, outSpan);
                }
            }
        });
    }

    /// <summary>
    /// Batched scaled dot-product attention where the token set is partitioned into independent
    /// windows: query token <c>w*qLen+i</c> may only attend to key/value tokens
    /// <c>w*kvLen+0..kvLen-1</c> of the SAME window <c>w</c> -- i.e. <paramref name="nWindows"/>
    /// fully independent attention groups batched into one call, rather than one dense
    /// [nTok,nTok] softmax. Covers both self-attention (pass the same buffer as q/k/v with
    /// qLen == kvLen) and cross-attention (q from one stream, k/v from another, qLen != kvLen
    /// allowed) -- used by Granite 4 Vision's WindowQFormer projector blocks, whose real
    /// reference (examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp,
    /// build_block()) reshapes Q/K/V to a 4th "window" ne[3] dimension and runs ggml's batched
    /// attention op per-window; looping windows directly here is mathematically identical and
    /// avoids materializing the reshape.
    /// Layout: q is [nWindows*qLen, heads*headDim] (window-major, i.e. row w*qLen+i); k/v are
    /// [nWindows*kvLen, heads*headDim] (row w*kvLen+j); output matches q's shape.
    /// </summary>
    public static void WindowedAttention(
        float[] q,
        float[] k,
        float[] v,
        int nWindows,
        int qLen,
        int kvLen,
        int heads,
        int headDim,
        float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int embd = heads * headDim;

        Parallel.For(0, nWindows * heads, wh =>
        {
            int w = wh / heads;
            int h = wh % heads;
            int headOff = h * headDim;
            int qBase = w * qLen;
            int kvBase = w * kvLen;

            var scores = new float[kvLen];

            for (int i = 0; i < qLen; i++)
            {
                int qOff = (qBase + i) * embd + headOff;
                var qi = new ReadOnlySpan<float>(q, qOff, headDim);

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < kvLen; j++)
                {
                    int kOff = (kvBase + j) * embd + headOff;
                    var kj = new ReadOnlySpan<float>(k, kOff, headDim);
                    float s = TensorPrimitives.Dot(qi, kj) * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float expSum = 0f;
                for (int j = 0; j < kvLen; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;
                var scoresSpan = new Span<float>(scores);
                TensorPrimitives.Multiply(scoresSpan, invSum, scoresSpan);

                int outOff = (qBase + i) * embd + headOff;
                var outSpan = new Span<float>(output, outOff, headDim);
                outSpan.Clear();
                for (int j = 0; j < kvLen; j++)
                {
                    int vOff = (kvBase + j) * embd + headOff;
                    var vj = new ReadOnlySpan<float>(v, vOff, headDim);
                    TensorPrimitives.MultiplyAdd(vj, scores[j], outSpan, outSpan);
                }
            }
        });
    }

    /// <summary>
    /// 2D average-pool spatial downsampling from a (side x side) raster grid of tokens to a
    /// (newSide x newSide) grid, kernel = side/newSide, matching the real reference's
    /// <c>interp_down()</c> (examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp
    /// lines 34-45: reshape to (embd,side,side,1) -> permute channel-last -> ggml_pool_2d AVG
    /// with kernel=stride=side/newSide -> permute back). Row-major token layout: input row
    /// y*side+x, output row oy*newSide+ox, matching this project's "gather" index convention
    /// (<c>make_win_idx</c> in clip.cpp uses the same y*side+x addressing) elsewhere in the same
    /// projector.
    /// </summary>
    public static void AvgPoolDownsample2D(
        float[] src,
        int side,
        int newSide,
        int dim,
        float[] dst)
    {
        int kernel = side / newSide;
        float inv = 1.0f / (kernel * kernel);

        Parallel.For(0, newSide * newSide, oIdx =>
        {
            int oy = oIdx / newSide;
            int ox = oIdx % newSide;
            var acc = new float[dim];

            for (int ky = 0; ky < kernel; ky++)
            {
                int y = oy * kernel + ky;
                for (int kx = 0; kx < kernel; kx++)
                {
                    int x = ox * kernel + kx;
                    int srcOff = (y * side + x) * dim;
                    TensorPrimitives.Add(new ReadOnlySpan<float>(acc), new ReadOnlySpan<float>(src, srcOff, dim), acc);
                }
            }

            int dstOff = oIdx * dim;
            var dstSpan = new Span<float>(dst, dstOff, dim);
            TensorPrimitives.Multiply(new ReadOnlySpan<float>(acc), inv, dstSpan);
        });
    }

    /// <summary>
    /// Gathers whole rows: <c>dst[i,:] = src[idx[i],:]</c> for each of <paramref name="idx"/>'s
    /// entries, matching <c>ggml_get_rows</c> as used throughout Granite 4 Vision's WindowQFormer
    /// projector for its window/unwindow/spatial-checkerboard permutations (all precomputed index
    /// arrays, not learned -- see the real reference's <c>gather()</c> helper in
    /// tools/mtmd/models/granite4-vision.cpp lines 24-32, and the actual index-construction math
    /// in tools/mtmd/clip.cpp lines 5144-5183, which this project's own index builders mirror).
    /// </summary>
    public static void GatherRows(float[] src, int[] idx, int dim, float[] dst)
    {
        Parallel.For(0, idx.Length, i =>
        {
            Array.Copy(src, (long)idx[i] * dim, dst, (long)i * dim, dim);
        });
    }

    /// <summary>
    /// Applies Gaussian Error Linear Unit (GELU) with Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GeluScalar(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
    }

    /// <summary>
    /// Applies GELU elementwise across a span in-place.
    /// </summary>
    public static void Gelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = GeluScalar(data[i]);
        }
    }

    /// <summary>
    /// Applies QuickGELU activation function: x * sigmoid(1.702 * x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float QuickGeluScalar(float x)
    {
        return x * (1.0f / (1.0f + MathF.Exp(-1.702f * x)));
    }

    /// <summary>
    /// Applies QuickGELU elementwise across a span in-place.
    /// </summary>
    public static void QuickGelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = QuickGeluScalar(data[i]);
        }
    }

    /// <summary>
    /// Applies SiLU (Swish) activation function: x * sigmoid(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SiluScalar(float x)
    {
        return x / (1.0f + MathF.Exp(-x));
    }

    /// <summary>
    /// Applies SiLU elementwise across a span in-place.
    /// </summary>
    public static void Silu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = SiluScalar(data[i]);
        }
    }

    /// <summary>
    /// Applies Squared ReLU activation: (max(0, x))^2.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SquaredReluScalar(float x)
    {
        float relu = MathF.Max(0f, x);
        return relu * relu;
    }

    /// <summary>
    /// Applies Squared ReLU elementwise across a span in-place.
    /// </summary>
    public static void SquaredRelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = SquaredReluScalar(data[i]);
        }
    }

    /// <summary>
    /// Continuous 2D Rotary Position Embedding (Pixtral-style). Confirmed against the real
    /// llama.cpp reference (<c>clip.cpp</c>'s <c>build_rope_2d</c> plus its real <c>pos_h</c>/
    /// <c>pos_w</c> fill: <c>pos_h[i] = i / n_patches_per_col</c> (row), <c>pos_w[i] = i %
    /// n_patches_per_col</c> (col), and <c>build_rope_2d(cur, pos_h, pos_w, ...)</c> feeds
    /// <c>pos_h</c> (row) to the FIRST half and <c>pos_w</c> (col) to the SECOND half) --
    /// two real, independent bugs found and fixed here (2026-09-01, found chasing a real
    /// numeric mismatch against scripts/pixtral_ref.py, a from-scratch port of the same real
    /// C++): (1) this method previously rotated the FIRST half by column (X) and the SECOND by
    /// row (Y) -- backwards; (2) the SECOND half is missing a real extra frequency scale,
    /// <c>freq_scale_odd = theta^(-2/headDim)</c> (applied via `ggml_rope_ext`'s own
    /// `freq_scale` parameter in the real call, present only for the second/odd half -- see the
    /// C++'s own comment: "then for the second half, we use freq_scale to shift the inv_freq").
    /// Without both fixes the two RoPE halves rotate by the wrong axis at the wrong rate, which
    /// is structural enough to completely decorrelate the embedding (measured: per-token cosine
    /// similarity against the real reference went from -0.02 -- i.e. no better than
    /// uncorrelated -- to a match after fixing both).
    /// </summary>
    public static void Continuous2DRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int heads,
        int headDim,
        float theta = 10000.0f)
    {
        int halfDim = headDim / 2;
        int quarterDim = halfDim / 2;
        float freqScaleOdd = MathF.Pow(theta, -2f / headDim);

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int tokenIdx = py * patchesX + px;

                for (int h = 0; h < heads; h++)
                {
                    int headOff = (tokenIdx * heads + h) * headDim;

                    // First half: rotate by ROW position (py), plain per-dim frequency.
                    for (int d = 0; d < quarterDim; d++)
                    {
                        float freq = MathF.Pow(theta, -(float)(2 * d) / halfDim);
                        float angle = py * freq;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int i0 = headOff + d * 2;
                        int i1 = headOff + d * 2 + 1;

                        float q0 = q[i0], q1 = q[i1];
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q0 * sin + q1 * cos;

                        float k0 = k[i0], k1 = k[i1];
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k0 * sin + k1 * cos;
                    }

                    // Second half: rotate by COLUMN position (px), with the extra
                    // freq_scale_odd factor real llama.cpp applies only to this half.
                    for (int d = 0; d < quarterDim; d++)
                    {
                        float freq = MathF.Pow(theta, -(float)(2 * d) / halfDim);
                        float angle = px * freq * freqScaleOdd;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int i0 = headOff + halfDim + d * 2;
                        int i1 = headOff + halfDim + d * 2 + 1;

                        float q0 = q[i0], q1 = q[i1];
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q0 * sin + q1 * cos;

                        float k0 = k[i0], k1 = k[i1];
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k0 * sin + k1 * cos;
                    }
                }
            }
        });
    }

    /// <summary>
    /// Interleaved 2D Rotary Position Embedding (Kimi / GLM style).
    /// </summary>
    public static void Interleaved2DRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int heads,
        int headDim,
        float theta = 10000.0f)
    {
        int halfDim = headDim / 2;

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int tokenIdx = py * patchesX + px;

                for (int h = 0; h < heads; h++)
                {
                    int headOff = (tokenIdx * heads + h) * headDim;

                    for (int d = 0; d < halfDim; d++)
                    {
                        float freq = MathF.Pow(theta, -(float)(2 * d) / headDim);
                        float angle = ((d % 2 == 0) ? px : py) * freq;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int i0 = headOff + d * 2;
                        int i1 = headOff + d * 2 + 1;

                        float q0 = q[i0], q1 = q[i1];
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q0 * sin + q1 * cos;

                        float k0 = k[i0], k1 = k[i1];
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k0 * sin + k1 * cos;
                    }
                }
            }
        });
    }

    /// <summary>
    /// Multimodal Rotary Position Embedding (M-RoPE, Qwen/MiMo/Exaone style) -- shared by
    /// Exaone4VisionEncoder and MimoVlVisionEncoder (Qwen2VL/GLM4V's own encoders each keep a
    /// private near-identical copy of this same fix; see Glm4VisionEncoder.ApplyMrope's doc
    /// comment for the full derivation from ggml_mrope_cache_init + GGML_ROPE_TYPE_VISION's
    /// rotate_pairs in ggml-cpu/ops.cpp).
    ///
    /// The real reference (ggml_rope_multi, sections all = headDim/4, n_dims=headDim/2) only ever
    /// selects 2 of its 4 declared position channels in practice: pairs span the index ic in
    /// [0,headDim/2) with its +headDim/2 partner (covering the FULL head_dim, not two disjoint
    /// local quarter-pairs as this method previously implemented) -- first quarter of that range
    /// rotates by row/Y, second quarter by column/X.
    /// </summary>
    public static void ApplyMRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int qHeads,
        int kvHeads,
        int headDim,
        float theta = 10000.0f,
        bool independentSections = false)
    {
        int half = headDim / 2;
        int quarter = headDim / 4;
        // ggml GGML_ROPE_TYPE_VISION computes each position section's angles independently
        // (ggml_mrope_cache_init indep_sects): the column quarter restarts at theta^0 instead of
        // continuing the row quarter's frequency ladder. Found 2026-09-26 on dots.ocr against
        // llama-mtmd-debug (Qcur_pos sum -657 ref vs +622 without the reset).

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;

                for (int h = 0; h < qHeads; h++)
                {
                    int headOff = (p * qHeads + h) * headDim;
                    for (int ic = 0; ic < half; ic++)
                    {
                        float pos = ic < quarter ? py : px;
                        int fi = independentSections && ic >= quarter ? ic - quarter : ic;
                        float freq = MathF.Pow(theta, -4.0f * fi / headDim);
                        float th = pos * freq;
                        float cosT = MathF.Cos(th);
                        float sinT = MathF.Sin(th);

                        float q0 = q[headOff + ic];
                        float q1 = q[headOff + ic + half];
                        q[headOff + ic] = q0 * cosT - q1 * sinT;
                        q[headOff + ic + half] = q0 * sinT + q1 * cosT;
                    }
                }

                for (int h = 0; h < kvHeads; h++)
                {
                    int headOff = (p * kvHeads + h) * headDim;
                    for (int ic = 0; ic < half; ic++)
                    {
                        float pos = ic < quarter ? py : px;
                        int fi = independentSections && ic >= quarter ? ic - quarter : ic;
                        float freq = MathF.Pow(theta, -4.0f * fi / headDim);
                        float th = pos * freq;
                        float cosT = MathF.Cos(th);
                        float sinT = MathF.Sin(th);

                        float k0 = k[headOff + ic];
                        float k1 = k[headOff + ic + half];
                        k[headOff + ic] = k0 * cosT - k1 * sinT;
                        k[headOff + ic + half] = k0 * sinT + k1 * cosT;
                    }
                }
            }
        });
    }

    public static void ApplyMRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int heads,
        int headDim,
        float theta = 10000.0f)
        => ApplyMRoPE(q, k, patchesX, patchesY, heads, heads, headDim, theta);

    /// <summary>
    /// 2D RoPE as llama.cpp's <c>clip_graph::build_rope_2d(cur, pos_w, pos_h, theta, interleave_freq: false)</c>
    /// (Kimi-VL, Step3-VL): the first half of each head is rotated by the patch COLUMN, the second half by
    /// the patch ROW; each half is an ordinary adjacent-pair RoPE over <c>headDim/2</c> dims with its own
    /// ladder <c>theta^(-2i/(headDim/2))</c>. Tokens are raster order (x fastest).
    /// </summary>
    public static void Rope2dHalves(float[] q, float[] k, int patchesX, int patchesY, int heads, int headDim, float theta)
    {
        int half = headDim / 2;
        int pairs = half / 2;
        var freqs = new float[pairs];
        for (int i = 0; i < pairs; i++) freqs[i] = MathF.Pow(theta, -2.0f * i / half);

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;
                for (int h = 0; h < heads; h++)
                {
                    int headOff = (p * heads + h) * headDim;
                    for (int part = 0; part < 2; part++)
                    {
                        float pos = part == 0 ? px : py;
                        int baseOff = headOff + part * half;
                        for (int i = 0; i < pairs; i++)
                        {
                            float a = pos * freqs[i];
                            float c = MathF.Cos(a), sn = MathF.Sin(a);
                            int o = baseOff + 2 * i;
                            float q0 = q[o], q1 = q[o + 1];
                            q[o] = q0 * c - q1 * sn; q[o + 1] = q0 * sn + q1 * c;
                            float k0 = k[o], k1 = k[o + 1];
                            k[o] = k0 * c - k1 * sn; k[o + 1] = k0 * sn + k1 * c;
                        }
                    }
                }
            }
        });
    }

    /// <summary>
    /// im2col for a 2D conv over a channels-last <paramref name="h"/> x <paramref name="w"/> x <paramref name="cin"/>
    /// grid (raster tokens). Each output row is <c>cin*k*k</c> long in (ci, ky, kx) order, matching a GGUF conv
    /// weight <c>[kw, kh, cin, cout]</c> read as <c>cout</c> rows, so <see cref="MatVecAny"/> / <see cref="MatVec"/>
    /// finish the conv. Zero padding.
    /// </summary>
    public static float[] Im2Col(float[] input, int h, int w, int cin, int k, int stride, int pad, out int ho, out int wo)
    {
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        ho = oh; wo = ow;
        int kk = cin * k * k;
        var cols = new float[oh * ow * kk];
        Parallel.For(0, oh, oy =>
        {
            for (int ox = 0; ox < ow; ox++)
            {
                int o = (oy * ow + ox) * kk;
                for (int ky = 0; ky < k; ky++)
                {
                    int iy = oy * stride + ky - pad;
                    for (int kx = 0; kx < k; kx++)
                    {
                        int ix = ox * stride + kx - pad;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                        int src = (iy * w + ix) * cin;
                        for (int ci = 0; ci < cin; ci++)
                            cols[o + (ci * k + ky) * k + kx] = input[src + ci];
                    }
                }
            }
        });
        return cols;
    }

    /// <summary>
    /// A learned square position table (<c>side x side</c> rows of <paramref name="embd"/>) resized to a
    /// <paramref name="width"/> x <paramref name="height"/> patch grid exactly as llama.cpp's
    /// <c>clip_graph::resize_position_embeddings()</c>: <c>ggml_interpolate</c> with
    /// <c>GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS</c> (PyTorch bilinear, align_corners=False,
    /// antialias=True: triangle filter, support widened when downscaling). Returns the table itself when
    /// the grid already matches. Used by Kimi-VL and PaddleOCR-VL (2026-09-27).
    /// </summary>
    public static float[] ResizePositionEmbeddings(float[] table, int embd, int width, int height)
    {
        if (table.Length == 0) return [];
        int side = (int)Math.Round(Math.Sqrt(table.Length / embd));
        if (width == side && height == side) return table;

        var dst = new float[width * height * embd];
        float sf0 = (float)width / side, sf1 = (float)height / side;
        float support0 = Math.Max(1f, 1f / sf0), invScale0 = 1f / support0;
        float support1 = Math.Max(1f, 1f / sf1), invScale1 = 1f / support1;
        const float off = 0.5f;
        static float Tri(float x) => Math.Max(1f - MathF.Abs(x), 0f);
        var src = table;

        Parallel.For(0, height, i1 =>
        {
            float y = (i1 + off) / sf1;
            int yMin = Math.Max((int)(y - support1 + off), 0);
            int yMax = Math.Min((int)(y + support1 + off), side);
            var acc = new float[embd];
            for (int i0 = 0; i0 < width; i0++)
            {
                float x = (i0 + off) / sf0;
                int xMin = Math.Max((int)(x - support0 + off), 0);
                int xMax = Math.Min((int)(x + support0 + off), side);
                Array.Clear(acc);
                float total = 0f;
                for (int sy = yMin; sy < yMax; sy++)
                {
                    float wy = Tri((sy - y + off) * invScale1);
                    for (int sx = xMin; sx < xMax; sx++)
                    {
                        float w = Tri((sx - x + off) * invScale0) * wy;
                        if (w <= 0f) continue;
                        int so = (sy * side + sx) * embd;
                        for (int d = 0; d < embd; d++) acc[d] += src[so + d] * w;
                        total += w;
                    }
                }
                int dOff = (i1 * width + i0) * embd;
                float inv = total > 0f ? 1f / total : 1f;
                for (int d = 0; d < embd; d++) dst[dOff + d] = acc[d] * inv;
            }
        });
        return dst;
    }

    /// <summary>
    /// PixelShuffle 2x2 spatial downsampler: merges 2x2 spatial blocks into 4x channel dimensions.
    /// Input: [gridY, gridX, inDim], Output: [gridY/2, gridX/2, 4*inDim].
    /// </summary>
    public static void PixelShuffle2x2(
        float[] input,
        int gridY,
        int gridX,
        int inDim,
        float[] output)
    {
        int outH = gridY / 2;
        int outW = gridX / 2;
        int outDim = inDim * 4;

        Parallel.For(0, outH, oy =>
        {
            for (int ox = 0; ox < outW; ox++)
            {
                int outTokenIdx = oy * outW + ox;
                int outOffset = outTokenIdx * outDim;

                int iy = oy * 2;
                int ix = ox * 2;

                int inIdx00 = (iy * gridX + ix) * inDim;
                int inIdx01 = (iy * gridX + (ix + 1)) * inDim;
                int inIdx10 = ((iy + 1) * gridX + ix) * inDim;
                int inIdx11 = ((iy + 1) * gridX + (ix + 1)) * inDim;

                Array.Copy(input, inIdx00, output, outOffset, inDim);
                Array.Copy(input, inIdx01, output, outOffset + inDim, inDim);
                Array.Copy(input, inIdx10, output, outOffset + 2 * inDim, inDim);
                Array.Copy(input, inIdx11, output, outOffset + 3 * inDim, inDim);
            }
        });
    }

    /// <summary>
    /// 3D Spatio-Temporal RoPE for video/temporal transformer architectures.
    /// </summary>
    public static void ApplyRoPE3D(
        float[] q,
        float[] k,
        int numTokens,
        int numHeads,
        int headDim,
        int tDim,
        int hDim,
        int wDim,
        float theta = 10000.0f)
    {
        int bandSize = headDim / 6;
        int hw = hDim * wDim;

        Parallel.For(0, numTokens, tokenIdx =>
        {
            int pt = tokenIdx / (hw > 0 ? hw : 1);
            int rem = tokenIdx % (hw > 0 ? hw : 1);
            int py = rem / (wDim > 0 ? wDim : 1);
            int px = rem % (wDim > 0 ? wDim : 1);

            for (int h = 0; h < numHeads; h++)
            {
                int headOff = (tokenIdx * numHeads + h) * headDim;

                // Temporal
                for (int d = 0; d < bandSize; d++)
                {
                    float freqT = MathF.Pow(theta, -2.0f * d / (bandSize * 2));
                    float cosT = MathF.Cos(pt * freqT);
                    float sinT = MathF.Sin(pt * freqT);

                    int i0 = headOff + d;
                    int i1 = headOff + d + bandSize;
                    float q0 = q[i0], q1 = q[i1];
                    q[i0] = q0 * cosT - q1 * sinT;
                    q[i1] = q0 * sinT + q1 * cosT;

                    float k0 = k[i0], k1 = k[i1];
                    k[i0] = k0 * cosT - k1 * sinT;
                    k[i1] = k0 * sinT + k1 * cosT;
                }

                // Height/Y
                for (int d = 0; d < bandSize; d++)
                {
                    float freqY = MathF.Pow(theta, -2.0f * d / (bandSize * 2));
                    float cosY = MathF.Cos(py * freqY);
                    float sinY = MathF.Sin(py * freqY);

                    int i0 = headOff + 2 * bandSize + d;
                    int i1 = headOff + 3 * bandSize + d;
                    float q0 = q[i0], q1 = q[i1];
                    q[i0] = q0 * cosY - q1 * sinY;
                    q[i1] = q0 * sinY + q1 * cosY;

                    float k0 = k[i0], k1 = k[i1];
                    k[i0] = k0 * cosY - k1 * sinY;
                    k[i1] = k0 * sinY + k1 * cosY;
                }

                // Width/X
                for (int d = 0; d < bandSize; d++)
                {
                    float freqX = MathF.Pow(theta, -2.0f * d / (bandSize * 2));
                    float cosX = MathF.Cos(px * freqX);
                    float sinX = MathF.Sin(px * freqX);

                    int i0 = headOff + 4 * bandSize + d;
                    int i1 = headOff + 5 * bandSize + d;
                    float q0 = q[i0], q1 = q[i1];
                    q[i0] = q0 * cosX - q1 * sinX;
                    q[i1] = q0 * sinX + q1 * cosX;

                    float k0 = k[i0], k1 = k[i1];
                    k[i0] = k0 * cosX - k1 * sinX;
                    k[i1] = k0 * sinX + k1 * cosX;
                }
            }
        });
    }

    /// <summary>
    /// Resolves typed unmanaged pointer to tensor data inside a GgufModel, checking multiple fallback
    /// aliases. Only used for <c>float</c> now (norm/bias tensors, read element-wise rather than
    /// matvec'd, genuinely always F32 in real mmproj files) -- matmul-bound weights (attention/FFN/
    /// proj) go through <see cref="GetTensor"/> + <see cref="MatVecAny"/> instead, which carry the
    /// tensor's real dtype end-to-end (see docs/done/vl-migration-plan-2026-08-20.md).
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The tensor was found under one of <paramref name="candidateNames"/> but its actual GGUF
    /// storage dtype does not match <typeparamref name="T"/> -- e.g. a quantized mmproj (Q8_0,
    /// Q4_K, ...) where this caller expects raw F32. Found 2026-08-20: every vision encoder in this
    /// project originally read weights via this method (then also used for Half) assuming a fixed
    /// dtype, with no verification against the GGUF's declared type. Against a Q8_0-quantized mmproj
    /// (InternVL3-2B, confirmed via `list-tensors`), that produced a real crash: the since-removed
    /// MatVecF16's row-stride pointer arithmetic assumed 2-byte-per-element Half data, but Q8_0
    /// packs ~1.06 bytes/element, so later rows walked past the tensor's true (smaller) allocation
    /// into unmapped memory -- an AccessViolationException deep inside a Parallel.For lambda, with
    /// no indication of which tensor or why. This check turns that into a clear, immediate,
    /// catchable error instead (see RunCommand.cs's catch around UnifiedVisionPipeline.Open). The
    /// real llama.cpp reference for these encoders is
    /// examples/llama.cpp/llama.cpp/tools/mtmd/models -- ggml's own kernels are dtype-generic, which
    /// is why the reference never needed this guard; this port's per-dtype assumption was the gap.
    /// </exception>
    public static T* GetTensorPtr<T>(GgufModel gguf, params string[] candidateNames) where T : unmanaged
    {
        foreach (var name in candidateNames)
        {
            var tensor = gguf.FindTensor(name);
            if (tensor.HasValue)
            {
                DType? expected = typeof(T) == typeof(Half) ? DType.Float16
                    : typeof(T) == typeof(float) ? DType.Float32
                    : null;
                if (expected is { } exp && tensor.Value.DType != exp)
                {
                    throw new NotSupportedException(
                        $"Tensor '{name}' is stored as {tensor.Value.DType}, but this vision encoder " +
                        $"expects {exp} ({typeof(T).Name}). Quantized mmproj weights (Q8_0, Q4_K, ...) " +
                        "are not yet supported by this encoder -- use an F16/F32 mmproj for this model, " +
                        "or see VisionOps.GetTensorPtr's doc comment for what a fix needs.");
                }
                return (T*)gguf.GetTensorDataPtr(tensor.Value);
            }
        }
        return null;
    }
}
