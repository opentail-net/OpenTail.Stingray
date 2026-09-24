using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Bounded weight-repacking cache and direct quantized matmul dispatcher for diffusion models.
/// Tensors in Q4_K with compatible dimensions are repacked once into Q4Kx8 layout and cached up to
/// an explicit byte budget, routing to <see cref="SimdKernels.TryMatMulBatchedQ4Kx8"/>.
/// Uncached/incompatible tensors fall back to <see cref="SimdKernels.MatMulBatched"/> directly on
/// raw mmap bytes without allocating FP32 weight buffers. FP32 weights used with a real token batch
/// (raw FP32 tensors, or any tensor from a loader without raw access such as safetensors, via
/// <see cref="IWeightLoader.ReadF32"/>) are packed once into <see cref="PackedSgemmF32"/> panels
/// under the same byte budget. Anything left over falls back to
/// <see cref="IWeightLoader.ReadF32"/> + <see cref="DiffusionOps.Linear"/>.
/// </summary>
public sealed class QuantizedWeightCache : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly long _budgetBytes;
    private long _usedBytes;
    private readonly Dictionary<string, nint> _repackedQ4Kx8 = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _packedF32 = new(StringComparer.Ordinal);
    private bool _disposed;

    public long BudgetBytes => _budgetBytes;
    public long UsedBytes => _usedBytes;
    public int CachedTensorCount
    {
        get
        {
            lock (_repackedQ4Kx8) return _repackedQ4Kx8.Count + _packedF32.Count;
        }
    }

    public QuantizedWeightCache(IWeightLoader weights, string prefix = "", long? budgetBytes = null)
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _prefix = prefix;
        _budgetBytes = budgetBytes ?? ResolveBudget();
    }

    private static long ResolveBudget()
    {
        string? raw = Environment.GetEnvironmentVariable("STINGRAY_Q4KX8_CACHE_MB");
        if (long.TryParse(raw, out long mb)) return mb > 0 ? mb * 1024 * 1024 : 0;
        long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        return available > 0 ? available / 3 : 0;
    }

    private string Resolve(string name) => _prefix.Length == 0 ? name : _prefix + name;

    public float[] Linear(string weightName, ReadOnlySpan<float> x, ReadOnlySpan<float> bias, int n, int inDim, int outDim, bool allowQ8 = true)
    {
        var result = new float[n * outDim];
        Linear(weightName, x, bias, result.AsSpan(), n, inDim, outDim, allowQ8);
        return result;
    }

    /// <param name="allowQ8">
    /// When true (default), quantized (Q4_K/Q5_K/Q6_K) weight tensors may be matmul'd against the
    /// activation via an int8-quantized-activation fast path (the repacked Q4_K_X8 kernel and/or
    /// <see cref="SimdKernels.MatMulBatched"/>'s own Q8 prefill tier) -- validated safe for LLM
    /// next-token perplexity (the workload this was tuned for), but NOT for diffusion-model
    /// precision: a 2026-09-20 SD3.5 investigation found this int8 activation quantization produces
    /// real, large per-element errors (maxDiff ~1.0 on a real `adaLN_modulation.1` Linear call,
    /// reproduced in isolation with the real checkpoint weight and real timestep-embedding input --
    /// far beyond ordinary float32 summation-order noise) once fed a wide-dynamic-range activation
    /// like a diffusion timestep/modulation vector, as opposed to the narrower-range token
    /// embeddings this optimization was validated against. Pass <c>false</c> for any diffusion-model
    /// Linear call where correctness matters more than the throughput this buys (every diffusion
    /// pipeline's block count is small -- tens, not the thousands of LLM decode steps this
    /// optimization amortizes over) -- this falls through to the plain float32 dequant+dot path.
    /// </param>
    public unsafe void Linear(string weightName, ReadOnlySpan<float> x, ReadOnlySpan<float> bias, Span<float> output, int n, int inDim, int outDim, bool allowQ8 = true)
    {
        string resolved = Resolve(weightName);

        if (_weights.TryGetRaw(resolved, out nint dataPtr, out _, out DType dtype, out int rows, out int cols))
        {
            if (rows == outDim && cols == inDim)
            {
                fixed (float* px = x, po = output)
                {
                    // 1. Try repacked Q4_K_X8 kernel (fastest path) -- also int8-activation-
                    // quantized internally (QuantizeRowToQ8KS), so gated on allowQ8 too.
                    if (allowQ8 && dtype == DType.Q4_K && SimdKernels.CanRepackQ4Kx8(rows, cols))
                    {
                        byte* packed = GetOrCreateRepackedQ4Kx8(resolved, (byte*)dataPtr, rows, cols);
                        if (packed != null && SimdKernels.TryMatMulBatchedQ4Kx8(po, packed, px, n, rows, cols))
                        {
                            ApplyBias(output, bias, n, outDim);
                            return;
                        }
                    }

                    // 2. FP32 weights with a real token batch: panel-packed 6x16 SGEMM
                    // (1.5-2x the dot-product kernel on DiT shapes, see PackedSgemmF32).
                    if (dtype == DType.Float32 && n >= MinBatchForPackedF32 && PackedSgemmF32.IsSupported)
                    {
                        float* packedF32 = GetOrCreatePackedF32(resolved, (float*)dataPtr, rows, cols);
                        if (packedF32 != null)
                        {
                            PackedSgemmF32.Gemm(po, px, packedF32, null, n, rows, cols);
                            ApplyBias(output, bias, n, outDim);
                            return;
                        }
                    }

                    // 3. Try direct raw-quantized MatMulBatched (MicroGemmQ4K, TryMatMulBatchedQ8, or fused MatVec)
                    SimdKernels.MatMulBatched(po, (byte*)dataPtr, px, n, rows, cols, dtype, allowQ8: allowQ8, allowBlas: true);
                    ApplyBias(output, bias, n, outDim);
                    return;
                }
            }
        }

        // 4. Loaders without raw access (SafetensorsLoader.TryGetRaw always declines): pack the
        // ReadF32 result once and keep it. Without this every call re-read (copied) the whole
        // weight tensor and ran the dot-product kernel -- ~95% of Wan's CPU DiT time.
        if (n >= MinBatchForPackedF32 && PackedSgemmF32.IsSupported)
        {
            float* packedF32 = GetOrCreatePackedF32(resolved, outDim, inDim);
            if (packedF32 != null)
            {
                fixed (float* px = x, po = output)
                    PackedSgemmF32.Gemm(po, px, packedF32, null, n, outDim, inDim);
                ApplyBias(output, bias, n, outDim);
                return;
            }
        }

        // 5. Fallback: ReadF32 + DiffusionOps.Linear
        float[] w = _weights.ReadF32(resolved);
        DiffusionOps.Linear(x, w, bias, output, n, inDim, outDim);
    }

    private unsafe float* GetOrCreatePackedF32(string name, int rows, int cols)
    {
        lock (_repackedQ4Kx8)
        {
            if (_packedF32.TryGetValue(name, out var cached))
                return (float*)cached;
            // Over budget: decline before reading, so the caller's fallback is the only ReadF32.
            if (_usedBytes + PackedSgemmF32.PackedFloats(rows, cols) * sizeof(float) > _budgetBytes)
                return null;
        }
        float[] w = _weights.ReadF32(name);
        if (w.Length != (long)rows * cols) return null;
        fixed (float* pw = w)
            return GetOrCreatePackedF32(name, pw, rows, cols);
    }

    private static void ApplyBias(Span<float> output, ReadOnlySpan<float> bias, int n, int outDim)
    {
        if (bias.IsEmpty) return;
        for (int t = 0; t < n; t++)
        {
            var row = output.Slice(t * outDim, outDim);
            TensorPrimitives.Add(row, bias, row);
        }
    }

    /// <summary>Below this many tokens the dot-product kernel's cost is dominated by the weight
    /// stream either way, and packing buys nothing.</summary>
    private const int MinBatchForPackedF32 = 16;

    private unsafe float* GetOrCreatePackedF32(string name, float* src, int rows, int cols)
    {
        lock (_repackedQ4Kx8) // one lock for both caches: they share _usedBytes
        {
            if (_packedF32.TryGetValue(name, out var cached))
                return (float*)cached;

            long bytes = PackedSgemmF32.PackedFloats(rows, cols) * sizeof(float);
            if (_usedBytes + bytes > _budgetBytes)
                return null;

            float* buf = PackedSgemmF32.PackWeights(src, rows, cols);
            _packedF32[name] = (nint)buf;
            _usedBytes += bytes;
            return buf;
        }
    }

    private unsafe byte* GetOrCreateRepackedQ4Kx8(string name, byte* src, int rows, int cols)
    {
        lock (_repackedQ4Kx8)
        {
            if (_repackedQ4Kx8.TryGetValue(name, out var cached))
                return (byte*)cached;

            long bytes = SimdKernels.Q4Kx8PackedBytes(rows, cols);
            if (bytes <= 0 || _usedBytes + bytes > _budgetBytes)
                return null;

            var buf = (byte*)NativeMemory.Alloc((nuint)bytes);
            try
            {
                SimdKernels.RepackQ4KMatrix(src, buf, rows, cols);
            }
            catch
            {
                NativeMemory.Free(buf);
                throw;
            }

            _repackedQ4Kx8[name] = (nint)buf;
            _usedBytes += bytes;
            return buf;
        }
    }

    public unsafe void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_repackedQ4Kx8)
        {
            foreach (var ptr in _repackedQ4Kx8.Values)
            {
                if (ptr != 0) NativeMemory.Free((void*)ptr);
            }
            _repackedQ4Kx8.Clear();
            foreach (var ptr in _packedF32.Values)
            {
                if (ptr != 0) NativeMemory.AlignedFree((void*)ptr);
            }
            _packedF32.Clear();
            _usedBytes = 0;
        }
        GC.SuppressFinalize(this);
    }

    ~QuantizedWeightCache() => Dispose();
}
