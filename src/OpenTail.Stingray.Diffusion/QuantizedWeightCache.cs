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
/// raw mmap bytes without allocating FP32 weight buffers. Safetensors/unsupported loaders fall back
/// to <see cref="IWeightLoader.ReadF32"/> + <see cref="DiffusionOps.Linear"/>.
/// </summary>
public sealed class QuantizedWeightCache : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly long _budgetBytes;
    private long _usedBytes;
    private readonly Dictionary<string, nint> _repackedQ4Kx8 = new(StringComparer.Ordinal);
    private bool _disposed;

    public long BudgetBytes => _budgetBytes;
    public long UsedBytes => _usedBytes;
    public int CachedTensorCount
    {
        get
        {
            lock (_repackedQ4Kx8) return _repackedQ4Kx8.Count;
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

    public float[] Linear(string weightName, ReadOnlySpan<float> x, ReadOnlySpan<float> bias, int n, int inDim, int outDim)
    {
        var result = new float[n * outDim];
        Linear(weightName, x, bias, result.AsSpan(), n, inDim, outDim);
        return result;
    }

    public unsafe void Linear(string weightName, ReadOnlySpan<float> x, ReadOnlySpan<float> bias, Span<float> output, int n, int inDim, int outDim)
    {
        string resolved = Resolve(weightName);

        if (_weights.TryGetRaw(resolved, out nint dataPtr, out _, out DType dtype, out int rows, out int cols))
        {
            if (rows == outDim && cols == inDim)
            {
                fixed (float* px = x, po = output)
                {
                    // 1. Try repacked Q4_K_X8 kernel (fastest path)
                    if (dtype == DType.Q4_K && SimdKernels.CanRepackQ4Kx8(rows, cols))
                    {
                        byte* packed = GetOrCreateRepackedQ4Kx8(resolved, (byte*)dataPtr, rows, cols);
                        if (packed != null && SimdKernels.TryMatMulBatchedQ4Kx8(po, packed, px, n, rows, cols))
                        {
                            ApplyBias(output, bias, n, outDim);
                            return;
                        }
                    }

                    // 2. Try direct raw-quantized MatMulBatched (MicroGemmQ4K, TryMatMulBatchedQ8, or fused MatVec)
                    SimdKernels.MatMulBatched(po, (byte*)dataPtr, px, n, rows, cols, dtype, allowQ8: true, allowBlas: true);
                    ApplyBias(output, bias, n, outDim);
                    return;
                }
            }
        }

        // 3. Fallback: ReadF32 + DiffusionOps.Linear (safetensors or non-2D tensors)
        float[] w = _weights.ReadF32(resolved);
        DiffusionOps.Linear(x, w, bias, output, n, inDim, outDim);
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
            _usedBytes = 0;
        }
        GC.SuppressFinalize(this);
    }

    ~QuantizedWeightCache() => Dispose();
}
