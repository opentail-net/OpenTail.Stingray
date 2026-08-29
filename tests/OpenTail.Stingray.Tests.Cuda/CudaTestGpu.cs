
namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// Process-wide cached CUDA availability probe. <see cref="CudaBackend.IsAvailable"/> does a real
/// cuBLAS handle create/destroy round-trip against the driver, which is what made every CUDA test
/// class pay its own probe cost independently (~74 classes each declared their own copy of this
/// check). Caching it once here means only the first caller pays that cost; every other CUDA test
/// in the run — and every one that skips because no device is present — reads a bool.
/// </summary>
internal static class CudaTestGpu
{
    /// <summary>Result of the single CUDA probe for this test run, computed on first access.</summary>
    public static bool IsAvailable { get; } = CudaBackend.IsAvailable();

    public static CudaBackend? TryCreate()
    {
        if (!IsAvailable) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }
}
