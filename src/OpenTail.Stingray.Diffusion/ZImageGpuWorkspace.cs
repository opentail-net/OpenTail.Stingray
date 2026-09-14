using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for Z-Image-Turbo's main `layers.N` blocks
/// (docs/075). Sized for a fixed total token count `t` (image + text tokens concatenated) and
/// reused across every ODE denoising step.
/// </summary>
public sealed class ZImageGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor X { get; }
    public CoreTensor Normed { get; }
    public CoreTensor Qkv { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor AttnProj { get; }
    public CoreTensor FfnGate { get; }
    public CoreTensor FfnUp { get; }
    public CoreTensor FfnOut { get; }
    public CoreTensor Zeros { get; }
    public CoreTensor RopeCos { get; }
    public CoreTensor RopeSin { get; }

    public ZImageGpuWorkspace(IComputeBackend backend, int t, int dim, int ffnHidden, ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, int headDim)
    {
        _backend = backend;

        X = backend.Allocate(TensorShape.D2(t, dim));
        Normed = backend.Allocate(TensorShape.D2(t, dim));
        Qkv = backend.Allocate(TensorShape.D2(t, 3 * dim));
        Q = backend.Allocate(TensorShape.D2(t, dim));
        K = backend.Allocate(TensorShape.D2(t, dim));
        V = backend.Allocate(TensorShape.D2(t, dim));
        AttnOut = backend.Allocate(TensorShape.D2(t, dim));
        AttnProj = backend.Allocate(TensorShape.D2(t, dim));
        FfnGate = backend.Allocate(TensorShape.D2(t, ffnHidden));
        FfnUp = backend.Allocate(TensorShape.D2(t, ffnHidden));
        FfnOut = backend.Allocate(TensorShape.D2(t, dim));

        // Z-Image's real AdaLN convention has NO shift term at all (unlike FLUX/Wan's shift+scale)
        // -- reuse the existing AdaLNModulate kernel (which requires both a shift and scale
        // tensor) by passing this zero-filled tensor as "shift", a mathematical no-op
        // (norm*(1+scale)+0 = norm*(1+scale), matching the real formula exactly). See docs/075's
        // own 2026-09-14 addendum for the full reasoning.
        var zerosData = new float[dim];
        Zeros = backend.Upload(zerosData, TensorShape.D1(dim), exact: true);

        int halfHead = headDim / 2;
        RopeCos = backend.Upload(ropeCos, TensorShape.D2(t, halfHead), exact: true);
        RopeSin = backend.Upload(ropeSin, TensorShape.D2(t, halfHead), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _backend.Free(X);
        _backend.Free(Normed);
        _backend.Free(Qkv);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(AttnProj);
        _backend.Free(FfnGate);
        _backend.Free(FfnUp);
        _backend.Free(FfnOut);
        _backend.Free(Zeros);
        _backend.Free(RopeCos);
        _backend.Free(RopeSin);
    }
}
