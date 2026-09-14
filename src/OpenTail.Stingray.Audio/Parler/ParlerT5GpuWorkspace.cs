using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for Parler's T5 text encoder. Structurally
/// identical to `OpenTail.Stingray.Diffusion.T5GpuWorkspace` -- duplicated for the same circular-
/// project-reference reason documented on <see cref="ParlerT5GpuWeights"/>.
/// </summary>
public sealed class ParlerT5GpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor X { get; }
    public CoreTensor XNorm { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor Gate { get; }
    public CoreTensor Val { get; }
    public CoreTensor FfOut { get; }
    public CoreTensor RelPosBias { get; }

    public ParlerT5GpuWorkspace(IComputeBackend backend, int seqLen, ReadOnlySpan<float> relPosBias, int dim, int heads, int ffDim)
    {
        _backend = backend;

        X = backend.Allocate(TensorShape.D2(seqLen, dim));
        XNorm = backend.Allocate(TensorShape.D2(seqLen, dim));
        Q = backend.Allocate(TensorShape.D2(seqLen, dim));
        K = backend.Allocate(TensorShape.D2(seqLen, dim));
        V = backend.Allocate(TensorShape.D2(seqLen, dim));
        AttnOut = backend.Allocate(TensorShape.D2(seqLen, dim));
        Gate = backend.Allocate(TensorShape.D2(seqLen, ffDim));
        Val = backend.Allocate(TensorShape.D2(seqLen, ffDim));
        FfOut = backend.Allocate(TensorShape.D2(seqLen, dim));

        RelPosBias = backend.Upload(relPosBias, TensorShape.D3(heads, seqLen, seqLen), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(X);
        _backend.Free(XNorm);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(Gate);
        _backend.Free(Val);
        _backend.Free(FfOut);
        _backend.Free(RelPosBias);
    }
}
