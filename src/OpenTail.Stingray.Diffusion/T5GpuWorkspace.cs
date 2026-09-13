using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for T5-XXL text encoder inference.
/// Holds all activation, projection, attention, and intermediate FFN buffers in device-local VRAM
/// across all 24 encoder layers.
/// </summary>
public sealed class T5GpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int SeqLen { get; }
    public int Dim { get; }
    public int Heads { get; }
    public int HeadDim { get; }
    public int FfDim { get; }

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

    public T5GpuWorkspace(
        IComputeBackend backend,
        int seqLen,
        ReadOnlySpan<float> relPosBias,
        int dim = 4096,
        int heads = 64,
        int headDim = 64,
        int ffDim = 10240)
    {
        _backend = backend;
        SeqLen = seqLen;
        Dim = dim;
        Heads = heads;
        HeadDim = headDim;
        FfDim = ffDim;

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
