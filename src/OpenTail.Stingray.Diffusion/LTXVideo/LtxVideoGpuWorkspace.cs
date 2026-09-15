using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// Preallocated GPU VRAM scratch buffers for LTX-Video-2B DiT execution.
/// Reused across all 28 transformer blocks and all denoising steps without reallocation.
/// </summary>
public sealed class LtxVideoGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int NumTokens { get; }
    public int NumTxt { get; }
    public int Dim { get; }
    public int FfnDim { get; }

    public CoreTensor X { get; }
    public CoreTensor Normed { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }

    public CoreTensor CrossQ { get; }
    public CoreTensor CrossK { get; }
    public CoreTensor CrossV { get; }
    public CoreTensor CrossAttnOut { get; }

    public CoreTensor Ffn1 { get; }
    public CoreTensor FfnOut { get; }

    public CoreTensor CaptionProj { get; }
    public CoreTensor CapIntermediate { get; }

    public CoreTensor EmbeddedTimestep { get; }
    public CoreTensor TimestepProj { get; }
    public CoreTensor BlockMod { get; }
    public CoreTensor FinalMod { get; }

    public CoreTensor ProjOut { get; }

    public LtxVideoGpuWorkspace(IComputeBackend backend, int numTokens, int numTxt, int dim, int inChannels)
    {
        _backend = backend;
        NumTokens = numTokens;
        NumTxt = numTxt;
        Dim = dim;
        FfnDim = dim * 4;

        X = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);
        Normed = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);
        Q = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);
        K = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);
        V = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);
        AttnOut = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);

        CrossQ = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);
        CrossK = backend.Allocate(TensorShape.D1(numTxt * dim), DType.Float32);
        CrossV = backend.Allocate(TensorShape.D1(numTxt * dim), DType.Float32);
        CrossAttnOut = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);

        Ffn1 = backend.Allocate(TensorShape.D1(numTokens * FfnDim), DType.Float32);
        FfnOut = backend.Allocate(TensorShape.D1(numTokens * dim), DType.Float32);

        CaptionProj = backend.Allocate(TensorShape.D1(numTxt * dim), DType.Float32);
        CapIntermediate = backend.Allocate(TensorShape.D1(numTxt * dim), DType.Float32);

        EmbeddedTimestep = backend.Allocate(TensorShape.D1(dim), DType.Float32);
        TimestepProj = backend.Allocate(TensorShape.D1(dim * 6), DType.Float32);
        BlockMod = backend.Allocate(TensorShape.D1(dim * 6), DType.Float32);
        FinalMod = backend.Allocate(TensorShape.D1(dim * 2), DType.Float32);

        ProjOut = backend.Allocate(TensorShape.D1(numTokens * inChannels), DType.Float32);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(X);
        _backend.Free(Normed);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);

        _backend.Free(CrossQ);
        _backend.Free(CrossK);
        _backend.Free(CrossV);
        _backend.Free(CrossAttnOut);

        _backend.Free(Ffn1);
        _backend.Free(FfnOut);

        _backend.Free(CaptionProj);
        _backend.Free(CapIntermediate);

        _backend.Free(EmbeddedTimestep);
        _backend.Free(TimestepProj);
        _backend.Free(BlockMod);
        _backend.Free(FinalMod);

        _backend.Free(ProjOut);
    }
}
