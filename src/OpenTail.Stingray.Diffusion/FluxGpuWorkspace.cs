using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for FLUX.1 (MM-DiT) inference.
/// Holds all activation, modulation, and intermediate attention buffers in device-local VRAM
/// across all 19 double blocks, 38 single blocks, and multi-step Euler denoising iterations.
/// </summary>
public sealed class FluxGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int NumSeq { get; }
    public int NumImg { get; }
    public int NumTxt { get; }
    public int Dim { get; }

    public CoreTensor ImgHidden { get; }
    public CoreTensor TxtHidden { get; }
    public CoreTensor X { get; }
    public CoreTensor NormedImg { get; }
    public CoreTensor NormedTxt { get; }
    public CoreTensor NormedSeq { get; }
    public CoreTensor QkvTxt { get; }
    public CoreTensor QkvImg { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor Lin1 { get; }
    public CoreTensor Combined { get; }
    public CoreTensor MlpBuf { get; }
    public CoreTensor OutImg { get; }
    public CoreTensor OutTxt { get; }
    public CoreTensor ImgMod { get; }
    public CoreTensor TxtMod { get; }
    public CoreTensor SingleMod { get; }
    public CoreTensor FinalMod { get; }
    public CoreTensor FinalOut { get; }
    public CoreTensor Latent { get; }
    public CoreTensor VecGpu { get; }

    public CoreTensor ImgRopeCos { get; }
    public CoreTensor ImgRopeSin { get; }
    public CoreTensor AllRopeCos { get; }
    public CoreTensor AllRopeSin { get; }

    public FluxGpuWorkspace(
        IComputeBackend backend,
        int nSeq,
        int nImg,
        int nTxt,
        int d,
        ReadOnlySpan<float> imgRopeCos,
        ReadOnlySpan<float> imgRopeSin,
        ReadOnlySpan<float> allRopeCos,
        ReadOnlySpan<float> allRopeSin)
    {
        _backend = backend;
        NumSeq = nSeq;
        NumImg = nImg;
        NumTxt = nTxt;
        Dim = d;

        int maxMlp = Math.Max(nSeq, Math.Max(nImg, nTxt));

        ImgHidden = backend.Allocate(TensorShape.D2(nImg, d));
        TxtHidden = backend.Allocate(TensorShape.D2(nTxt, d));
        X = backend.Allocate(TensorShape.D2(nSeq, d));
        NormedImg = backend.Allocate(TensorShape.D2(nImg, d));
        NormedTxt = backend.Allocate(TensorShape.D2(nTxt, d));
        NormedSeq = backend.Allocate(TensorShape.D2(nSeq, d));
        QkvTxt = backend.Allocate(TensorShape.D2(nTxt, d * 3));
        QkvImg = backend.Allocate(TensorShape.D2(nImg, d * 3));
        Q = backend.Allocate(TensorShape.D2(nSeq, d));
        K = backend.Allocate(TensorShape.D2(nSeq, d));
        V = backend.Allocate(TensorShape.D2(nSeq, d));
        AttnOut = backend.Allocate(TensorShape.D2(nSeq, d));
        Lin1 = backend.Allocate(TensorShape.D2(nSeq, d * 7));
        Combined = backend.Allocate(TensorShape.D2(nSeq, d * 5));
        MlpBuf = backend.Allocate(TensorShape.D2(maxMlp, d * 4));
        OutImg = backend.Allocate(TensorShape.D2(nImg, d));
        OutTxt = backend.Allocate(TensorShape.D2(nTxt, d));
        ImgMod = backend.Allocate(TensorShape.D1(d * 6));
        TxtMod = backend.Allocate(TensorShape.D1(d * 6));
        SingleMod = backend.Allocate(TensorShape.D1(d * 3));
        FinalMod = backend.Allocate(TensorShape.D1(d * 2));
        FinalOut = backend.Allocate(TensorShape.D2(nImg, 64));
        Latent = backend.Allocate(TensorShape.D2(nImg, 64));
        VecGpu = backend.AllocatePinned(TensorShape.D1(d));

        ImgRopeCos = backend.Upload(imgRopeCos, TensorShape.D2(nImg, 64), exact: true);
        ImgRopeSin = backend.Upload(imgRopeSin, TensorShape.D2(nImg, 64), exact: true);
        AllRopeCos = backend.Upload(allRopeCos, TensorShape.D2(nSeq, 64), exact: true);
        AllRopeSin = backend.Upload(allRopeSin, TensorShape.D2(nSeq, 64), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(ImgHidden);
        _backend.Free(TxtHidden);
        _backend.Free(X);
        _backend.Free(NormedImg);
        _backend.Free(NormedTxt);
        _backend.Free(NormedSeq);
        _backend.Free(QkvTxt);
        _backend.Free(QkvImg);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(Lin1);
        _backend.Free(Combined);
        _backend.Free(MlpBuf);
        _backend.Free(OutImg);
        _backend.Free(OutTxt);
        _backend.Free(ImgMod);
        _backend.Free(TxtMod);
        _backend.Free(SingleMod);
        _backend.Free(FinalMod);
        _backend.Free(FinalOut);
        _backend.Free(Latent);
        _backend.Free(VecGpu);
        _backend.Free(ImgRopeCos);
        _backend.Free(ImgRopeSin);
        _backend.Free(AllRopeCos);
        _backend.Free(AllRopeSin);
    }
}
