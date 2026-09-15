using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.SD3;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for SD 3 / 3.5 MMDiT inference.
/// Holds all activation, modulation, and intermediate attention buffers in device VRAM
/// across all 24 joint transformer blocks and multi-step Euler denoising iterations.
/// </summary>
public sealed class MMDiTGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int NumImgTokens { get; }
    public int NumTextTokens { get; }
    public int TotalTokens { get; }
    public int HiddenSize { get; }
    public int OutPatchDim { get; }

    public CoreTensor X { get; }
    public CoreTensor C { get; }
    public CoreTensor NormedImg { get; }
    public CoreTensor NormedImg2 { get; }
    public CoreTensor NormedTxt { get; }
    public CoreTensor QkvImg { get; }
    public CoreTensor QkvTxt { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor OutImg { get; }
    public CoreTensor OutTxt { get; }
    public CoreTensor MlpBuf { get; }
    public CoreTensor ImgMod { get; }
    public CoreTensor TxtMod { get; }
    public CoreTensor FinalMod { get; }
    public CoreTensor Unpatchified { get; }
    public CoreTensor TVecGpu { get; }

    public MMDiTGpuWorkspace(
        IComputeBackend backend,
        int numImgTokens,
        int numTextTokens,
        int hiddenSize,
        int outPatchDim)
    {
        _backend = backend;
        NumImgTokens = numImgTokens;
        NumTextTokens = numTextTokens;
        TotalTokens = numImgTokens + numTextTokens;
        HiddenSize = hiddenSize;
        OutPatchDim = outPatchDim;

        int maxTokens = Math.Max(TotalTokens, Math.Max(numImgTokens, numTextTokens));
        int mlpHidden = hiddenSize * 4;

        X = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        C = backend.Allocate(TensorShape.D2(numTextTokens, hiddenSize));
        NormedImg = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        NormedImg2 = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        NormedTxt = backend.Allocate(TensorShape.D2(numTextTokens, hiddenSize));

        QkvImg = backend.Allocate(TensorShape.D2(numImgTokens, 3 * hiddenSize));
        QkvTxt = backend.Allocate(TensorShape.D2(numTextTokens, 3 * hiddenSize));
        Q = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        K = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        V = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        AttnOut = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));

        OutImg = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        OutTxt = backend.Allocate(TensorShape.D2(numTextTokens, hiddenSize));
        MlpBuf = backend.Allocate(TensorShape.D2(maxTokens, mlpHidden));

        ImgMod = backend.Allocate(TensorShape.D1(9 * hiddenSize));
        TxtMod = backend.Allocate(TensorShape.D1(6 * hiddenSize));
        FinalMod = backend.Allocate(TensorShape.D1(2 * hiddenSize));
        Unpatchified = backend.Allocate(TensorShape.D2(numImgTokens, outPatchDim));
        TVecGpu = backend.Allocate(TensorShape.D1(hiddenSize));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(X);
        _backend.Free(C);
        _backend.Free(NormedImg);
        _backend.Free(NormedImg2);
        _backend.Free(NormedTxt);
        _backend.Free(QkvImg);
        _backend.Free(QkvTxt);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(OutImg);
        _backend.Free(OutTxt);
        _backend.Free(MlpBuf);
        _backend.Free(ImgMod);
        _backend.Free(TxtMod);
        _backend.Free(FinalMod);
        _backend.Free(Unpatchified);
        _backend.Free(TVecGpu);
    }
}
