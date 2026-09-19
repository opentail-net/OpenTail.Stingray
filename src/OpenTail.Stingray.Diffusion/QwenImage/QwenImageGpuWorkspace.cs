using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for Qwen Image's 60-layer MM-DiT inference
/// (docs/094 Phase 2). Structural sibling of <see cref="SD3.MMDiTGpuWorkspace"/>, adapted for
/// Qwen Image's separate (non-fused) Q/K/V projections -- see <see cref="QwenImageGpuWeights"/>'s
/// doc comment for the full list of real architectural differences from FLUX.2/SD3.5.
/// </summary>
public sealed class QwenImageGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int NumImgTokens { get; }
    public int NumTxtTokens { get; }
    public int TotalTokens { get; }

    public CoreTensor X { get; }
    public CoreTensor C { get; }
    public CoreTensor NormedImg1 { get; }
    public CoreTensor NormedImg2 { get; }
    public CoreTensor NormedTxt1 { get; }
    public CoreTensor NormedTxt2 { get; }

    public CoreTensor ImgQ { get; }
    public CoreTensor ImgK { get; }
    public CoreTensor ImgV { get; }
    public CoreTensor TxtQ { get; }
    public CoreTensor TxtK { get; }
    public CoreTensor TxtV { get; }

    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor ImgAttnOut { get; }
    public CoreTensor TxtAttnOut { get; }

    public CoreTensor ImgMlpBuf { get; }
    public CoreTensor TxtMlpBuf { get; }
    public CoreTensor ImgOut { get; }
    public CoreTensor TxtOut { get; }

    public CoreTensor ImgMod { get; }
    public CoreTensor TxtMod { get; }
    public CoreTensor FinalMod { get; }
    public CoreTensor Unpatchified { get; }
    public CoreTensor TVecSiluGpu { get; }

    /// <summary>Compact per-pair RoPE tables (<c>[TotalTokens, headDim/2]</c>), same storage
    /// convention as <see cref="Flux2.Flux2GpuWorkspace"/>'s <c>RopeCos</c>/<c>RopeSin</c> --
    /// reused directly since both share the same adjacent-pair rotation math (see
    /// <see cref="QwenImageRoPE"/>'s doc comment on the interleaved/GPT-J pairing fix).</summary>
    public CoreTensor RopeCos { get; }
    public CoreTensor RopeSin { get; }

    public QwenImageGpuWorkspace(IComputeBackend backend, int numImgTokens, int numTxtTokens, int hiddenSize, int outChannels, float[] ropeCosFull, float[] ropeSinFull, int headDim)
    {
        _backend = backend;
        NumImgTokens = numImgTokens;
        NumTxtTokens = numTxtTokens;
        TotalTokens = numImgTokens + numTxtTokens;

        int maxTokens = Math.Max(numImgTokens, numTxtTokens);
        int mlpHidden = hiddenSize * 4;

        X = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        C = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));
        NormedImg1 = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        NormedImg2 = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        NormedTxt1 = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));
        NormedTxt2 = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));

        ImgQ = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        ImgK = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        ImgV = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        TxtQ = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));
        TxtK = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));
        TxtV = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));

        Q = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        K = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        V = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        AttnOut = backend.Allocate(TensorShape.D2(TotalTokens, hiddenSize));
        ImgAttnOut = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        TxtAttnOut = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));

        ImgMlpBuf = backend.Allocate(TensorShape.D2(maxTokens, mlpHidden));
        TxtMlpBuf = backend.Allocate(TensorShape.D2(maxTokens, mlpHidden));
        ImgOut = backend.Allocate(TensorShape.D2(numImgTokens, hiddenSize));
        TxtOut = backend.Allocate(TensorShape.D2(numTxtTokens, hiddenSize));

        ImgMod = backend.Allocate(TensorShape.D1(6 * hiddenSize));
        TxtMod = backend.Allocate(TensorShape.D1(6 * hiddenSize));
        FinalMod = backend.Allocate(TensorShape.D1(2 * hiddenSize));
        Unpatchified = backend.Allocate(TensorShape.D2(numImgTokens, outChannels));
        TVecSiluGpu = backend.Allocate(TensorShape.D1(hiddenSize));

        int nPairs = headDim / 2;
        var cosCompact = new float[TotalTokens * nPairs];
        var sinCompact = new float[TotalTokens * nPairs];
        for (int t = 0; t < TotalTokens; t++)
        for (int i = 0; i < nPairs; i++)
        {
            cosCompact[t * nPairs + i] = ropeCosFull[t * headDim + i * 2];
            sinCompact[t * nPairs + i] = ropeSinFull[t * headDim + i * 2];
        }
        RopeCos = backend.Upload(cosCompact, TensorShape.D2(TotalTokens, nPairs), exact: true);
        RopeSin = backend.Upload(sinCompact, TensorShape.D2(TotalTokens, nPairs), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(X);
        _backend.Free(C);
        _backend.Free(NormedImg1);
        _backend.Free(NormedImg2);
        _backend.Free(NormedTxt1);
        _backend.Free(NormedTxt2);
        _backend.Free(ImgQ);
        _backend.Free(ImgK);
        _backend.Free(ImgV);
        _backend.Free(TxtQ);
        _backend.Free(TxtK);
        _backend.Free(TxtV);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(ImgAttnOut);
        _backend.Free(TxtAttnOut);
        _backend.Free(ImgMlpBuf);
        _backend.Free(TxtMlpBuf);
        _backend.Free(ImgOut);
        _backend.Free(TxtOut);
        _backend.Free(ImgMod);
        _backend.Free(TxtMod);
        _backend.Free(FinalMod);
        _backend.Free(Unpatchified);
        _backend.Free(TVecSiluGpu);
        _backend.Free(RopeCos);
        _backend.Free(RopeSin);
    }
}
