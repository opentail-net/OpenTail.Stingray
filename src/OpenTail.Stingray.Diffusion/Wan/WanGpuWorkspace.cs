using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for Wan 2.1 DiT inference.
/// Holds all activation tensors and precomputed text cross-attention K/V buffers in device-local VRAM
/// across all 30 layers and 20 Euler denoising steps.
/// </summary>
public sealed class WanGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor X { get; }
    public CoreTensor Normed1 { get; }
    public CoreTensor Qkv { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor NormedCross { get; }
    public CoreTensor CrossQ { get; }
    public CoreTensor CrossAttnOut { get; }
    public CoreTensor Normed2 { get; }
    public CoreTensor Ffn1 { get; }
    public CoreTensor FfnOut { get; }
    public CoreTensor OutPacked { get; }
    public CoreTensor Ones { get; }
    public CoreTensor Zeros { get; }
    public CoreTensor Mod { get; }
    public CoreTensor RopeCos { get; }
    public CoreTensor RopeSin { get; }

    public (CoreTensor K, CoreTensor V)[] CrossKvCache { get; }

    public WanGpuWorkspace(IComputeBackend backend, int numTokens, int dim, int ffnDim, int numLayers, int numTxtTokens,
        ReadOnlySpan<float> ropeCos = default, ReadOnlySpan<float> ropeSin = default, int headDim = 128)
    {
        _backend = backend;

        X = backend.Allocate(TensorShape.D2(numTokens, dim));
        Normed1 = backend.Allocate(TensorShape.D2(numTokens, dim));
        Qkv = backend.Allocate(TensorShape.D2(numTokens, dim * 3));
        Q = backend.Allocate(TensorShape.D2(numTokens, dim));
        K = backend.Allocate(TensorShape.D2(numTokens, dim));
        V = backend.Allocate(TensorShape.D2(numTokens, dim));
        AttnOut = backend.Allocate(TensorShape.D2(numTokens, dim));
        NormedCross = backend.Allocate(TensorShape.D2(numTokens, dim));
        CrossQ = backend.Allocate(TensorShape.D2(numTokens, dim));
        CrossAttnOut = backend.Allocate(TensorShape.D2(numTokens, dim));
        Normed2 = backend.Allocate(TensorShape.D2(numTokens, dim));
        Ffn1 = backend.Allocate(TensorShape.D2(numTokens, ffnDim));
        FfnOut = backend.Allocate(TensorShape.D2(numTokens, dim));
        OutPacked = backend.Allocate(TensorShape.D2(numTokens, WanModel.InChannels));
        Mod = backend.AllocatePinned(TensorShape.D1(dim * 6));

        if (!ropeCos.IsEmpty)
        {
            RopeCos = backend.Upload(ropeCos, TensorShape.D2(numTokens, headDim / 2), exact: true);
            RopeSin = backend.Upload(ropeSin, TensorShape.D2(numTokens, headDim / 2), exact: true);
        }
        else
        {
            RopeCos = backend.Allocate(TensorShape.D2(numTokens, headDim / 2));
            RopeSin = backend.Allocate(TensorShape.D2(numTokens, headDim / 2));
        }

        var onesData = new float[dim];
        Array.Fill(onesData, 1.0f);
        Ones = backend.Upload(onesData, TensorShape.D1(dim), exact: true);

        var zerosData = new float[dim];
        Zeros = backend.Upload(zerosData, TensorShape.D1(dim), exact: true);

        CrossKvCache = new (CoreTensor K, CoreTensor V)[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            var k = backend.Allocate(TensorShape.D2(numTxtTokens, dim));
            var v = backend.Allocate(TensorShape.D2(numTxtTokens, dim));
            CrossKvCache[i] = (k, v);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(X);
        _backend.Free(Normed1);
        _backend.Free(Qkv);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(NormedCross);
        _backend.Free(CrossQ);
        _backend.Free(CrossAttnOut);
        _backend.Free(Normed2);
        _backend.Free(Ffn1);
        _backend.Free(FfnOut);
        _backend.Free(OutPacked);
        _backend.Free(Ones);
        _backend.Free(Zeros);
        _backend.Free(Mod);
        _backend.Free(RopeCos);
        _backend.Free(RopeSin);

        for (int i = 0; i < CrossKvCache.Length; i++)
        {
            if (CrossKvCache[i].K is not null) _backend.Free(CrossKvCache[i].K);
            if (CrossKvCache[i].V is not null) _backend.Free(CrossKvCache[i].V);
        }
    }
}
