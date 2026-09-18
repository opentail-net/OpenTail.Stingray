using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for Stable Audio 3 Medium DiT.
/// Sized for 24 layers, 1536 embed dim, 6144 FFN inner, and differential attention
/// (5x QKV for self-attn, 2x Q / 3x KV for cross-attn).
/// Reused across every ODE Euler denoising step.
/// </summary>
public sealed class StableAudioMediumGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int MaxSeqLen { get; }
    public int MaxTotalSeq { get; }
    public int Dim { get; }
    public int IoChannels { get; }
    public int FfInner { get; }
    public int NCondMax { get; }

    // Latent buffers
    public CoreTensor Latent { get; }
    public CoreTensor LatentPre { get; }
    public CoreTensor XIn { get; }
    public CoreTensor XFull { get; }
    public CoreTensor Stripped { get; }
    public CoreTensor OutLow { get; }
    public CoreTensor Velocity { get; }

    // Transformer layer scratch buffers (reused across all layers)
    public CoreTensor Normed { get; }
    public CoreTensor Qkv { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor QDiff { get; }
    public CoreTensor KDiff { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor AttnOutDiff { get; }
    public CoreTensor AttnProj { get; }

    public CoreTensor CrossNormed { get; }
    public CoreTensor CrossQBoth { get; }
    public CoreTensor CrossQ { get; }
    public CoreTensor CrossQDiff { get; }
    public CoreTensor CrossAttnOut { get; }
    public CoreTensor CrossAttnOutDiff { get; }
    public CoreTensor CrossAttnProj { get; }

    public CoreTensor FfNormed { get; }
    public CoreTensor FfGate { get; }
    public CoreTensor FfUp { get; }
    public CoreTensor FfOut { get; }

    // Modulation vectors and layer modulations
    public CoreTensor[] LayerMods { get; }

    // Precomputed cross-attention differential KV cache per layer (invariant across steps)
    public CoreTensor[] CondLayerK { get; }
    public CoreTensor[] CondLayerKDiff { get; }
    public CoreTensor[] CondLayerV { get; }
    public CoreTensor[] UncondLayerK { get; }
    public CoreTensor[] UncondLayerKDiff { get; }
    public CoreTensor[] UncondLayerV { get; }

    // CFG buffers
    public CoreTensor CondOutput { get; }
    public CoreTensor UncondOutput { get; }

    public StableAudioMediumGpuWorkspace(
        IComputeBackend backend,
        int maxSeqLen = 256,
        int memoryTokens = 64,
        int dim = 1536,
        int ioChannels = 256,
        int ffInner = 6144,
        int nCondMax = 257,
        int numLayers = 24)
    {
        _backend = backend;
        MaxSeqLen = maxSeqLen;
        MaxTotalSeq = memoryTokens + maxSeqLen;
        Dim = dim;
        IoChannels = ioChannels;
        FfInner = ffInner;
        NCondMax = nCondMax;

        Latent = backend.Allocate(TensorShape.D2(maxSeqLen, ioChannels));
        LatentPre = backend.Allocate(TensorShape.D2(maxSeqLen, ioChannels));
        XIn = backend.Allocate(TensorShape.D2(maxSeqLen, dim));
        XFull = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        Stripped = backend.Allocate(TensorShape.D2(maxSeqLen, dim));
        OutLow = backend.Allocate(TensorShape.D2(maxSeqLen, ioChannels));
        Velocity = backend.Allocate(TensorShape.D2(maxSeqLen, ioChannels));

        Normed = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        Qkv = backend.Allocate(TensorShape.D2(MaxTotalSeq, 5 * dim));
        Q = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        K = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        V = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        QDiff = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        KDiff = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        AttnOut = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        AttnOutDiff = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        AttnProj = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));

        CrossNormed = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossQBoth = backend.Allocate(TensorShape.D2(MaxTotalSeq, 2 * dim));
        CrossQ = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossQDiff = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossAttnOut = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossAttnOutDiff = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossAttnProj = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));

        FfNormed = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        FfGate = backend.Allocate(TensorShape.D2(MaxTotalSeq, ffInner));
        FfUp = backend.Allocate(TensorShape.D2(MaxTotalSeq, ffInner));
        FfOut = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));

        LayerMods = new CoreTensor[numLayers];
        CondLayerK = new CoreTensor[numLayers];
        CondLayerKDiff = new CoreTensor[numLayers];
        CondLayerV = new CoreTensor[numLayers];
        UncondLayerK = new CoreTensor[numLayers];
        UncondLayerKDiff = new CoreTensor[numLayers];
        UncondLayerV = new CoreTensor[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            LayerMods[i] = backend.AllocatePinned(TensorShape.D1(6 * dim));
            CondLayerK[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            CondLayerKDiff[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            CondLayerV[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            UncondLayerK[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            UncondLayerKDiff[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            UncondLayerV[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
        }

        CondOutput = backend.Allocate(TensorShape.D2(maxSeqLen, ioChannels));
        UncondOutput = backend.Allocate(TensorShape.D2(maxSeqLen, ioChannels));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(Latent);
        _backend.Free(LatentPre);
        _backend.Free(XIn);
        _backend.Free(XFull);
        _backend.Free(Stripped);
        _backend.Free(OutLow);
        _backend.Free(Velocity);

        _backend.Free(Normed);
        _backend.Free(Qkv);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(QDiff);
        _backend.Free(KDiff);
        _backend.Free(AttnOut);
        _backend.Free(AttnOutDiff);
        _backend.Free(AttnProj);

        _backend.Free(CrossNormed);
        _backend.Free(CrossQBoth);
        _backend.Free(CrossQ);
        _backend.Free(CrossQDiff);
        _backend.Free(CrossAttnOut);
        _backend.Free(CrossAttnOutDiff);
        _backend.Free(CrossAttnProj);

        _backend.Free(FfNormed);
        _backend.Free(FfGate);
        _backend.Free(FfUp);
        _backend.Free(FfOut);

        for (int i = 0; i < LayerMods.Length; i++)
        {
            _backend.Free(LayerMods[i]);
            _backend.Free(CondLayerK[i]);
            _backend.Free(CondLayerKDiff[i]);
            _backend.Free(CondLayerV[i]);
            _backend.Free(UncondLayerK[i]);
            _backend.Free(UncondLayerKDiff[i]);
            _backend.Free(UncondLayerV[i]);
        }

        _backend.Free(CondOutput);
        _backend.Free(UncondOutput);
    }
}
