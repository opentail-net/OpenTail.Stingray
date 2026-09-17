using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for Stable Audio 3 DiT.
/// Sized for a maximum sequence length (memory tokens + acoustic latent tokens) and
/// reused across every ODE Euler denoising step.
/// </summary>
public sealed class StableAudio3GpuWorkspace : IDisposable
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
    public CoreTensor AttnOut { get; }
    public CoreTensor AttnProj { get; }
    public CoreTensor CrossNormed { get; }
    public CoreTensor CrossQ { get; }
    public CoreTensor CrossAttnOut { get; }
    public CoreTensor CrossAttnProj { get; }
    public CoreTensor FfNormed { get; }
    public CoreTensor FfProj { get; }
    public CoreTensor FfGate { get; }
    public CoreTensor FfUp { get; }
    public CoreTensor FfOut { get; }

    // Conditioning and cross-attention buffers
    public CoreTensor CondEmbed { get; }
    public CoreTensor CrossKv { get; }
    public CoreTensor CrossK { get; }
    public CoreTensor CrossV { get; }

    // Modulation vectors and layer modulations
    public CoreTensor[] LayerMods { get; }

    // Precomputed cross-attention KV cache per layer (invariant across steps)
    public CoreTensor[] CondLayerK { get; }
    public CoreTensor[] CondLayerV { get; }
    public CoreTensor[] UncondLayerK { get; }
    public CoreTensor[] UncondLayerV { get; }

    // CFG buffers
    public CoreTensor CondOutput { get; }
    public CoreTensor UncondOutput { get; }

    public StableAudio3GpuWorkspace(IComputeBackend backend, int maxSeqLen = 256, int memoryTokens = 64, int dim = 1024, int ioChannels = 256, int ffInner = 4096, int nCondMax = 257, int numLayers = 20)
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
        Qkv = backend.Allocate(TensorShape.D2(MaxTotalSeq, 3 * dim));
        Q = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        K = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        V = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        AttnOut = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        AttnProj = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));

        CrossNormed = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossQ = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossAttnOut = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        CrossAttnProj = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));

        FfNormed = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));
        FfProj = backend.Allocate(TensorShape.D2(MaxTotalSeq, 2 * ffInner));
        FfGate = backend.Allocate(TensorShape.D2(MaxTotalSeq, ffInner));
        FfUp = backend.Allocate(TensorShape.D2(MaxTotalSeq, ffInner));
        FfOut = backend.Allocate(TensorShape.D2(MaxTotalSeq, dim));

        CondEmbed = backend.Allocate(TensorShape.D2(nCondMax, dim));
        CrossKv = backend.Allocate(TensorShape.D2(nCondMax, 2 * dim));
        CrossK = backend.Allocate(TensorShape.D2(nCondMax, dim));
        CrossV = backend.Allocate(TensorShape.D2(nCondMax, dim));

        LayerMods = new CoreTensor[numLayers];
        CondLayerK = new CoreTensor[numLayers];
        CondLayerV = new CoreTensor[numLayers];
        UncondLayerK = new CoreTensor[numLayers];
        UncondLayerV = new CoreTensor[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            LayerMods[i] = backend.AllocatePinned(TensorShape.D1(6 * dim));
            CondLayerK[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            CondLayerV[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
            UncondLayerK[i] = backend.Allocate(TensorShape.D2(nCondMax, dim));
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
        _backend.Free(AttnOut);
        _backend.Free(AttnProj);

        _backend.Free(CrossNormed);
        _backend.Free(CrossQ);
        _backend.Free(CrossAttnOut);
        _backend.Free(CrossAttnProj);

        _backend.Free(FfNormed);
        _backend.Free(FfProj);
        _backend.Free(FfGate);
        _backend.Free(FfUp);
        _backend.Free(FfOut);

        _backend.Free(CondEmbed);
        _backend.Free(CrossKv);
        _backend.Free(CrossK);
        _backend.Free(CrossV);

        for (int i = 0; i < LayerMods.Length; i++)
        {
            _backend.Free(LayerMods[i]);
            _backend.Free(CondLayerK[i]);
            _backend.Free(CondLayerV[i]);
            _backend.Free(UncondLayerK[i]);
            _backend.Free(UncondLayerV[i]);
        }

        _backend.Free(CondOutput);
        _backend.Free(UncondOutput);
    }
}
