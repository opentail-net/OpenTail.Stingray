using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.AceStep.Transformer;

/// <summary>
/// Preallocated GPU device workspace for ACE-Step Turbo DiT inference.
/// Holds all activation, projection, attention, SwiGLU, and modulation buffers resident in VRAM.
/// Sized once for the generation duration and reused across all 8 Euler ODE steps without reallocations.
/// </summary>
public sealed class AceStepDiTGpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int MaxFrames { get; }
    public int MaxTokens { get; }
    public int CondLen { get; }

    public CoreTensor Latent { get; }
    public CoreTensor ContextLatents { get; }
    public CoreTensor ProjInWindow { get; }
    public CoreTensor XFull { get; }

    public CoreTensor Normed1 { get; }
    public CoreTensor Q { get; }
    public CoreTensor K8 { get; }
    public CoreTensor V8 { get; }
    public CoreTensor K16 { get; }
    public CoreTensor V16 { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor AttnProj { get; }
    public CoreTensor AfterSelf { get; }

    public CoreTensor Normed2 { get; }
    public CoreTensor CrossQ { get; }
    public CoreTensor CrossAttnOut { get; }
    public CoreTensor CrossAttnProj { get; }
    public CoreTensor AfterCross { get; }

    public CoreTensor Normed3 { get; }
    public CoreTensor MlpGate { get; }
    public CoreTensor MlpUp { get; }
    public CoreTensor MlpOut { get; }

    public CoreTensor Velocity { get; }

    public CoreTensor CondFlat { get; }
    public CoreTensor CondK8 { get; }
    public CoreTensor CondV8 { get; }
    public CoreTensor[] CondLayerK { get; }
    public CoreTensor[] CondLayerV { get; }

    public CoreTensor LayerMods { get; }
    public CoreTensor FinalMod { get; }

    public AceStepDiTGpuWorkspace(IComputeBackend backend, int maxFrames, int condLen, int numLayers = 24)
    {
        _backend = backend;
        MaxFrames = maxFrames;
        MaxTokens = (maxFrames + AceStepConfig.PatchSize - 1) / AceStepConfig.PatchSize;
        CondLen = condLen;

        int hidden = AceStepConfig.HiddenSize; // 2048
        int acousticDim = AceStepConfig.AudioAcousticHiddenDim; // 64
        int inCh = AceStepConfig.InChannels; // 192
        int patch = AceStepConfig.PatchSize; // 2
        int inDim = inCh * patch; // 384
        int outDim = acousticDim * patch; // 128
        int kvDim = AceStepConfig.NumKeyValueHeads * AceStepConfig.HeadDim; // 1024
        int ffn = AceStepConfig.IntermediateSize; // 6144

        Latent = backend.Allocate(TensorShape.D2(MaxTokens * patch, acousticDim));
        ContextLatents = backend.Allocate(TensorShape.D2(MaxTokens * patch, 2 * acousticDim));
        ProjInWindow = backend.Allocate(TensorShape.D2(MaxTokens, inDim));
        XFull = backend.Allocate(TensorShape.D2(MaxTokens, hidden));

        Normed1 = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        Q = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        K8 = backend.Allocate(TensorShape.D2(MaxTokens, kvDim));
        V8 = backend.Allocate(TensorShape.D2(MaxTokens, kvDim));
        K16 = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        V16 = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        AttnOut = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        AttnProj = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        AfterSelf = backend.Allocate(TensorShape.D2(MaxTokens, hidden));

        Normed2 = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        CrossQ = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        CrossAttnOut = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        CrossAttnProj = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        AfterCross = backend.Allocate(TensorShape.D2(MaxTokens, hidden));

        Normed3 = backend.Allocate(TensorShape.D2(MaxTokens, hidden));
        MlpGate = backend.Allocate(TensorShape.D2(MaxTokens, ffn));
        MlpUp = backend.Allocate(TensorShape.D2(MaxTokens, ffn));
        MlpOut = backend.Allocate(TensorShape.D2(MaxTokens, hidden));

        Velocity = backend.Allocate(TensorShape.D2(MaxTokens, outDim));

        CondFlat = backend.Allocate(TensorShape.D2(condLen, hidden));
        CondK8 = backend.Allocate(TensorShape.D2(condLen, kvDim));
        CondV8 = backend.Allocate(TensorShape.D2(condLen, kvDim));

        CondLayerK = new CoreTensor[numLayers];
        CondLayerV = new CoreTensor[numLayers];
        for (int l = 0; l < numLayers; l++)
        {
            CondLayerK[l] = backend.Allocate(TensorShape.D2(condLen, hidden));
            CondLayerV[l] = backend.Allocate(TensorShape.D2(condLen, hidden));
        }

        LayerMods = backend.AllocatePinned(TensorShape.D1(numLayers * 6 * hidden));
        FinalMod = backend.AllocatePinned(TensorShape.D1(2 * hidden));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(Latent);
        _backend.Free(ContextLatents);
        _backend.Free(ProjInWindow);
        _backend.Free(XFull);

        _backend.Free(Normed1);
        _backend.Free(Q);
        _backend.Free(K8);
        _backend.Free(V8);
        _backend.Free(K16);
        _backend.Free(V16);
        _backend.Free(AttnOut);
        _backend.Free(AttnProj);
        _backend.Free(AfterSelf);

        _backend.Free(Normed2);
        _backend.Free(CrossQ);
        _backend.Free(CrossAttnOut);
        _backend.Free(CrossAttnProj);
        _backend.Free(AfterCross);

        _backend.Free(Normed3);
        _backend.Free(MlpGate);
        _backend.Free(MlpUp);
        _backend.Free(MlpOut);

        _backend.Free(Velocity);

        _backend.Free(CondFlat);
        _backend.Free(CondK8);
        _backend.Free(CondV8);

        foreach (var t in CondLayerK) _backend.Free(t);
        foreach (var t in CondLayerV) _backend.Free(t);

        _backend.Free(LayerMods);
        _backend.Free(FinalMod);
    }
}
