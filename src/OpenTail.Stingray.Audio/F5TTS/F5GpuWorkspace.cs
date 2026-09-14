using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for F5-TTS's 22-block DiT, mirroring
/// <c>WanGpuWorkspace</c>'s pattern. Sized for a fixed frame count <c>t</c> (the mel sequence
/// length for one generation) and reused across every ODE denoising step.
/// </summary>
public sealed class F5GpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor X { get; }
    public CoreTensor Normed { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor AttnProj { get; }
    public CoreTensor FfnMid { get; }
    public CoreTensor FfnOut { get; }
    public CoreTensor Modulation { get; }
    public CoreTensor RopeCos { get; }
    public CoreTensor RopeSin { get; }

    public F5GpuWorkspace(IComputeBackend backend, int t, ReadOnlySpan<float> rotaryCos = default, ReadOnlySpan<float> rotarySin = default)
    {
        _backend = backend;
        int dim = F5TtsWeights.HiddenDim;
        int ffn = F5TtsWeights.FfnDim;
        int halfHead = F5TtsWeights.HeadDim / 2;

        X = backend.Allocate(TensorShape.D2(t, dim));
        Normed = backend.Allocate(TensorShape.D2(t, dim));
        Q = backend.Allocate(TensorShape.D2(t, dim));
        K = backend.Allocate(TensorShape.D2(t, dim));
        V = backend.Allocate(TensorShape.D2(t, dim));

        // Real `pe_attn_head=1` RoPE (only head 0 gets rotated) now runs entirely on GPU via
        // Shaders.PartialHeadRoPE -- uploaded once here, reused across every block/ODE step,
        // eliminating the 22-round-trip-per-forward-pass CPU fallback this workspace previously
        // needed (see docs/080's 2026-09-14 "real bottleneck" finding).
        if (!rotaryCos.IsEmpty)
        {
            RopeCos = backend.Upload(rotaryCos, TensorShape.D2(t, halfHead), exact: true);
            RopeSin = backend.Upload(rotarySin, TensorShape.D2(t, halfHead), exact: true);
        }
        else
        {
            RopeCos = backend.Allocate(TensorShape.D2(t, halfHead));
            RopeSin = backend.Allocate(TensorShape.D2(t, halfHead));
        }
        AttnOut = backend.Allocate(TensorShape.D2(t, dim));
        AttnProj = backend.Allocate(TensorShape.D2(t, dim));
        FfnMid = backend.Allocate(TensorShape.D2(t, ffn));
        FfnOut = backend.Allocate(TensorShape.D2(t, dim));
        // AdaLN modulation is dim*6-wide per block, computed entirely on GPU each block via a
        // real Sgemm (attn_norm.linear(silu(tEmb)) -- unlike Wan's cheap host-side param+add,
        // F5's modulation is a genuine per-block Linear) -- see F5DiTBlock.ForwardGpu.
        Modulation = backend.Allocate(TensorShape.D1(dim * 6));
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
        _backend.Free(AttnProj);
        _backend.Free(FfnMid);
        _backend.Free(FfnOut);
        _backend.Free(Modulation);
        _backend.Free(RopeCos);
        _backend.Free(RopeSin);
    }
}
