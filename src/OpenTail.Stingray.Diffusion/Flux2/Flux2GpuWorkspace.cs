using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Preallocated zero-allocation GPU memory workspace for FLUX.2's double-stream-block-only GPU
/// residency (docs/091). Sized for a single (nImg, nTxt) shape and reused across all 8 double
/// blocks and every denoising step -- the caller re-creates it if the token counts change (e.g. a
/// different resolution).
///
/// Adapted from <see cref="FluxGpuWorkspace"/> (FLUX.1's own working port) with FLUX.2's real,
/// confirmed differences:
/// <list type="bullet">
/// <item>No bias buffers anywhere -- FLUX.2's linears are bias-free.</item>
/// <item>Modulation is SHARED across all 8 double blocks (one <see cref="ImgMod"/>/
/// <see cref="TxtMod"/> pair computed ONCE from <c>vec</c> before the block loop starts, not
/// recomputed per-block like FLUX.1's own per-block AdaLN).</item>
/// <item><see cref="MlpUpBuf"/> is <c>2*mlpHidden</c> wide (the SiLU-gated up-projection's full
/// output, before gating) and <see cref="MlpGatedBuf"/> is <c>mlpHidden</c> wide (the gated
/// activation, after <c>SiluGateMul</c>) -- two buffers, unlike FLUX.1's single <c>4*d</c>-wide
/// GELU buffer, because the gate-then-project shape needs an intermediate width change.</item>
/// <item>RoPE cos/sin are a SINGLE table covering the full concatenated [txt, img] sequence
/// (<see cref="RopeCos"/>/<see cref="RopeSin"/>) -- unlike HunyuanVideo, FLUX.2's real reference
/// RoPEs both text AND image tokens (confirmed 2026-09-19 against `model.py`'s `_prepare_qkv`),
/// so there is no separate img-only table.</item>
/// </list>
/// </summary>
public sealed class Flux2GpuWorkspace : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public int NumSeq { get; }
    public int NumImg { get; }
    public int NumTxt { get; }
    public int Dim { get; }
    public int MlpHidden { get; }

    public CoreTensor ImgHidden { get; private set; }
    public CoreTensor TxtHidden { get; private set; }
    public CoreTensor NormedImg { get; }
    public CoreTensor NormedTxt { get; }
    public CoreTensor QkvTxt { get; }
    public CoreTensor QkvImg { get; }
    public CoreTensor Q { get; }
    public CoreTensor K { get; }
    public CoreTensor V { get; }
    public CoreTensor AttnOut { get; }
    public CoreTensor OutImg { get; }
    public CoreTensor OutTxt { get; }

    /// <summary>Unified [nSeq, d] sequence holding [Txt; Img] concatenated for single-stream blocks.</summary>
    public CoreTensor Unified { get; private set; }

    /// <summary>Normalized unified sequence [nSeq, d] after AdaLN modulation in single-stream blocks.</summary>
    public CoreTensor NormedSeq { get; }

    /// <summary>Shared modulation vector [3*d] for single-stream blocks (shift, scale, gate).</summary>
    public CoreTensor SingleMod { get; }

    /// <summary>Output of linear1 in single-stream blocks: [nSeq, 3*d + 2*mlpHidden].</summary>
    public CoreTensor Lin1Out { get; }

    /// <summary>Input to linear2 in single-stream blocks: [nSeq, d + mlpHidden] (AttnOut concatenated with gated MLP).</summary>
    public CoreTensor Lin2In { get; }

    /// <summary>Output of linear2 in single-stream blocks: [nSeq, d].</summary>
    public CoreTensor OutSeq { get; }

    /// <summary>SiLU-gated FFN up-projection output, `[n, 2*mlpHidden]` (n = max(nImg, nTxt)) --
    /// wide enough to hold either stream's up-projection.</summary>
    public CoreTensor MlpUpBuf { get; }

    /// <summary>SiLU-gated FFN activation after gating, `[n, mlpHidden]`.</summary>
    public CoreTensor MlpGatedBuf { get; }

    /// <summary>Shared modulation vectors, computed ONCE from `vec` before the double-block loop
    /// (real finding, docs/087/091 -- NOT per-block like FLUX.1).</summary>
    public CoreTensor ImgMod { get; }
    public CoreTensor TxtMod { get; }

    /// <summary>Compact `[nSeq, headDim/2]` RoPE table covering the full [txt, img] sequence
    /// (see class doc comment -- both streams are rotated in FLUX.2, unlike HunyuanVideo).</summary>
    public CoreTensor RopeCos { get; }
    public CoreTensor RopeSin { get; }

    public Flux2GpuWorkspace(
        IComputeBackend backend,
        int nImg,
        int nTxt,
        int d,
        int mlpHidden,
        int headDim,
        ReadOnlySpan<float> ropeCosCompact,
        ReadOnlySpan<float> ropeSinCompact,
        ReadOnlySpan<float> initialImgHidden = default,
        ReadOnlySpan<float> initialTxtHidden = default)
    {
        _backend = backend;
        int nSeq = nImg + nTxt;
        NumSeq = nSeq;
        NumImg = nImg;
        NumTxt = nTxt;
        Dim = d;
        MlpHidden = mlpHidden;

        int maxStream = Math.Max(nImg, nTxt);
        int nPairs = headDim / 2;

        // ImgHidden/TxtHidden are uploaded directly from the caller's initial data when supplied
        // (there is no generic "upload into an existing tensor" op on IComputeBackend, only
        // Upload which allocates fresh) -- falls back to a zero-initialized buffer of the right
        // shape otherwise.
        ImgHidden = initialImgHidden.IsEmpty
            ? backend.Allocate(TensorShape.D2(nImg, d))
            : backend.Upload(initialImgHidden, TensorShape.D2(nImg, d), exact: true);
        TxtHidden = initialTxtHidden.IsEmpty
            ? backend.Allocate(TensorShape.D2(nTxt, d))
            : backend.Upload(initialTxtHidden, TensorShape.D2(nTxt, d), exact: true);
        NormedImg = backend.Allocate(TensorShape.D2(nImg, d));
        NormedTxt = backend.Allocate(TensorShape.D2(nTxt, d));
        QkvTxt = backend.Allocate(TensorShape.D2(nTxt, d * 3));
        QkvImg = backend.Allocate(TensorShape.D2(nImg, d * 3));
        Q = backend.Allocate(TensorShape.D2(nSeq, d));
        K = backend.Allocate(TensorShape.D2(nSeq, d));
        V = backend.Allocate(TensorShape.D2(nSeq, d));
        AttnOut = backend.Allocate(TensorShape.D2(nSeq, d));
        OutImg = backend.Allocate(TensorShape.D2(nImg, d));
        OutTxt = backend.Allocate(TensorShape.D2(nTxt, d));
        MlpUpBuf = backend.Allocate(TensorShape.D2(maxStream, 2 * mlpHidden));
        MlpGatedBuf = backend.Allocate(TensorShape.D2(maxStream, mlpHidden));

        ImgMod = backend.Allocate(TensorShape.D1(d * 6));
        TxtMod = backend.Allocate(TensorShape.D1(d * 6));

        Unified = backend.Allocate(TensorShape.D2(nSeq, d));
        NormedSeq = backend.Allocate(TensorShape.D2(nSeq, d));
        SingleMod = backend.Allocate(TensorShape.D1(d * 3));
        Lin1Out = backend.Allocate(TensorShape.D2(nSeq, 3 * d + 2 * mlpHidden));
        Lin2In = backend.Allocate(TensorShape.D2(nSeq, d + mlpHidden));
        OutSeq = backend.Allocate(TensorShape.D2(nSeq, d));

        RopeCos = backend.Upload(ropeCosCompact, TensorShape.D2(nSeq, nPairs), exact: true);
        RopeSin = backend.Upload(ropeSinCompact, TensorShape.D2(nSeq, nPairs), exact: true);
    }

    /// <summary>Re-uploads this step's img/txt hidden state, freeing the previous buffers first --
    /// used when reusing an already-shape-matched workspace across denoising steps instead of
    /// rebuilding the whole workspace (which would also needlessly re-upload the RoPE table).</summary>
    public void RefreshHidden(IComputeBackend backend, ReadOnlySpan<float> img, ReadOnlySpan<float> txt)
    {
        backend.Free(ImgHidden);
        backend.Free(TxtHidden);
        ImgHidden = backend.Upload(img, TensorShape.D2(NumImg, Dim), exact: true);
        TxtHidden = backend.Upload(txt, TensorShape.D2(NumTxt, Dim), exact: true);
    }

    /// <summary>Uploads data directly into ws.Unified (used for standalone single-block tests).</summary>
    public void SetUnified(IComputeBackend backend, ReadOnlySpan<float> data)
    {
        backend.Free(Unified);
        Unified = backend.Upload(data, TensorShape.D2(NumSeq, Dim), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(ImgHidden);
        _backend.Free(TxtHidden);
        _backend.Free(NormedImg);
        _backend.Free(NormedTxt);
        _backend.Free(QkvTxt);
        _backend.Free(QkvImg);
        _backend.Free(Q);
        _backend.Free(K);
        _backend.Free(V);
        _backend.Free(AttnOut);
        _backend.Free(OutImg);
        _backend.Free(OutTxt);
        _backend.Free(Unified);
        _backend.Free(NormedSeq);
        _backend.Free(SingleMod);
        _backend.Free(Lin1Out);
        _backend.Free(Lin2In);
        _backend.Free(OutSeq);
        _backend.Free(MlpUpBuf);
        _backend.Free(MlpGatedBuf);
        _backend.Free(ImgMod);
        _backend.Free(TxtMod);
        _backend.Free(RopeCos);
        _backend.Free(RopeSin);
    }
}
