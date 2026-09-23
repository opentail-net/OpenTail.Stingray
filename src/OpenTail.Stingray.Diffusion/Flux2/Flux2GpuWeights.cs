using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// GPU-resident VRAM representation of FLUX.2 DiT weights (docs/091's scoped GPU-residency plan).
/// Uploads all linear projections, modulations, and normalization vectors to device VRAM once,
/// enabling zero CPU-GPU round-trip execution during all denoising steps.
///
/// Adapted from <see cref="FluxGpuWeights"/> (FLUX.1's own working port, same double/single-stream
/// block lineage) with FLUX.2's real, confirmed differences (docs/087):
/// <list type="bullet">
/// <item>Every linear projection is bias-free -- no `.bias` tensor exists anywhere in the real
/// checkpoint (confirmed via direct safetensors/GGUF header inspection).</item>
/// <item>Modulation is SHARED across all blocks of a given type (double-img, double-txt, single),
/// NOT per-block AdaLN like FLUX.1 -- only 3 modulation weight matrices total, uploaded once, not
/// once per block.</item>
/// <item>The MLP is SiLU-gated (`SiLU(gate) * value` on a `2*mlpHidden`-wide up-projection split
/// in half), not FLUX.1's GEGLU -- `Mlp0`/`Mlp2` weight shapes differ accordingly
/// (`mlp0: [2*mlpHidden, d]`, vs FLUX.1's `[4*d, d]`).</item>
/// <item>Single-block `linear1` fuses QKV + the 2x-wide gated-MLP up-projection:
/// `[3*d + 2*mlpHidden, d]`, and `linear2` consumes `[d + mlpHidden]` (attn output concatenated
/// with the already-gated MLP activation), not FLUX.1's `[d + 4*d]`/`[d, d+4*d]` shapes.</item>
/// </list>
///
/// <para><b>Scope, 2026-09-19 (docs/088 Pass 2 §2d)</b>: by default this only uploads the 8
/// double-stream blocks (≈15.7GB) -- NOT the 48 single-stream blocks (≈47.1GB), which would push
/// total GPU-resident memory to ≈63GB, this machine's entire RAM budget (see the constructor's own
/// doc comment for the real numbers). The 48 single blocks stay on the existing, already-fast CPU
/// `QuantizedWeightCache` path. This is a deliberate, real, measured scoping decision, not an
/// oversight -- do not "complete" this by flipping `includeSingleBlocks` to `true` without first
/// re-deriving whether the target machine actually has the headroom.</para>
/// </summary>
public sealed class Flux2GpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private readonly Func<string, float[]> _getWeight;
    private readonly Flux2Params _p;
    private bool _disposed;

    public CoreTensor ImgInWeight { get; }
    public CoreTensor TxtInWeight { get; }

    public CoreTensor TimeInLayer0Weight { get; }
    public CoreTensor TimeInLayer2Weight { get; }
    public CoreTensor? GuidanceInLayer0Weight { get; }
    public CoreTensor? GuidanceInLayer2Weight { get; }

    /// <summary>Shared modulation weights -- ONE set total, reused by every double/single block of
    /// that type (real finding, docs/087 -- NOT per-block AdaLN).</summary>
    public CoreTensor DoubleModImgWeight { get; }
    public CoreTensor DoubleModTxtWeight { get; }
    public CoreTensor SingleModWeight { get; }

    public DoubleBlockGpuWeights[] DoubleBlocks { get; }
    public SingleBlockGpuWeights[] SingleBlocks { get; }

    public CoreTensor FinalModWeight { get; }
    public CoreTensor FinalLinearWeight { get; }

    public sealed class DoubleBlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor ImgAttnQkv { get; }
        public CoreTensor TxtAttnQkv { get; }

        public CoreTensor ImgQkNormQScale { get; }
        public CoreTensor ImgQkNormKScale { get; }
        public CoreTensor TxtQkNormQScale { get; }
        public CoreTensor TxtQkNormKScale { get; }

        public CoreTensor ImgAttnProjWeight { get; }
        public CoreTensor TxtAttnProjWeight { get; }

        /// <summary>Gated-FFN up-projection: `[2*mlpHidden, d]` (SiLU-gate half + value half, NOT
        /// FLUX.1's plain GEGLU `[4*d, d]`).</summary>
        public CoreTensor ImgMlp0Weight { get; }
        public CoreTensor ImgMlp2Weight { get; }
        public CoreTensor TxtMlp0Weight { get; }
        public CoreTensor TxtMlp2Weight { get; }

        public DoubleBlockGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int idx, int d, int mlpHidden)
        {
            _backend = backend;
            string p = $"double_blocks.{idx}.";

            ImgAttnQkv = UploadWeight(backend, getWeight($"{p}img_attn.qkv.weight"), TensorShape.D2(d * 3, d));
            TxtAttnQkv = UploadWeight(backend, getWeight($"{p}txt_attn.qkv.weight"), TensorShape.D2(d * 3, d));

            ImgQkNormQScale = backend.Upload(getWeight($"{p}img_attn.norm.query_norm.scale"), TensorShape.D1(128), exact: true);
            ImgQkNormKScale = backend.Upload(getWeight($"{p}img_attn.norm.key_norm.scale"), TensorShape.D1(128), exact: true);
            TxtQkNormQScale = backend.Upload(getWeight($"{p}txt_attn.norm.query_norm.scale"), TensorShape.D1(128), exact: true);
            TxtQkNormKScale = backend.Upload(getWeight($"{p}txt_attn.norm.key_norm.scale"), TensorShape.D1(128), exact: true);

            ImgAttnProjWeight = UploadWeight(backend, getWeight($"{p}img_attn.proj.weight"), TensorShape.D2(d, d));
            TxtAttnProjWeight = UploadWeight(backend, getWeight($"{p}txt_attn.proj.weight"), TensorShape.D2(d, d));

            ImgMlp0Weight = UploadWeight(backend, getWeight($"{p}img_mlp.0.weight"), TensorShape.D2(2 * mlpHidden, d));
            ImgMlp2Weight = UploadWeight(backend, getWeight($"{p}img_mlp.2.weight"), TensorShape.D2(d, mlpHidden));
            TxtMlp0Weight = UploadWeight(backend, getWeight($"{p}txt_mlp.0.weight"), TensorShape.D2(2 * mlpHidden, d));
            TxtMlp2Weight = UploadWeight(backend, getWeight($"{p}txt_mlp.2.weight"), TensorShape.D2(d, mlpHidden));
        }

        public void Dispose()
        {
            _backend.Free(ImgAttnQkv);
            _backend.Free(TxtAttnQkv);
            _backend.Free(ImgQkNormQScale);
            _backend.Free(ImgQkNormKScale);
            _backend.Free(TxtQkNormQScale);
            _backend.Free(TxtQkNormKScale);
            _backend.Free(ImgAttnProjWeight);
            _backend.Free(TxtAttnProjWeight);
            _backend.Free(ImgMlp0Weight);
            _backend.Free(ImgMlp2Weight);
            _backend.Free(TxtMlp0Weight);
            _backend.Free(TxtMlp2Weight);
        }
    }

    public sealed class SingleBlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        /// <summary>Fused QKV + gated-MLP-up projection: `[3*d + 2*mlpHidden, d]` (FLUX.2's real
        /// shape -- NOT FLUX.1's `[7*d, d]` GEGLU-sized fusion).</summary>
        public CoreTensor Linear1Weight { get; }
        public CoreTensor QkNormQScale { get; }
        public CoreTensor QkNormKScale { get; }

        /// <summary>Consumes `[attn (d) ++ gated-mlp-activation (mlpHidden)]`: `[d, d + mlpHidden]`
        /// (NOT FLUX.1's `[d, d + 4*d]`).</summary>
        public CoreTensor Linear2Weight { get; }

        public SingleBlockGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int idx, int d, int mlpHidden)
        {
            _backend = backend;
            string p = $"single_blocks.{idx}.";

            Linear1Weight = UploadWeight(backend, getWeight($"{p}linear1.weight"), TensorShape.D2(3 * d + 2 * mlpHidden, d));
            QkNormQScale = backend.Upload(getWeight($"{p}norm.query_norm.scale"), TensorShape.D1(128), exact: true);
            QkNormKScale = backend.Upload(getWeight($"{p}norm.key_norm.scale"), TensorShape.D1(128), exact: true);
            Linear2Weight = UploadWeight(backend, getWeight($"{p}linear2.weight"), TensorShape.D2(d, d + mlpHidden));
        }

        public void Dispose()
        {
            _backend.Free(Linear1Weight);
            _backend.Free(QkNormQScale);
            _backend.Free(QkNormKScale);
            _backend.Free(Linear2Weight);
        }
    }

    /// <summary>
    /// <c>includeSingleBlocks</c> defaults to <c>false</c> -- REAL, MEASURED FINDING (2026-09-19,
    /// docs/088 Pass 2 §2d, docs/091): FLUX.2 has 48 single-stream blocks at
    /// <c>HiddenSize=6144</c>/<c>MlpRatio=3.0</c>, each with a <c>[55296,6144]</c> `linear1` (340M
    /// params) and `[24576,6144]` `linear2` (151M params) -- uploading all 48 at FP16 needs
    /// ≈47.1GB, plus ≈15.7GB for the 8 double blocks, ≈63GB total. This machine has 63GB of TOTAL
    /// system RAM (an integrated Radeon iGPU, not a discrete GPU with its own VRAM -- see
    /// CLAUDE.md rule 13), leaving nothing for the OS, the Mistral-24B text encoder, or host
    /// buffers. Uploading all single blocks by default WILL exhaust this machine's memory.
    /// Pass <c>includeSingleBlocks: true</c> only on a machine with real headroom to spare
    /// (discrete GPU with its own VRAM, or a host with substantially more than 63GB RAM), and even
    /// then, measure before assuming it's safe -- do not flip this default without re-deriving the
    /// real numbers for whatever checkpoint/hardware is in front of you.
    /// </summary>
    public Flux2GpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Flux2Params p, bool includeSingleBlocks = false)
    {
        _backend = backend;
        _getWeight = getWeight;
        _p = p;
        int d = p.HiddenSize;
        int mlpHidden = (int)(d * p.MlpRatio);

        ImgInWeight = UploadWeight(backend, getWeight("img_in.weight"), TensorShape.D2(d, p.InChannels));
        TxtInWeight = UploadWeight(backend, getWeight("txt_in.weight"), TensorShape.D2(d, p.ContextInDim));

        TimeInLayer0Weight = UploadWeight(backend, getWeight("time_in.in_layer.weight"), TensorShape.D2(d, 256));
        TimeInLayer2Weight = UploadWeight(backend, getWeight("time_in.out_layer.weight"), TensorShape.D2(d, d));

        if (p.GuidanceEmbed)
        {
            GuidanceInLayer0Weight = UploadWeight(backend, getWeight("guidance_in.in_layer.weight"), TensorShape.D2(d, 256));
            GuidanceInLayer2Weight = UploadWeight(backend, getWeight("guidance_in.out_layer.weight"), TensorShape.D2(d, d));
        }

        // Shared modulation -- ONE set, uploaded once (real finding, docs/087).
        DoubleModImgWeight = UploadWeight(backend, getWeight("double_stream_modulation_img.lin.weight"), TensorShape.D2(6 * d, d));
        DoubleModTxtWeight = UploadWeight(backend, getWeight("double_stream_modulation_txt.lin.weight"), TensorShape.D2(6 * d, d));
        SingleModWeight = UploadWeight(backend, getWeight("single_stream_modulation.lin.weight"), TensorShape.D2(3 * d, d));

        DoubleBlocks = new DoubleBlockGpuWeights[p.DepthDoubleBlocks];
        for (int i = 0; i < p.DepthDoubleBlocks; i++)
            DoubleBlocks[i] = new DoubleBlockGpuWeights(backend, getWeight, i, d, mlpHidden);

        // See this constructor's own doc comment -- uploading all 48 single blocks by default
        // would need ≈47.1GB on top of the double blocks' ≈15.7GB, exhausting this machine's 63GB
        // total RAM. Empty array (not null) when skipped, so callers can safely check .Length.
        SingleBlocks = includeSingleBlocks ? new SingleBlockGpuWeights[p.DepthSingleBlocks] : [];
        for (int i = 0; i < SingleBlocks.Length; i++)
            SingleBlocks[i] = new SingleBlockGpuWeights(backend, getWeight, i, d, mlpHidden);

        FinalModWeight = UploadWeight(backend, getWeight("final_layer.adaLN_modulation.1.weight"), TensorShape.D2(2 * d, d));
        FinalLinearWeight = UploadWeight(backend, getWeight("final_layer.linear.weight"), TensorShape.D2(p.OutChannels, d));

        GC.Collect();
    }

    /// <summary>
    /// Loads a single block's GPU weights on demand (for dynamic streaming mode, avoiding 47GB VRAM residency).
    /// If static residency was configured (includeSingleBlocks: true), returns the resident instance.
    /// </summary>
    public SingleBlockGpuWeights LoadSingleBlock(int layerIdx)
    {
        if (layerIdx < SingleBlocks.Length && SingleBlocks[layerIdx] != null)
            return SingleBlocks[layerIdx];
        int d = _p.HiddenSize;
        int mlpHidden = (int)(d * _p.MlpRatio);
        return new SingleBlockGpuWeights(_backend, _getWeight, layerIdx, d, mlpHidden);
    }

    private static CoreTensor UploadWeight(IComputeBackend backend, float[] f32Data, TensorShape shape)
    {
        if (backend.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[f32Data.Length];
            for (int i = 0; i < f32Data.Length; i++) half[i] = (Half)f32Data[i];
            return backend.UploadHalf(half, shape);
        }
        if (backend.BestSgemmPrecision == SgemmPrecision.Bf16)
        {
            var bf16 = new ushort[f32Data.Length];
            for (int i = 0; i < f32Data.Length; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits(f32Data[i]);
                bf16[i] = (ushort)(bits >> 16);
            }
            return backend.UploadBf16(bf16, shape);
        }
        return backend.Upload(f32Data, shape, exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(ImgInWeight);
        _backend.Free(TxtInWeight);
        _backend.Free(TimeInLayer0Weight);
        _backend.Free(TimeInLayer2Weight);
        if (GuidanceInLayer0Weight is { } g0) _backend.Free(g0);
        if (GuidanceInLayer2Weight is { } g2) _backend.Free(g2);

        _backend.Free(DoubleModImgWeight);
        _backend.Free(DoubleModTxtWeight);
        _backend.Free(SingleModWeight);

        for (int i = 0; i < DoubleBlocks.Length; i++)
            DoubleBlocks[i]?.Dispose();
        for (int i = 0; i < SingleBlocks.Length; i++)
            SingleBlocks[i]?.Dispose();

        _backend.Free(FinalModWeight);
        _backend.Free(FinalLinearWeight);
    }
}
