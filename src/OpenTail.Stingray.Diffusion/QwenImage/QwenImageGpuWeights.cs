using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// GPU-resident VRAM representation of Qwen Image's 60-layer MM-DiT weights (docs/094 Phase 2).
///
/// <para>Architecture confirmed by direct inspection of <see cref="QwenImageModel"/> (real
/// checkpoint tensor names, not guessed): <c>HeadDim=128</c> -- the SAME head dimension as
/// FLUX.1/FLUX.2, so this port reuses the existing <c>MultiHeadAttentionTiled128</c>/<c>SgemmF16</c>
/// Vulkan kernels directly, unlike SD1.5 (which genuinely needed new
/// <c>MultiHeadAttentionTiled40/80/160</c> shader variants for its non-64/128 head dims). The joint
/// attention shape (separate img/txt Q/K/V projections, per-stream QK-RMSNorm, concatenated
/// [txt;img] sequence, 3D-RoPE) mirrors FLUX.2's double-stream block closely enough that
/// <see cref="Flux2.Flux2GpuWeights"/> is the direct structural template for this class.</para>
///
/// <para>Real, checkpoint-confirmed differences from FLUX.2 (see <see cref="QwenImageModel"/>'s own
/// doc comments, each citing a real bug found against the actual tensor names/shapes, not assumed):
/// <list type="bullet">
/// <item>Per-block AdaLN modulation (<c>img_mod.1</c>/<c>txt_mod.1</c>, <c>[6*d, d]</c> each) --
/// NOT FLUX.2's shared-across-all-blocks modulation.</item>
/// <item>Pre-modulation norm is an explicit affine-free LayerNorm (<c>LayerNormNoAffine</c>) applied
/// BEFORE <c>Flux::modulate</c>, not a direct modulate of the raw residual stream.</item>
/// <item>Plain GELU FFN (<c>net.0.proj</c>: <c>[4*d, d]</c>, single width) -- NOT FLUX.2's SiLU-gated
/// 2-way-split MLP.</item>
/// <item>Separate (non-fused) Q/K/V projections (<c>to_q</c>/<c>to_k</c>/<c>to_v</c>/
/// <c>add_q_proj</c>/<c>add_k_proj</c>/<c>add_v_proj</c>) -- NOT FLUX.2's fused QKV weight.</item>
/// <item>Final layer uses <c>norm_out.linear</c> (AdaLN shift/scale, <c>[2*d, d]</c>) + <c>proj_out</c>
/// -- NOT FLUX.2's <c>final_layer.adaLN_modulation.1</c>/<c>final_layer.linear</c> naming.</item>
/// </list></para>
///
/// <para><b>Not yet wired to a <c>ForwardGpu</c></b> -- this pass (docs/094 Phase 2) only completes
/// the GPU-resident weight upload. The dispatch-per-block forward pass (mirroring
/// <c>MMDiTModel.ForwardGpu</c>/<c>Flux2DiT.ForwardGpu</c>'s structure: per-block AdaLN modulation,
/// affine-free LayerNorm, joint QKV projection + per-stream QK-RMSNorm + 3D-RoPE + joint tiled
/// attention, plain-GELU FFN, gated residual add) is the next real chunk of this port.</para>
/// </summary>
public sealed class QwenImageGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor ImgInWeight { get; }
    public CoreTensor? ImgInBias { get; }
    public CoreTensor TxtInWeight { get; }
    public CoreTensor? TxtInBias { get; }

    public BlockGpuWeights[] Blocks { get; }

    public CoreTensor FinalNormLinearWeight { get; }
    public CoreTensor? FinalNormLinearBias { get; }
    public CoreTensor FinalProjOutWeight { get; }
    public CoreTensor? FinalProjOutBias { get; }

    public sealed class BlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor ImgModWeight { get; }
        public CoreTensor? ImgModBias { get; }
        public CoreTensor TxtModWeight { get; }
        public CoreTensor? TxtModBias { get; }

        public CoreTensor ImgToQWeight { get; }
        public CoreTensor ImgToKWeight { get; }
        public CoreTensor ImgToVWeight { get; }
        public CoreTensor TxtAddQWeight { get; }
        public CoreTensor TxtAddKWeight { get; }
        public CoreTensor TxtAddVWeight { get; }

        public CoreTensor ImgNormQScale { get; }
        public CoreTensor ImgNormKScale { get; }
        public CoreTensor TxtNormQScale { get; }
        public CoreTensor TxtNormKScale { get; }

        public CoreTensor ImgToOutWeight { get; }
        public CoreTensor? ImgToOutBias { get; }
        public CoreTensor TxtToAddOutWeight { get; }
        public CoreTensor? TxtToAddOutBias { get; }

        /// <summary>Plain (non-gated) GELU FFN up-projection: <c>[4*d, d]</c> -- confirmed against
        /// the real checkpoint's own tensor shape (see <see cref="QwenImageModel.FeedForward"/>'s
        /// doc comment: NOT a GEGLU-style <c>[8*d, d]</c> gate+value split).</summary>
        public CoreTensor ImgMlpUpWeight { get; }
        public CoreTensor? ImgMlpUpBias { get; }
        public CoreTensor ImgMlpDownWeight { get; }
        public CoreTensor? ImgMlpDownBias { get; }
        public CoreTensor TxtMlpUpWeight { get; }
        public CoreTensor? TxtMlpUpBias { get; }
        public CoreTensor TxtMlpDownWeight { get; }
        public CoreTensor? TxtMlpDownBias { get; }

        public BlockGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Func<string, float[]?> tryGetWeight, int idx, int d, int headDim, bool forceFp32 = false)
        {
            _backend = backend;
            string p = $"transformer_blocks.{idx}.";

            ImgModWeight = UploadWeight(backend, getWeight($"{p}img_mod.1.weight"), TensorShape.D2(6 * d, d), forceFp32);
            ImgModBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mod.1.bias"));
            TxtModWeight = UploadWeight(backend, getWeight($"{p}txt_mod.1.weight"), TensorShape.D2(6 * d, d), forceFp32);
            TxtModBias = UploadOptionalBias(backend, tryGetWeight($"{p}txt_mod.1.bias"));

            ImgToQWeight = UploadWeight(backend, getWeight($"{p}attn.to_q.weight"), TensorShape.D2(d, d), forceFp32);
            ImgToKWeight = UploadWeight(backend, getWeight($"{p}attn.to_k.weight"), TensorShape.D2(d, d), forceFp32);
            ImgToVWeight = UploadWeight(backend, getWeight($"{p}attn.to_v.weight"), TensorShape.D2(d, d), forceFp32);
            TxtAddQWeight = UploadWeight(backend, getWeight($"{p}attn.add_q_proj.weight"), TensorShape.D2(d, d), forceFp32);
            TxtAddKWeight = UploadWeight(backend, getWeight($"{p}attn.add_k_proj.weight"), TensorShape.D2(d, d), forceFp32);
            TxtAddVWeight = UploadWeight(backend, getWeight($"{p}attn.add_v_proj.weight"), TensorShape.D2(d, d), forceFp32);

            ImgNormQScale = backend.Upload(getWeight($"{p}attn.norm_q.weight"), TensorShape.D1(headDim), exact: true);
            ImgNormKScale = backend.Upload(getWeight($"{p}attn.norm_k.weight"), TensorShape.D1(headDim), exact: true);
            TxtNormQScale = backend.Upload(getWeight($"{p}attn.norm_added_q.weight"), TensorShape.D1(headDim), exact: true);
            TxtNormKScale = backend.Upload(getWeight($"{p}attn.norm_added_k.weight"), TensorShape.D1(headDim), exact: true);

            ImgToOutWeight = UploadWeight(backend, getWeight($"{p}attn.to_out.0.weight"), TensorShape.D2(d, d), forceFp32);
            ImgToOutBias = UploadOptionalBias(backend, tryGetWeight($"{p}attn.to_out.0.bias"));
            TxtToAddOutWeight = UploadWeight(backend, getWeight($"{p}attn.to_add_out.weight"), TensorShape.D2(d, d), forceFp32);
            TxtToAddOutBias = UploadOptionalBias(backend, tryGetWeight($"{p}attn.to_add_out.bias"));

            int ffDim = d * 4;
            ImgMlpUpWeight = UploadWeight(backend, getWeight($"{p}img_mlp.net.0.proj.weight"), TensorShape.D2(ffDim, d), forceFp32);
            ImgMlpUpBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mlp.net.0.proj.bias"));
            ImgMlpDownWeight = UploadWeight(backend, getWeight($"{p}img_mlp.net.2.weight"), TensorShape.D2(d, ffDim), forceFp32);
            ImgMlpDownBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mlp.net.2.bias"));
            TxtMlpUpWeight = UploadWeight(backend, getWeight($"{p}txt_mlp.net.0.proj.weight"), TensorShape.D2(ffDim, d), forceFp32);
            TxtMlpUpBias = UploadOptionalBias(backend, tryGetWeight($"{p}txt_mlp.net.0.proj.bias"));
            TxtMlpDownWeight = UploadWeight(backend, getWeight($"{p}txt_mlp.net.2.weight"), TensorShape.D2(d, ffDim), forceFp32);
            TxtMlpDownBias = UploadOptionalBias(backend, tryGetWeight($"{p}txt_mlp.net.2.bias"));
        }

        public void Dispose()
        {
            _backend.Free(ImgModWeight);
            if (ImgModBias is { } b1) _backend.Free(b1);
            _backend.Free(TxtModWeight);
            if (TxtModBias is { } b2) _backend.Free(b2);
            _backend.Free(ImgToQWeight);
            _backend.Free(ImgToKWeight);
            _backend.Free(ImgToVWeight);
            _backend.Free(TxtAddQWeight);
            _backend.Free(TxtAddKWeight);
            _backend.Free(TxtAddVWeight);
            _backend.Free(ImgNormQScale);
            _backend.Free(ImgNormKScale);
            _backend.Free(TxtNormQScale);
            _backend.Free(TxtNormKScale);
            _backend.Free(ImgToOutWeight);
            if (ImgToOutBias is { } b3) _backend.Free(b3);
            _backend.Free(TxtToAddOutWeight);
            if (TxtToAddOutBias is { } b4) _backend.Free(b4);
            _backend.Free(ImgMlpUpWeight);
            if (ImgMlpUpBias is { } b5) _backend.Free(b5);
            _backend.Free(ImgMlpDownWeight);
            if (ImgMlpDownBias is { } b6) _backend.Free(b6);
            _backend.Free(TxtMlpUpWeight);
            if (TxtMlpUpBias is { } b7) _backend.Free(b7);
            _backend.Free(TxtMlpDownWeight);
            if (TxtMlpDownBias is { } b8) _backend.Free(b8);
        }
    }

    public QwenImageGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Func<string, float[]?> tryGetWeight, int numLayers, int hiddenDim, int inChannels, int contextDim, int headDim)
    {
        _backend = backend;

        ImgInWeight = UploadWeight(backend, getWeight("img_in.weight"), TensorShape.D2(hiddenDim, inChannels));
        ImgInBias = UploadOptionalBias(backend, tryGetWeight("img_in.bias"));
        TxtInWeight = UploadWeight(backend, getWeight("txt_in.weight"), TensorShape.D2(hiddenDim, contextDim));
        TxtInBias = UploadOptionalBias(backend, tryGetWeight("txt_in.bias"));

        Blocks = new BlockGpuWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
            Blocks[i] = new BlockGpuWeights(backend, getWeight, tryGetWeight, i, hiddenDim, headDim);

        FinalNormLinearWeight = UploadWeight(backend, getWeight("norm_out.linear.weight"), TensorShape.D2(2 * hiddenDim, hiddenDim));
        FinalNormLinearBias = UploadOptionalBias(backend, tryGetWeight("norm_out.linear.bias"));
        FinalProjOutWeight = UploadWeight(backend, getWeight("proj_out.weight"), TensorShape.D2(inChannels, hiddenDim));
        FinalProjOutBias = UploadOptionalBias(backend, tryGetWeight("proj_out.bias"));

        GC.Collect();
    }

    private static CoreTensor? UploadOptionalBias(IComputeBackend backend, float[]? bias)
        => bias is null ? null : backend.Upload(bias, TensorShape.D1(bias.Length), exact: true);

    // Tried and REJECTED, 2026-09-20 (docs/094 Phase 2): forcing FP32 weight upload to test the
    // FP16-precision-sensitivity hypothesis (two independent block-by-block bisections found the
    // divergence onset shifts with input choice, pointing at rounding sensitivity, not a fixed
    // logic bug) hit a real `VkErrorOutOfHostMemory` -- this checkpoint's ~20.8B params at FP32
    // (~2x the FP16 upload's already-large footprint) simply do not fit in this hardware's
    // GPU-accessible memory. The confirming experiment could not even run to completion, let alone
    // produce a result -- a real, decisive negative finding (infeasibility, not inconclusiveness).
    // Kept `false` permanently; FP32 upload is not a viable path forward on this hardware for this
    // model's full 60-layer weight set.
    private static readonly bool ForceFp32WeightsExperiment = false;

    private static CoreTensor UploadWeight(IComputeBackend backend, float[] f32Data, TensorShape shape, bool forceFp32 = false)
    {
        bool skipReducedPrecision = ForceFp32WeightsExperiment || forceFp32;
        if (!skipReducedPrecision && backend.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[f32Data.Length];
            for (int i = 0; i < f32Data.Length; i++) half[i] = (Half)f32Data[i];
            return backend.UploadHalf(half, shape);
        }
        if (!skipReducedPrecision && backend.BestSgemmPrecision == SgemmPrecision.Bf16)
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
        if (ImgInBias is { } ib) _backend.Free(ib);
        _backend.Free(TxtInWeight);
        if (TxtInBias is { } tb) _backend.Free(tb);

        for (int i = 0; i < Blocks.Length; i++)
            Blocks[i]?.Dispose();

        _backend.Free(FinalNormLinearWeight);
        if (FinalNormLinearBias is { } fb) _backend.Free(fb);
        _backend.Free(FinalProjOutWeight);
        if (FinalProjOutBias is { } pb) _backend.Free(pb);
    }
}
