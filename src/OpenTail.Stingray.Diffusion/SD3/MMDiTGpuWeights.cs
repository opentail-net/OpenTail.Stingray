using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.SD3;

/// <summary>
/// GPU-resident VRAM representation of Stable Diffusion 3 / 3.5 MMDiT weights.
/// Uploads all linear projections, modulations, and normalizations to device VRAM once,
/// enabling zero CPU-GPU round-trip execution during all denoising steps.
/// </summary>
public sealed class MMDiTGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor XEmbedderWeight { get; }
    public CoreTensor? XEmbedderBias { get; }
    public CoreTensor ContextEmbedderWeight { get; }
    public CoreTensor? ContextEmbedderBias { get; }

    public JointBlockGpuWeights[] JointBlocks { get; }

    public CoreTensor FinalModWeight { get; }
    public CoreTensor FinalModBias { get; }
    public CoreTensor FinalLinearWeight { get; }
    public CoreTensor FinalLinearBias { get; }

    /// <summary>Timestep/pooled-Y embedder weights (2026-09-20 perf pass, docs/094 Phase 1):
    /// previously not GPU-resident at all -- `ForwardGpu` computed these via the CPU-path `Lin()`
    /// helper (Upload+Sgemm+Download, 3 separate submit+fence-wait cycles per call, 4 calls total)
    /// even when running on a Vulkan backend, then re-uploaded the CPU result back to GPU as
    /// `tVecGpu`. Real `STINGRAY_PROFILE_GPU_SPLIT=1` profiling found this and `x_embedder`'s own
    /// analogous CPU round-trip together account for roughly half of a real generation's ~1300
    /// total submit+fence-wait cycles. Loading these here lets `ForwardGpu` compute the whole
    /// timestep/pooled-Y embedding GPU-resident, batched with everything else.</summary>
    public CoreTensor TEmbedder0Weight { get; }
    public CoreTensor? TEmbedder0Bias { get; }
    public CoreTensor TEmbedder2Weight { get; }
    public CoreTensor? TEmbedder2Bias { get; }
    public CoreTensor YEmbedder0Weight { get; }
    public CoreTensor? YEmbedder0Bias { get; }
    public CoreTensor YEmbedder2Weight { get; }
    public CoreTensor? YEmbedder2Bias { get; }

    public sealed class JointBlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public bool DualAttn { get; }
        public bool ContextPreOnly { get; }
        public int ImgModChunks { get; }
        public int TxtModChunks { get; }

        public CoreTensor ImgModWeight { get; }
        public CoreTensor ImgModBias { get; }
        public CoreTensor TxtModWeight { get; }
        public CoreTensor TxtModBias { get; }

        public CoreTensor ImgAttnQkvWeight { get; }
        public CoreTensor? ImgAttnQkvBias { get; }
        public CoreTensor TxtAttnQkvWeight { get; }
        public CoreTensor? TxtAttnQkvBias { get; }

        public CoreTensor? ImgAttnLnQ { get; }
        public CoreTensor? ImgAttnLnK { get; }
        public CoreTensor? TxtAttnLnQ { get; }
        public CoreTensor? TxtAttnLnK { get; }

        public CoreTensor ImgAttnProjWeight { get; }
        public CoreTensor? ImgAttnProjBias { get; }
        public CoreTensor? TxtAttnProjWeight { get; }
        public CoreTensor? TxtAttnProjBias { get; }

        public CoreTensor? ImgAttn2QkvWeight { get; }
        public CoreTensor? ImgAttn2QkvBias { get; }
        public CoreTensor? ImgAttn2LnQ { get; }
        public CoreTensor? ImgAttn2LnK { get; }
        public CoreTensor? ImgAttn2ProjWeight { get; }
        public CoreTensor? ImgAttn2ProjBias { get; }

        public CoreTensor ImgMlp0Weight { get; }
        public CoreTensor? ImgMlp0Bias { get; }
        public CoreTensor ImgMlp2Weight { get; }
        public CoreTensor? ImgMlp2Bias { get; }

        public CoreTensor? TxtMlp0Weight { get; }
        public CoreTensor? TxtMlp0Bias { get; }
        public CoreTensor? TxtMlp2Weight { get; }
        public CoreTensor? TxtMlp2Bias { get; }

        public JointBlockGpuWeights(
            IComputeBackend backend,
            Func<string, float[]> getWeight,
            Func<string, float[]?> tryGetWeight,
            int idx,
            int d,
            int totalDepth,
            int headDim,
            RawWeightSource? rawSource = null)
        {
            _backend = backend;
            string blk = $"joint_blocks.{idx}";

            DualAttn = tryGetWeight($"{blk}.x_block.attn2.qkv.weight") is not null;
            ContextPreOnly = idx == totalDepth - 1;
            ImgModChunks = DualAttn ? 9 : 6;
            TxtModChunks = ContextPreOnly ? 2 : 6;

            ImgModWeight = UploadWeight(backend, rawSource, $"{blk}.x_block.adaLN_modulation.1.weight", getWeight, TensorShape.D2(ImgModChunks * d, d));
            var imgModB = tryGetWeight($"{blk}.x_block.adaLN_modulation.1.bias");
            ImgModBias = imgModB is not null ? backend.Upload(imgModB, TensorShape.D1(ImgModChunks * d), exact: true) : backend.Upload(new float[ImgModChunks * d], TensorShape.D1(ImgModChunks * d), exact: true);

            TxtModWeight = UploadWeight(backend, rawSource, $"{blk}.context_block.adaLN_modulation.1.weight", getWeight, TensorShape.D2(TxtModChunks * d, d));
            var txtModB = tryGetWeight($"{blk}.context_block.adaLN_modulation.1.bias");
            TxtModBias = txtModB is not null ? backend.Upload(txtModB, TensorShape.D1(TxtModChunks * d), exact: true) : backend.Upload(new float[TxtModChunks * d], TensorShape.D1(TxtModChunks * d), exact: true);

            ImgAttnQkvWeight = UploadWeight(backend, rawSource, $"{blk}.x_block.attn.qkv.weight", getWeight, TensorShape.D2(3 * d, d));
            var imgQkvB = tryGetWeight($"{blk}.x_block.attn.qkv.bias");
            if (imgQkvB is not null) ImgAttnQkvBias = backend.Upload(imgQkvB, TensorShape.D1(3 * d), exact: true);

            TxtAttnQkvWeight = UploadWeight(backend, rawSource, $"{blk}.context_block.attn.qkv.weight", getWeight, TensorShape.D2(3 * d, d));
            var txtQkvB = tryGetWeight($"{blk}.context_block.attn.qkv.bias");
            if (txtQkvB is not null) TxtAttnQkvBias = backend.Upload(txtQkvB, TensorShape.D1(3 * d), exact: true);

            var imgLnQ = tryGetWeight($"{blk}.x_block.attn.ln_q.weight");
            if (imgLnQ is not null) ImgAttnLnQ = backend.Upload(imgLnQ, TensorShape.D1(headDim), exact: true);
            var imgLnK = tryGetWeight($"{blk}.x_block.attn.ln_k.weight");
            if (imgLnK is not null) ImgAttnLnK = backend.Upload(imgLnK, TensorShape.D1(headDim), exact: true);

            var txtLnQ = tryGetWeight($"{blk}.context_block.attn.ln_q.weight");
            if (txtLnQ is not null) TxtAttnLnQ = backend.Upload(txtLnQ, TensorShape.D1(headDim), exact: true);
            var txtLnK = tryGetWeight($"{blk}.context_block.attn.ln_k.weight");
            if (txtLnK is not null) TxtAttnLnK = backend.Upload(txtLnK, TensorShape.D1(headDim), exact: true);

            ImgAttnProjWeight = UploadWeight(backend, rawSource, $"{blk}.x_block.attn.proj.weight", getWeight, TensorShape.D2(d, d));
            var imgProjB = tryGetWeight($"{blk}.x_block.attn.proj.bias");
            if (imgProjB is not null) ImgAttnProjBias = backend.Upload(imgProjB, TensorShape.D1(d), exact: true);

            if (!ContextPreOnly)
            {
                TxtAttnProjWeight = UploadWeight(backend, rawSource, $"{blk}.context_block.attn.proj.weight", getWeight, TensorShape.D2(d, d));
                var txtProjB = tryGetWeight($"{blk}.context_block.attn.proj.bias");
                if (txtProjB is not null) TxtAttnProjBias = backend.Upload(txtProjB, TensorShape.D1(d), exact: true);
            }

            if (DualAttn)
            {
                ImgAttn2QkvWeight = UploadWeight(backend, rawSource, $"{blk}.x_block.attn2.qkv.weight", getWeight, TensorShape.D2(3 * d, d));
                var img2QkvB = tryGetWeight($"{blk}.x_block.attn2.qkv.bias");
                if (img2QkvB is not null) ImgAttn2QkvBias = backend.Upload(img2QkvB, TensorShape.D1(3 * d), exact: true);

                var img2LnQ = tryGetWeight($"{blk}.x_block.attn2.ln_q.weight");
                if (img2LnQ is not null) ImgAttn2LnQ = backend.Upload(img2LnQ, TensorShape.D1(headDim), exact: true);
                var img2LnK = tryGetWeight($"{blk}.x_block.attn2.ln_k.weight");
                if (img2LnK is not null) ImgAttn2LnK = backend.Upload(img2LnK, TensorShape.D1(headDim), exact: true);

                ImgAttn2ProjWeight = UploadWeight(backend, rawSource, $"{blk}.x_block.attn2.proj.weight", getWeight, TensorShape.D2(d, d));
                var img2ProjB = tryGetWeight($"{blk}.x_block.attn2.proj.bias");
                if (img2ProjB is not null) ImgAttn2ProjBias = backend.Upload(img2ProjB, TensorShape.D1(d), exact: true);
            }

            int mlpHidden = d * 4;
            ImgMlp0Weight = UploadWeight(backend, rawSource, $"{blk}.x_block.mlp.fc1.weight", getWeight, TensorShape.D2(mlpHidden, d));
            var imgMlp0B = tryGetWeight($"{blk}.x_block.mlp.fc1.bias");
            if (imgMlp0B is not null) ImgMlp0Bias = backend.Upload(imgMlp0B, TensorShape.D1(mlpHidden), exact: true);

            ImgMlp2Weight = UploadWeight(backend, rawSource, $"{blk}.x_block.mlp.fc2.weight", getWeight, TensorShape.D2(d, mlpHidden));
            var imgMlp2B = tryGetWeight($"{blk}.x_block.mlp.fc2.bias");
            if (imgMlp2B is not null) ImgMlp2Bias = backend.Upload(imgMlp2B, TensorShape.D1(d), exact: true);

            if (!ContextPreOnly)
            {
                TxtMlp0Weight = UploadWeight(backend, rawSource, $"{blk}.context_block.mlp.fc1.weight", getWeight, TensorShape.D2(mlpHidden, d));
                var txtMlp0B = tryGetWeight($"{blk}.context_block.mlp.fc1.bias");
                if (txtMlp0B is not null) TxtMlp0Bias = backend.Upload(txtMlp0B, TensorShape.D1(mlpHidden), exact: true);

                TxtMlp2Weight = UploadWeight(backend, rawSource, $"{blk}.context_block.mlp.fc2.weight", getWeight, TensorShape.D2(d, mlpHidden));
                var txtMlp2B = tryGetWeight($"{blk}.context_block.mlp.fc2.bias");
                if (txtMlp2B is not null) TxtMlp2Bias = backend.Upload(txtMlp2B, TensorShape.D1(d), exact: true);
            }
        }

        public void Dispose()
        {
            _backend.Free(ImgModWeight);
            _backend.Free(ImgModBias);
            _backend.Free(TxtModWeight);
            _backend.Free(TxtModBias);

            _backend.Free(ImgAttnQkvWeight);
            if (ImgAttnQkvBias is not null) _backend.Free(ImgAttnQkvBias);
            _backend.Free(TxtAttnQkvWeight);
            if (TxtAttnQkvBias is not null) _backend.Free(TxtAttnQkvBias);

            if (ImgAttnLnQ is not null) _backend.Free(ImgAttnLnQ);
            if (ImgAttnLnK is not null) _backend.Free(ImgAttnLnK);
            if (TxtAttnLnQ is not null) _backend.Free(TxtAttnLnQ);
            if (TxtAttnLnK is not null) _backend.Free(TxtAttnLnK);

            _backend.Free(ImgAttnProjWeight);
            if (ImgAttnProjBias is not null) _backend.Free(ImgAttnProjBias);
            if (TxtAttnProjWeight is not null) _backend.Free(TxtAttnProjWeight);
            if (TxtAttnProjBias is not null) _backend.Free(TxtAttnProjBias);

            if (ImgAttn2QkvWeight is not null) _backend.Free(ImgAttn2QkvWeight);
            if (ImgAttn2QkvBias is not null) _backend.Free(ImgAttn2QkvBias);
            if (ImgAttn2LnQ is not null) _backend.Free(ImgAttn2LnQ);
            if (ImgAttn2LnK is not null) _backend.Free(ImgAttn2LnK);
            if (ImgAttn2ProjWeight is not null) _backend.Free(ImgAttn2ProjWeight);
            if (ImgAttn2ProjBias is not null) _backend.Free(ImgAttn2ProjBias);

            _backend.Free(ImgMlp0Weight);
            if (ImgMlp0Bias is not null) _backend.Free(ImgMlp0Bias);
            _backend.Free(ImgMlp2Weight);
            if (ImgMlp2Bias is not null) _backend.Free(ImgMlp2Bias);

            if (TxtMlp0Weight is not null) _backend.Free(TxtMlp0Weight);
            if (TxtMlp0Bias is not null) _backend.Free(TxtMlp0Bias);
            if (TxtMlp2Weight is not null) _backend.Free(TxtMlp2Weight);
            if (TxtMlp2Bias is not null) _backend.Free(TxtMlp2Bias);
        }
    }

    public MMDiTGpuWeights(
        IComputeBackend backend,
        Func<string, float[]> getWeight,
        Func<string, float[]?> tryGetWeight,
        int hiddenSize,
        int depth,
        int headDim,
        int inChannels,
        int outChannels,
        int patchSize,
        int contextSize,
        int admInChannels,
        RawWeightSource? rawSource = null)
    {
        _backend = backend;
        int d = hiddenSize;
        int inPatchDim = inChannels * patchSize * patchSize;
        int outPatchDim = outChannels * patchSize * patchSize;

        XEmbedderWeight = UploadWeight(backend, getWeight("x_embedder.proj.weight"), TensorShape.D2(d, inPatchDim));
        var xEmbB = tryGetWeight("x_embedder.proj.bias");
        if (xEmbB is not null) XEmbedderBias = backend.Upload(xEmbB, TensorShape.D1(d), exact: true);

        ContextEmbedderWeight = UploadWeight(backend, getWeight("context_embedder.weight"), TensorShape.D2(d, contextSize));
        var ctxEmbB = tryGetWeight("context_embedder.bias");
        if (ctxEmbB is not null) ContextEmbedderBias = backend.Upload(ctxEmbB, TensorShape.D1(d), exact: true);

        // t_embedder: Fourier(timestep) [256] -> [d] -> SiLU -> [d]
        TEmbedder0Weight = UploadWeight(backend, getWeight("t_embedder.mlp.0.weight"), TensorShape.D2(d, 256));
        var tEmb0B = tryGetWeight("t_embedder.mlp.0.bias");
        if (tEmb0B is not null) TEmbedder0Bias = backend.Upload(tEmb0B, TensorShape.D1(d), exact: true);
        TEmbedder2Weight = UploadWeight(backend, getWeight("t_embedder.mlp.2.weight"), TensorShape.D2(d, d));
        var tEmb2B = tryGetWeight("t_embedder.mlp.2.bias");
        if (tEmb2B is not null) TEmbedder2Bias = backend.Upload(tEmb2B, TensorShape.D1(d), exact: true);

        // y_embedder: pooledY [admInChannels] -> [d] -> SiLU -> [d]
        YEmbedder0Weight = UploadWeight(backend, getWeight("y_embedder.mlp.0.weight"), TensorShape.D2(d, admInChannels));
        var yEmb0B = tryGetWeight("y_embedder.mlp.0.bias");
        if (yEmb0B is not null) YEmbedder0Bias = backend.Upload(yEmb0B, TensorShape.D1(d), exact: true);
        YEmbedder2Weight = UploadWeight(backend, getWeight("y_embedder.mlp.2.weight"), TensorShape.D2(d, d));
        var yEmb2B = tryGetWeight("y_embedder.mlp.2.bias");
        if (yEmb2B is not null) YEmbedder2Bias = backend.Upload(yEmb2B, TensorShape.D1(d), exact: true);

        JointBlocks = new JointBlockGpuWeights[depth];
        for (int i = 0; i < depth; i++)
        {
            JointBlocks[i] = new JointBlockGpuWeights(backend, getWeight, tryGetWeight, i, d, depth, headDim, rawSource);
        }

        FinalModWeight = UploadWeight(backend, getWeight("final_layer.adaLN_modulation.1.weight"), TensorShape.D2(2 * d, d));
        var finalModB = tryGetWeight("final_layer.adaLN_modulation.1.bias");
        FinalModBias = finalModB is not null ? backend.Upload(finalModB, TensorShape.D1(2 * d), exact: true) : backend.Upload(new float[2 * d], TensorShape.D1(2 * d), exact: true);

        FinalLinearWeight = UploadWeight(backend, getWeight("final_layer.linear.weight"), TensorShape.D2(outPatchDim, d));
        var finalLinB = tryGetWeight("final_layer.linear.bias");
        FinalLinearBias = finalLinB is not null ? backend.Upload(finalLinB, TensorShape.D1(outPatchDim), exact: true) : backend.Upload(new float[outPatchDim], TensorShape.D1(outPatchDim), exact: true);

        GC.Collect();
    }

    /// <summary>Q3_K/Q4_K/Q5_K weights stay quantized on the GPU when the backend supports it
    /// (see <see cref="QuantizedGpuUpload"/>); STINGRAY_SD3_GPU_FP16=1 forces the FP16 path.</summary>
    private static CoreTensor UploadWeight(IComputeBackend backend, RawWeightSource? raw, string name,
        Func<string, float[]> getWeight, TensorShape shape)
        => QuantizedGpuUpload.TryUpload(backend, raw, name, shape, "STINGRAY_SD3_GPU_FP16")
           ?? UploadWeight(backend, getWeight(name), shape);

    private static CoreTensor UploadWeight(IComputeBackend backend, float[] f32Data, TensorShape shape)
    {
        if (Environment.GetEnvironmentVariable("STINGRAY_SD3_FORCE_GPU_WEIGHTS_F32") == "1")
            return backend.Upload(f32Data, shape, exact: true);
        if (backend.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[f32Data.Length];
            TensorPrimitives.ConvertToHalf(f32Data, half);
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

        _backend.Free(XEmbedderWeight);
        if (XEmbedderBias is not null) _backend.Free(XEmbedderBias);
        _backend.Free(ContextEmbedderWeight);
        if (ContextEmbedderBias is not null) _backend.Free(ContextEmbedderBias);

        _backend.Free(TEmbedder0Weight);
        if (TEmbedder0Bias is not null) _backend.Free(TEmbedder0Bias);
        _backend.Free(TEmbedder2Weight);
        if (TEmbedder2Bias is not null) _backend.Free(TEmbedder2Bias);
        _backend.Free(YEmbedder0Weight);
        if (YEmbedder0Bias is not null) _backend.Free(YEmbedder0Bias);
        _backend.Free(YEmbedder2Weight);
        if (YEmbedder2Bias is not null) _backend.Free(YEmbedder2Bias);

        for (int i = 0; i < JointBlocks.Length; i++)
            JointBlocks[i]?.Dispose();

        _backend.Free(FinalModWeight);
        _backend.Free(FinalModBias);
        _backend.Free(FinalLinearWeight);
        _backend.Free(FinalLinearBias);
    }
}
