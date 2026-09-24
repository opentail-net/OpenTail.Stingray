using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// GPU-resident VRAM representation of FLUX.1 MM-DiT weights.
/// Uploads all linear projections, modulations, and normalization vectors to device VRAM once,
/// enabling zero CPU-GPU round-trip execution during all denoising steps.
/// </summary>
public sealed class FluxGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor ImgInWeight { get; }
    public CoreTensor ImgInBias { get; }
    public CoreTensor TxtInWeight { get; }
    public CoreTensor TxtInBias { get; }

    public DoubleBlockGpuWeights[] DoubleBlocks { get; }
    public SingleBlockGpuWeights[] SingleBlocks { get; }

    public CoreTensor FinalModWeight { get; }
    public CoreTensor FinalModBias { get; }
    public CoreTensor FinalLinearWeight { get; }
    public CoreTensor FinalLinearBias { get; }

    public sealed class DoubleBlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor ImgModWeight { get; }
        public CoreTensor ImgModBias { get; }
        public CoreTensor TxtModWeight { get; }
        public CoreTensor TxtModBias { get; }

        public CoreTensor TxtAttnQkv { get; }
        public CoreTensor ImgAttnQkv { get; }

        public CoreTensor TxtQkNormQScale { get; }
        public CoreTensor TxtQkNormKScale { get; }
        public CoreTensor ImgQkNormQScale { get; }
        public CoreTensor ImgQkNormKScale { get; }

        public CoreTensor TxtAttnProjWeight { get; }
        public CoreTensor TxtAttnProjBias { get; }
        public CoreTensor ImgAttnProjWeight { get; }
        public CoreTensor ImgAttnProjBias { get; }

        public CoreTensor TxtMlp0Weight { get; }
        public CoreTensor TxtMlp0Bias { get; }
        public CoreTensor TxtMlp2Weight { get; }
        public CoreTensor TxtMlp2Bias { get; }

        public CoreTensor ImgMlp0Weight { get; }
        public CoreTensor ImgMlp0Bias { get; }
        public CoreTensor ImgMlp2Weight { get; }
        public CoreTensor ImgMlp2Bias { get; }

        public DoubleBlockGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Func<string, float[]?> optWeight, int idx, int d, RawWeightSource? rawSource = null)
        {
            _backend = backend;
            string p = $"model.diffusion_model.double_blocks.{idx}";

            ImgModWeight = UploadWeight(backend, rawSource, $"{p}.img_mod.lin.weight", getWeight, TensorShape.D2(d * 6, d));
            ImgModBias = backend.Upload(getWeight($"{p}.img_mod.lin.bias"), TensorShape.D1(d * 6), exact: true);
            TxtModWeight = UploadWeight(backend, rawSource, $"{p}.txt_mod.lin.weight", getWeight, TensorShape.D2(d * 6, d));
            TxtModBias = backend.Upload(getWeight($"{p}.txt_mod.lin.bias"), TensorShape.D1(d * 6), exact: true);

            TxtAttnQkv = UploadWeight(backend, rawSource, $"{p}.txt_attn.qkv.weight", getWeight, TensorShape.D2(d * 3, d));
            ImgAttnQkv = UploadWeight(backend, rawSource, $"{p}.img_attn.qkv.weight", getWeight, TensorShape.D2(d * 3, d));

            TxtQkNormQScale = backend.Upload(getWeight($"{p}.txt_attn.norm.query_norm.scale"), TensorShape.D1(128), exact: true);
            TxtQkNormKScale = backend.Upload(getWeight($"{p}.txt_attn.norm.key_norm.scale"), TensorShape.D1(128), exact: true);
            ImgQkNormQScale = backend.Upload(getWeight($"{p}.img_attn.norm.query_norm.scale"), TensorShape.D1(128), exact: true);
            ImgQkNormKScale = backend.Upload(getWeight($"{p}.img_attn.norm.key_norm.scale"), TensorShape.D1(128), exact: true);

            TxtAttnProjWeight = UploadWeight(backend, rawSource, $"{p}.txt_attn.proj.weight", getWeight, TensorShape.D2(d, d));
            var txtProjB = optWeight($"{p}.txt_attn.proj.bias");
            TxtAttnProjBias = txtProjB is not null ? backend.Upload(txtProjB, TensorShape.D1(d), exact: true) : backend.Upload(new float[d], TensorShape.D1(d), exact: true);

            ImgAttnProjWeight = UploadWeight(backend, rawSource, $"{p}.img_attn.proj.weight", getWeight, TensorShape.D2(d, d));
            var imgProjB = optWeight($"{p}.img_attn.proj.bias");
            ImgAttnProjBias = imgProjB is not null ? backend.Upload(imgProjB, TensorShape.D1(d), exact: true) : backend.Upload(new float[d], TensorShape.D1(d), exact: true);

            TxtMlp0Weight = UploadWeight(backend, rawSource, $"{p}.txt_mlp.0.weight", getWeight, TensorShape.D2(d * 4, d));
            TxtMlp0Bias = backend.Upload(getWeight($"{p}.txt_mlp.0.bias"), TensorShape.D1(d * 4), exact: true);
            TxtMlp2Weight = UploadWeight(backend, rawSource, $"{p}.txt_mlp.2.weight", getWeight, TensorShape.D2(d, d * 4));
            TxtMlp2Bias = backend.Upload(getWeight($"{p}.txt_mlp.2.bias"), TensorShape.D1(d), exact: true);

            ImgMlp0Weight = UploadWeight(backend, rawSource, $"{p}.img_mlp.0.weight", getWeight, TensorShape.D2(d * 4, d));
            ImgMlp0Bias = backend.Upload(getWeight($"{p}.img_mlp.0.bias"), TensorShape.D1(d * 4), exact: true);
            ImgMlp2Weight = UploadWeight(backend, rawSource, $"{p}.img_mlp.2.weight", getWeight, TensorShape.D2(d, d * 4));
            ImgMlp2Bias = backend.Upload(getWeight($"{p}.img_mlp.2.bias"), TensorShape.D1(d), exact: true);
        }

        public void Dispose()
        {
            _backend.Free(ImgModWeight);
            _backend.Free(ImgModBias);
            _backend.Free(TxtModWeight);
            _backend.Free(TxtModBias);
            _backend.Free(TxtAttnQkv);
            _backend.Free(ImgAttnQkv);
            _backend.Free(TxtQkNormQScale);
            _backend.Free(TxtQkNormKScale);
            _backend.Free(ImgQkNormQScale);
            _backend.Free(ImgQkNormKScale);
            _backend.Free(TxtAttnProjWeight);
            _backend.Free(TxtAttnProjBias);
            _backend.Free(ImgAttnProjWeight);
            _backend.Free(ImgAttnProjBias);
            _backend.Free(TxtMlp0Weight);
            _backend.Free(TxtMlp0Bias);
            _backend.Free(TxtMlp2Weight);
            _backend.Free(TxtMlp2Bias);
            _backend.Free(ImgMlp0Weight);
            _backend.Free(ImgMlp0Bias);
            _backend.Free(ImgMlp2Weight);
            _backend.Free(ImgMlp2Bias);
        }
    }

    public sealed class SingleBlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor ModWeight { get; }
        public CoreTensor ModBias { get; }
        public CoreTensor Linear1Weight { get; }
        public CoreTensor QkNormQScale { get; }
        public CoreTensor QkNormKScale { get; }
        public CoreTensor Linear2Weight { get; }

        public SingleBlockGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int idx, int d, RawWeightSource? rawSource = null)
        {
            _backend = backend;
            string p = $"model.diffusion_model.single_blocks.{idx}";

            ModWeight = UploadWeight(backend, rawSource, $"{p}.modulation.lin.weight", getWeight, TensorShape.D2(d * 3, d));
            ModBias = backend.Upload(getWeight($"{p}.modulation.lin.bias"), TensorShape.D1(d * 3), exact: true);

            Linear1Weight = UploadWeight(backend, rawSource, $"{p}.linear1.weight", getWeight, TensorShape.D2(d * 7, d));
            QkNormQScale = backend.Upload(getWeight($"{p}.norm.query_norm.scale"), TensorShape.D1(128), exact: true);
            QkNormKScale = backend.Upload(getWeight($"{p}.norm.key_norm.scale"), TensorShape.D1(128), exact: true);
            Linear2Weight = UploadWeight(backend, rawSource, $"{p}.linear2.weight", getWeight, TensorShape.D2(d, d * 5));
        }

        public void Dispose()
        {
            _backend.Free(ModWeight);
            _backend.Free(ModBias);
            _backend.Free(Linear1Weight);
            _backend.Free(QkNormQScale);
            _backend.Free(QkNormKScale);
            _backend.Free(Linear2Weight);
        }
    }

    public FluxGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Func<string, float[]?> optWeight, FluxParams p, RawWeightSource? rawSource = null)
    {
        _backend = backend;
        int d = p.HiddenSize;

        ImgInWeight = UploadWeight(backend, getWeight("model.diffusion_model.img_in.weight"), TensorShape.D2(d, p.InChannels));
        ImgInBias = backend.Upload(getWeight("model.diffusion_model.img_in.bias"), TensorShape.D1(d), exact: true);

        TxtInWeight = UploadWeight(backend, getWeight("model.diffusion_model.txt_in.weight"), TensorShape.D2(d, p.ContextDim));
        TxtInBias = backend.Upload(getWeight("model.diffusion_model.txt_in.bias"), TensorShape.D1(d), exact: true);

        DoubleBlocks = new DoubleBlockGpuWeights[p.DoubleBlocks];
        for (int i = 0; i < p.DoubleBlocks; i++)
            DoubleBlocks[i] = new DoubleBlockGpuWeights(backend, getWeight, optWeight, i, d, rawSource);

        SingleBlocks = new SingleBlockGpuWeights[p.SingleBlocks];
        for (int i = 0; i < p.SingleBlocks; i++)
            SingleBlocks[i] = new SingleBlockGpuWeights(backend, getWeight, i, d, rawSource);

        FinalModWeight = UploadWeight(backend, getWeight("model.diffusion_model.final_layer.adaLN_modulation.1.weight"), TensorShape.D2(d * 2, d));
        FinalModBias = backend.Upload(getWeight("model.diffusion_model.final_layer.adaLN_modulation.1.bias"), TensorShape.D1(d * 2), exact: true);

        FinalLinearWeight = UploadWeight(backend, getWeight("model.diffusion_model.final_layer.linear.weight"), TensorShape.D2(p.OutChannels, d));
        FinalLinearBias = backend.Upload(getWeight("model.diffusion_model.final_layer.linear.bias"), TensorShape.D1(p.OutChannels), exact: true);

        GC.Collect();
    }

    /// <summary>Q3_K/Q4_K/Q5_K weights stay quantized on the GPU when the backend supports it (see
    /// <see cref="QuantizedGpuUpload"/>): the Q4_K_S DiT is ~6.5 GB there instead of ~24 GB of FP16.
    /// STINGRAY_FLUX_GPU_FP16=1 forces the FP16 path.</summary>
    private static CoreTensor UploadWeight(IComputeBackend backend, RawWeightSource? raw, string name,
        Func<string, float[]> getWeight, TensorShape shape)
        => QuantizedGpuUpload.TryUpload(backend, raw, name, shape, "STINGRAY_FLUX_GPU_FP16")
           ?? UploadWeight(backend, getWeight(name), shape);

    private static CoreTensor UploadWeight(IComputeBackend backend, float[] f32Data, TensorShape shape)
    {
        if (backend.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[f32Data.Length];
            System.Numerics.Tensors.TensorPrimitives.ConvertToHalf(f32Data, half);
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
        _backend.Free(ImgInBias);
        _backend.Free(TxtInWeight);
        _backend.Free(TxtInBias);

        for (int i = 0; i < DoubleBlocks.Length; i++)
            DoubleBlocks[i]?.Dispose();

        for (int i = 0; i < SingleBlocks.Length; i++)
            SingleBlocks[i]?.Dispose();

        _backend.Free(FinalModWeight);
        _backend.Free(FinalModBias);
        _backend.Free(FinalLinearWeight);
        _backend.Free(FinalLinearBias);
    }
}
