using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// GPU-resident VRAM representation of Stable Audio 3 DiT weights.
/// Holds all 20 transformer layers (QKV, Out, CrossQ, CrossKV, CrossOut, SwiGLU FFN),
/// input/output projections, memory tokens, and conditioning embedders resident in VRAM as FP16.
/// Sized and structured for zero-allocation reuse across every Euler ODE step.
/// </summary>
public sealed class StableAudioGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public sealed class BlockWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public float[] ToScaleShiftGate { get; }
        public CoreTensor PreNormGamma { get; }
        public CoreTensor SelfAttnQkvW { get; }
        public CoreTensor SelfAttnQNormGamma { get; }
        public CoreTensor SelfAttnKNormGamma { get; }
        public CoreTensor SelfAttnToOutW { get; }
        public CoreTensor CrossAttendNormGamma { get; }
        public CoreTensor CrossAttnQW { get; }
        public CoreTensor CrossAttnKvW { get; }
        public CoreTensor CrossAttnQNormGamma { get; }
        public CoreTensor CrossAttnKNormGamma { get; }
        public CoreTensor CrossAttnToOutW { get; }
        public CoreTensor FfNormGamma { get; }
        public CoreTensor Ff0UpW { get; }
        public CoreTensor Ff0UpB { get; }
        public CoreTensor Ff0GateW { get; }
        public CoreTensor Ff0GateB { get; }
        public CoreTensor Ff2W { get; }
        public CoreTensor Ff2B { get; }

        public BlockWeights(IComputeBackend backend, Func<string, float[]> getWeight, string prefix, int dim, int headDim, int ffInner)
        {
            _backend = backend;

            ToScaleShiftGate = getWeight($"{prefix}.to_scale_shift_gate");
            PreNormGamma = backend.Upload(getWeight($"{prefix}.pre_norm.gamma"), TensorShape.D1(dim), exact: true);
            SelfAttnQkvW = UploadWeight(backend, getWeight($"{prefix}.self_attn.to_qkv.weight"), TensorShape.D2(3 * dim, dim));
            SelfAttnQNormGamma = backend.Upload(getWeight($"{prefix}.self_attn.q_norm.gamma"), TensorShape.D1(headDim), exact: true);
            SelfAttnKNormGamma = backend.Upload(getWeight($"{prefix}.self_attn.k_norm.gamma"), TensorShape.D1(headDim), exact: true);
            SelfAttnToOutW = UploadWeight(backend, getWeight($"{prefix}.self_attn.to_out.weight"), TensorShape.D2(dim, dim));

            CrossAttendNormGamma = backend.Upload(getWeight($"{prefix}.cross_attend_norm.gamma"), TensorShape.D1(dim), exact: true);
            CrossAttnQW = UploadWeight(backend, getWeight($"{prefix}.cross_attn.to_q.weight"), TensorShape.D2(dim, dim));
            CrossAttnKvW = UploadWeight(backend, getWeight($"{prefix}.cross_attn.to_kv.weight"), TensorShape.D2(2 * dim, dim));
            CrossAttnQNormGamma = backend.Upload(getWeight($"{prefix}.cross_attn.q_norm.gamma"), TensorShape.D1(headDim), exact: true);
            CrossAttnKNormGamma = backend.Upload(getWeight($"{prefix}.cross_attn.k_norm.gamma"), TensorShape.D1(headDim), exact: true);
            CrossAttnToOutW = UploadWeight(backend, getWeight($"{prefix}.cross_attn.to_out.weight"), TensorShape.D2(dim, dim));

            FfNormGamma = backend.Upload(getWeight($"{prefix}.ff_norm.gamma"), TensorShape.D1(dim), exact: true);

            var ff0W = getWeight($"{prefix}.ff.ff.0.proj.weight");
            var ff0B = getWeight($"{prefix}.ff.ff.0.proj.bias");

            var upW = new float[ffInner * dim];
            var gateW = new float[ffInner * dim];
            Array.Copy(ff0W, 0, upW, 0, ffInner * dim);
            Array.Copy(ff0W, ffInner * dim, gateW, 0, ffInner * dim);

            var upB = new float[ffInner];
            var gateB = new float[ffInner];
            Array.Copy(ff0B, 0, upB, 0, ffInner);
            Array.Copy(ff0B, ffInner, gateB, 0, ffInner);

            Ff0UpW = UploadWeight(backend, upW, TensorShape.D2(ffInner, dim));
            Ff0UpB = backend.Upload(upB, TensorShape.D1(ffInner), exact: true);
            Ff0GateW = UploadWeight(backend, gateW, TensorShape.D2(ffInner, dim));
            Ff0GateB = backend.Upload(gateB, TensorShape.D1(ffInner), exact: true);

            Ff2W = UploadWeight(backend, getWeight($"{prefix}.ff.ff.2.weight"), TensorShape.D2(dim, ffInner));
            Ff2B = backend.Upload(getWeight($"{prefix}.ff.ff.2.bias"), TensorShape.D1(dim), exact: true);
        }

        public void Dispose()
        {
            _backend.Free(PreNormGamma);
            _backend.Free(SelfAttnQkvW);
            _backend.Free(SelfAttnQNormGamma);
            _backend.Free(SelfAttnKNormGamma);
            _backend.Free(SelfAttnToOutW);
            _backend.Free(CrossAttendNormGamma);
            _backend.Free(CrossAttnQW);
            _backend.Free(CrossAttnKvW);
            _backend.Free(CrossAttnQNormGamma);
            _backend.Free(CrossAttnKNormGamma);
            _backend.Free(CrossAttnToOutW);
            _backend.Free(FfNormGamma);
            _backend.Free(Ff0UpW);
            _backend.Free(Ff0UpB);
            _backend.Free(Ff0GateW);
            _backend.Free(Ff0GateB);
            _backend.Free(Ff2W);
            _backend.Free(Ff2B);
        }
    }

    public CoreTensor PreprocessConvW { get; }
    public CoreTensor ProjInW { get; }
    public CoreTensor MemoryTokens { get; }
    public CoreTensor ProjOutW { get; }
    public CoreTensor PostprocessConvW { get; }

    public BlockWeights[] Layers { get; }

    public StableAudioGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int numLayers = 20, int dim = 1024, int ioChannels = 256, int headDim = 64, int ffInner = 4096, int memoryTokens = 64)
    {
        _backend = backend;

        PreprocessConvW = UploadWeight(backend, getWeight("model.model.preprocess_conv.weight"), TensorShape.D2(ioChannels, ioChannels));
        ProjInW = UploadWeight(backend, getWeight("model.model.transformer.project_in.weight"), TensorShape.D2(dim, ioChannels));
        MemoryTokens = backend.Upload(getWeight("model.model.transformer.memory_tokens"), TensorShape.D2(memoryTokens, dim), exact: true);
        ProjOutW = UploadWeight(backend, getWeight("model.model.transformer.project_out.weight"), TensorShape.D2(ioChannels, dim));
        PostprocessConvW = UploadWeight(backend, getWeight("model.model.postprocess_conv.weight"), TensorShape.D2(ioChannels, ioChannels));

        Layers = new BlockWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Layers[i] = new BlockWeights(backend, getWeight, $"model.model.transformer.layers.{i}", dim, headDim, ffInner);
        }
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

        _backend.Free(PreprocessConvW);
        _backend.Free(ProjInW);
        _backend.Free(MemoryTokens);
        _backend.Free(ProjOutW);
        _backend.Free(PostprocessConvW);

        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i].Dispose();
        }
    }
}
