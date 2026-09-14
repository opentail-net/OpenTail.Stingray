using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// GPU-resident VRAM representation of Parler's T5 text encoder weights. Structurally identical
/// to `OpenTail.Stingray.Diffusion.T5GpuWeights` (FLUX's own T5-XXL GPU-residency class, docs/071,
/// already proven) -- duplicated here rather than referenced directly because
/// `OpenTail.Stingray.Diffusion` already has a `ProjectReference` on `OpenTail.Stingray.Audio`
/// (confirmed via its own .csproj), so `Audio -> Diffusion` would be a circular project reference.
/// Same real T5 tensor names/math (unscaled attention, RMSNorm, gated-GELU FFN), only the
/// construction dims differ (Parler/T5-Large: dim=1024/ffDim=2816/24 layers vs T5-XXL's
/// 4096/10240/24).
/// </summary>
public sealed class ParlerT5GpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public T5LayerGpuWeights[] Layers { get; }
    public CoreTensor FinalLayerNormWeight { get; }

    public sealed class T5LayerGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor LayerNorm0Weight { get; }
        public CoreTensor QWeight { get; }
        public CoreTensor KWeight { get; }
        public CoreTensor VWeight { get; }
        public CoreTensor OWeight { get; }

        public CoreTensor LayerNorm1Weight { get; }
        public CoreTensor Wi0Weight { get; }
        public CoreTensor Wi1Weight { get; }
        public CoreTensor WoWight { get; }

        public T5LayerGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int idx, int dim, int ffDim)
        {
            _backend = backend;
            string p = $"encoder.block.{idx}.layer";

            LayerNorm0Weight = backend.Upload(getWeight($"{p}.0.layer_norm.weight"), TensorShape.D1(dim), exact: true);
            QWeight = UploadWeight(backend, getWeight($"{p}.0.SelfAttention.q.weight"), TensorShape.D2(dim, dim));
            KWeight = UploadWeight(backend, getWeight($"{p}.0.SelfAttention.k.weight"), TensorShape.D2(dim, dim));
            VWeight = UploadWeight(backend, getWeight($"{p}.0.SelfAttention.v.weight"), TensorShape.D2(dim, dim));
            OWeight = UploadWeight(backend, getWeight($"{p}.0.SelfAttention.o.weight"), TensorShape.D2(dim, dim));

            LayerNorm1Weight = backend.Upload(getWeight($"{p}.1.layer_norm.weight"), TensorShape.D1(dim), exact: true);
            Wi0Weight = UploadWeight(backend, getWeight($"{p}.1.DenseReluDense.wi_0.weight"), TensorShape.D2(ffDim, dim));
            Wi1Weight = UploadWeight(backend, getWeight($"{p}.1.DenseReluDense.wi_1.weight"), TensorShape.D2(ffDim, dim));
            WoWight = UploadWeight(backend, getWeight($"{p}.1.DenseReluDense.wo.weight"), TensorShape.D2(dim, ffDim));
        }

        public void Dispose()
        {
            _backend.Free(LayerNorm0Weight);
            _backend.Free(QWeight);
            _backend.Free(KWeight);
            _backend.Free(VWeight);
            _backend.Free(OWeight);
            _backend.Free(LayerNorm1Weight);
            _backend.Free(Wi0Weight);
            _backend.Free(Wi1Weight);
            _backend.Free(WoWight);
        }
    }

    public ParlerT5GpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int numLayers, int dim, int ffDim)
    {
        _backend = backend;
        Layers = new T5LayerGpuWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Layers[i] = new T5LayerGpuWeights(backend, getWeight, i, dim, ffDim);
        }
        FinalLayerNormWeight = backend.Upload(getWeight("encoder.final_layer_norm.weight"), TensorShape.D1(dim), exact: true);
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

        for (int i = 0; i < Layers.Length; i++)
            Layers[i]?.Dispose();

        _backend.Free(FinalLayerNormWeight);
    }
}
