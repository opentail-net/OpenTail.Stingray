using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// GPU-resident VRAM representation of UMT5-XXL text encoder weights for Wan 2.1/2.2.
/// Uploads all 24 encoder layer projections and normalization vectors to device VRAM once,
/// enabling zero CPU-GPU round-trip execution during prompt encoding.
/// </summary>
public sealed class UMT5GpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public UMT5LayerGpuWeights[] Layers { get; }
    public CoreTensor FinalLayerNormWeight { get; }

    public sealed class UMT5LayerGpuWeights : IDisposable
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
        public CoreTensor WoWeight { get; }

        public UMT5LayerGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int idx, int dim, int ffDim)
        {
            _backend = backend;
            string p = $"blocks.{idx}";

            LayerNorm0Weight = backend.Upload(getWeight($"{p}.norm1.weight"), TensorShape.D1(dim), exact: true);
            QWeight = UploadWeight(backend, getWeight($"{p}.attn.q.weight"), TensorShape.D2(dim, dim));
            KWeight = UploadWeight(backend, getWeight($"{p}.attn.k.weight"), TensorShape.D2(dim, dim));
            VWeight = UploadWeight(backend, getWeight($"{p}.attn.v.weight"), TensorShape.D2(dim, dim));
            OWeight = UploadWeight(backend, getWeight($"{p}.attn.o.weight"), TensorShape.D2(dim, dim));

            LayerNorm1Weight = backend.Upload(getWeight($"{p}.norm2.weight"), TensorShape.D1(dim), exact: true);
            Wi0Weight = UploadWeight(backend, getWeight($"{p}.ffn.gate.0.weight"), TensorShape.D2(ffDim, dim));
            Wi1Weight = UploadWeight(backend, getWeight($"{p}.ffn.fc1.weight"), TensorShape.D2(ffDim, dim));
            WoWeight = UploadWeight(backend, getWeight($"{p}.ffn.fc2.weight"), TensorShape.D2(dim, ffDim));
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
            _backend.Free(WoWeight);
        }
    }

    public UMT5GpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int numLayers = 24, int dim = 4096, int ffDim = 10240)
    {
        _backend = backend;
        Layers = new UMT5LayerGpuWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Layers[i] = new UMT5LayerGpuWeights(backend, getWeight, i, dim, ffDim);
        }
        FinalLayerNormWeight = backend.Upload(getWeight("norm.weight"), TensorShape.D1(dim), exact: true);
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
        foreach (var layer in Layers) layer.Dispose();
        _backend.Free(FinalLayerNormWeight);
    }
}
