using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// GPU-resident VRAM representation of Z-Image-Turbo's 30 main `layers.N` transformer blocks
/// (docs/075) — the dominant-cost blocks; the 2+2 `context_refiner`/`noise_refiner` blocks are a
/// real, separate, smaller-scope follow-up (not covered here, see docs/075's own scoping note).
/// Mirrors `FluxGpuWeights`/`WanGpuWeights`'s pattern: upload every projection once, resident in
/// VRAM, reused across every denoising step and every block invocation.
/// </summary>
public sealed class ZImageGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public sealed class BlockWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor AdaLnModW { get; }
        public CoreTensor AdaLnModB { get; }
        public CoreTensor AttnNorm1W { get; }
        public CoreTensor AttnNorm2W { get; }
        /// <summary>
        /// Host-side copy of `attention_norm1.weight`/`ffn_norm1.weight` (the learned per-channel
        /// RMSNorm gamma applied BEFORE the AdaLN modulation scale on the CPU path, via
        /// `DiffusionOps.RmsNorm(x, ..., normW1, ...)` then a separate multiply by scaleMsa).
        /// `AdaLNModulate`'s shader is an UNWEIGHTED RMSNorm -- it has no way to apply a learned
        /// gamma itself, only an externally-supplied per-token scale+shift. Kept host-side (not
        /// just GPU-resident) so `ApplyBlockGpu` can fold it into the modulation scale vector
        /// on the CPU before uploading, since both are per-channel and RMSNorm's normalization
        /// commutes with elementwise scaling: RMSNorm(x)*normW1*scaleMsa == RMSNorm(x)*(normW1*scaleMsa).
        /// </summary>
        public float[] AttnNorm1WHost { get; }
        public float[] FfnNorm1WHost { get; }
        public CoreTensor QkvW { get; }
        public CoreTensor QNormW { get; }
        public CoreTensor KNormW { get; }
        public CoreTensor OutW { get; }
        public CoreTensor FfnNorm1W { get; }
        public CoreTensor FfnNorm2W { get; }
        public CoreTensor W1 { get; }
        public CoreTensor W3 { get; }
        public CoreTensor W2 { get; }

        public BlockWeights(IComputeBackend backend, Func<string, float[]> getWeight, string prefix, int dim, int headDim, int ffnHidden, int adalnEmbedDim)
        {
            _backend = backend;

            AdaLnModW = UploadWeight(backend, getWeight($"{prefix}.adaLN_modulation.0.weight"), TensorShape.D2(4 * dim, adalnEmbedDim));
            AdaLnModB = backend.Upload(getWeight($"{prefix}.adaLN_modulation.0.bias"), TensorShape.D1(4 * dim), exact: true);

            AttnNorm1WHost = getWeight($"{prefix}.attention_norm1.weight");
            AttnNorm1W = backend.Upload(AttnNorm1WHost, TensorShape.D1(dim), exact: true);
            AttnNorm2W = backend.Upload(getWeight($"{prefix}.attention_norm2.weight"), TensorShape.D1(dim), exact: true);

            QkvW = UploadWeight(backend, getWeight($"{prefix}.attention.qkv.weight"), TensorShape.D2(3 * dim, dim));
            QNormW = backend.Upload(getWeight($"{prefix}.attention.q_norm.weight"), TensorShape.D1(headDim), exact: true);
            KNormW = backend.Upload(getWeight($"{prefix}.attention.k_norm.weight"), TensorShape.D1(headDim), exact: true);
            OutW = UploadWeight(backend, getWeight($"{prefix}.attention.out.weight"), TensorShape.D2(dim, dim));

            FfnNorm1WHost = getWeight($"{prefix}.ffn_norm1.weight");
            FfnNorm1W = backend.Upload(FfnNorm1WHost, TensorShape.D1(dim), exact: true);
            FfnNorm2W = backend.Upload(getWeight($"{prefix}.ffn_norm2.weight"), TensorShape.D1(dim), exact: true);

            W1 = UploadWeight(backend, getWeight($"{prefix}.feed_forward.w1.weight"), TensorShape.D2(ffnHidden, dim));
            W3 = UploadWeight(backend, getWeight($"{prefix}.feed_forward.w3.weight"), TensorShape.D2(ffnHidden, dim));
            W2 = UploadWeight(backend, getWeight($"{prefix}.feed_forward.w2.weight"), TensorShape.D2(dim, ffnHidden));
        }

        public void Dispose()
        {
            _backend.Free(AdaLnModW); _backend.Free(AdaLnModB);
            _backend.Free(AttnNorm1W); _backend.Free(AttnNorm2W);
            _backend.Free(QkvW); _backend.Free(QNormW); _backend.Free(KNormW); _backend.Free(OutW);
            _backend.Free(FfnNorm1W); _backend.Free(FfnNorm2W);
            _backend.Free(W1); _backend.Free(W3); _backend.Free(W2);
        }
    }

    public BlockWeights[] Layers { get; }

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

    public ZImageGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, int numLayers, int dim, int headDim, int ffnHidden, int adalnEmbedDim)
    {
        _backend = backend;
        Layers = new BlockWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
            Layers[i] = new BlockWeights(backend, getWeight, $"layers.{i}", dim, headDim, ffnHidden, adalnEmbedDim);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var l in Layers) l.Dispose();
    }
}
