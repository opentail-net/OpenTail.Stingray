using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// GPU-resident VRAM representation of Wan 2.1 DiT weights.
/// Uploads all transformer projections, modulation parameters, and norm vectors to GPU memory once,
/// enabling zero CPU-GPU round-trip execution during the 20 Euler denoising steps.
/// </summary>
public sealed class WanGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor PatchEmbedding { get; }
    public CoreTensor TimeEmbed0 { get; }
    public CoreTensor TimeEmbed2 { get; }
    public CoreTensor TimeProj1 { get; }
    public CoreTensor TextEmbed0 { get; }
    public CoreTensor TextEmbed2 { get; }

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

    public sealed class BlockWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public float[] HostModulation { get; }
        public CoreTensor Modulation { get; }
        public CoreTensor Norm3Weight { get; }
        public CoreTensor Norm3Bias { get; }
        public CoreTensor SelfAttnQkv { get; }
        public CoreTensor SelfAttnQ { get; }
        public CoreTensor SelfAttnK { get; }
        public CoreTensor SelfAttnV { get; }
        public CoreTensor SelfAttnO { get; }
        public CoreTensor? SelfAttnNormQ { get; }
        public CoreTensor? SelfAttnNormK { get; }
        public CoreTensor CrossAttnQ { get; }
        public CoreTensor CrossAttnK { get; }
        public CoreTensor CrossAttnV { get; }
        public CoreTensor CrossAttnO { get; }
        public CoreTensor? CrossAttnNormQ { get; }
        public CoreTensor? CrossAttnNormK { get; }
        public CoreTensor Ffn0 { get; }
        public CoreTensor Ffn2 { get; }

        public BlockWeights(IComputeBackend backend, IWeightLoader weights, string prefix, int dim, int ffnDim)
        {
            _backend = backend;

            HostModulation = weights.ReadF32($"{prefix}.modulation");
            Modulation = backend.Upload(HostModulation, TensorShape.D1(dim * 6), exact: true);
            Norm3Weight = backend.Upload(weights.ReadF32($"{prefix}.norm3.weight"), TensorShape.D1(dim), exact: true);
            Norm3Bias = backend.Upload(weights.ReadF32($"{prefix}.norm3.bias"), TensorShape.D1(dim), exact: true);

            var qW = weights.ReadF32($"{prefix}.self_attn.q.weight");
            var kW = weights.ReadF32($"{prefix}.self_attn.k.weight");
            var vW = weights.ReadF32($"{prefix}.self_attn.v.weight");
            var qkvW = new float[dim * 3 * dim];
            qW.CopyTo(qkvW.AsSpan(0, dim * dim));
            kW.CopyTo(qkvW.AsSpan(dim * dim, dim * dim));
            vW.CopyTo(qkvW.AsSpan(dim * dim * 2, dim * dim));
            SelfAttnQkv = UploadWeight(backend, qkvW, TensorShape.D2(dim * 3, dim));

            SelfAttnQ = UploadWeight(backend, qW, TensorShape.D2(dim, dim));
            SelfAttnK = UploadWeight(backend, kW, TensorShape.D2(dim, dim));
            SelfAttnV = UploadWeight(backend, vW, TensorShape.D2(dim, dim));
            SelfAttnO = UploadWeight(backend, weights.ReadF32($"{prefix}.self_attn.o.weight"), TensorShape.D2(dim, dim));

            string normQKey = $"{prefix}.self_attn.norm_q.weight";
            if (weights.Contains(normQKey))
                SelfAttnNormQ = backend.Upload(weights.ReadF32(normQKey), TensorShape.D1(dim), exact: true);

            string normKKey = $"{prefix}.self_attn.norm_k.weight";
            if (weights.Contains(normKKey))
                SelfAttnNormK = backend.Upload(weights.ReadF32(normKKey), TensorShape.D1(dim), exact: true);

            CrossAttnQ = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.q.weight"), TensorShape.D2(dim, dim));
            CrossAttnK = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.k.weight"), TensorShape.D2(dim, dim));
            CrossAttnV = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.v.weight"), TensorShape.D2(dim, dim));
            CrossAttnO = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.o.weight"), TensorShape.D2(dim, dim));

            string crossNormQKey = $"{prefix}.cross_attn.norm_q.weight";
            if (weights.Contains(crossNormQKey))
                CrossAttnNormQ = backend.Upload(weights.ReadF32(crossNormQKey), TensorShape.D1(dim), exact: true);

            string crossNormKKey = $"{prefix}.cross_attn.norm_k.weight";
            if (weights.Contains(crossNormKKey))
                CrossAttnNormK = backend.Upload(weights.ReadF32(crossNormKKey), TensorShape.D1(dim), exact: true);

            Ffn0 = UploadWeight(backend, weights.ReadF32($"{prefix}.ffn.0.weight"), TensorShape.D2(ffnDim, dim));
            Ffn2 = UploadWeight(backend, weights.ReadF32($"{prefix}.ffn.2.weight"), TensorShape.D2(dim, ffnDim));
        }

        public void Dispose()
        {
            _backend.Free(Modulation);
            _backend.Free(Norm3Weight);
            _backend.Free(Norm3Bias);
            _backend.Free(SelfAttnQkv);
            _backend.Free(SelfAttnQ);
            _backend.Free(SelfAttnK);
            _backend.Free(SelfAttnV);
            _backend.Free(SelfAttnO);
            if (SelfAttnNormQ is not null) _backend.Free(SelfAttnNormQ);
            if (SelfAttnNormK is not null) _backend.Free(SelfAttnNormK);
            _backend.Free(CrossAttnQ);
            _backend.Free(CrossAttnK);
            _backend.Free(CrossAttnV);
            _backend.Free(CrossAttnO);
            if (CrossAttnNormQ is not null) _backend.Free(CrossAttnNormQ);
            if (CrossAttnNormK is not null) _backend.Free(CrossAttnNormK);
            _backend.Free(Ffn0);
            _backend.Free(Ffn2);
        }
    }

    public BlockWeights[] Blocks { get; }
    public float[] HostHeadModulation { get; }
    public CoreTensor HeadModulation { get; }
    public CoreTensor HeadWeight { get; }

    public WanGpuWeights(IComputeBackend backend, IWeightLoader weights, string prefix, int numLayers, int dim, int ffnDim)
    {
        _backend = backend;

        PatchEmbedding = UploadWeight(backend, weights.ReadF32($"{prefix}patch_embedding.weight"), TensorShape.D2(dim, WanModel.InChannels));
        TimeEmbed0 = UploadWeight(backend, weights.ReadF32($"{prefix}time_embedding.0.weight"), TensorShape.D2(dim, 256));
        TimeEmbed2 = UploadWeight(backend, weights.ReadF32($"{prefix}time_embedding.2.weight"), TensorShape.D2(dim, dim));
        TimeProj1 = UploadWeight(backend, weights.ReadF32($"{prefix}time_projection.1.weight"), TensorShape.D2(dim * 6, dim));
        TextEmbed0 = UploadWeight(backend, weights.ReadF32($"{prefix}text_embedding.0.weight"), TensorShape.D2(dim, WanModel.TextDim));
        TextEmbed2 = UploadWeight(backend, weights.ReadF32($"{prefix}text_embedding.2.weight"), TensorShape.D2(dim, dim));

        Blocks = new BlockWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Blocks[i] = new BlockWeights(backend, weights, $"{prefix}blocks.{i}", dim, ffnDim);
        }

        HostHeadModulation = weights.ReadF32($"{prefix}head.modulation");
        HeadModulation = backend.Upload(HostHeadModulation, TensorShape.D1(dim * 2), exact: true);
        HeadWeight = UploadWeight(backend, weights.ReadF32($"{prefix}head.head.weight"), TensorShape.D2(WanModel.InChannels, dim));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(PatchEmbedding);
        _backend.Free(TimeEmbed0);
        _backend.Free(TimeEmbed2);
        _backend.Free(TimeProj1);
        _backend.Free(TextEmbed0);
        _backend.Free(TextEmbed2);

        foreach (var block in Blocks)
        {
            block.Dispose();
        }

        _backend.Free(HeadModulation);
        _backend.Free(HeadWeight);
    }
}
