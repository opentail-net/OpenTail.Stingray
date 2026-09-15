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
    public CoreTensor? PatchEmbeddingBias { get; }
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
        public CoreTensor? SelfAttnQkvBias { get; }
        public CoreTensor SelfAttnQ { get; }
        public CoreTensor SelfAttnK { get; }
        public CoreTensor SelfAttnV { get; }
        public CoreTensor SelfAttnO { get; }
        public CoreTensor? SelfAttnOBias { get; }
        public CoreTensor? SelfAttnNormQ { get; }
        public CoreTensor? SelfAttnNormK { get; }
        public CoreTensor CrossAttnQ { get; }
        public CoreTensor? CrossAttnQBias { get; }
        public CoreTensor CrossAttnK { get; }
        public CoreTensor? CrossAttnKBias { get; }
        public CoreTensor CrossAttnV { get; }
        public CoreTensor? CrossAttnVBias { get; }
        public CoreTensor CrossAttnO { get; }
        public CoreTensor? CrossAttnOBias { get; }
        public CoreTensor? CrossAttnNormQ { get; }
        public CoreTensor? CrossAttnNormK { get; }
        public CoreTensor Ffn0 { get; }
        public CoreTensor? Ffn0Bias { get; }
        public CoreTensor Ffn2 { get; }
        public CoreTensor? Ffn2Bias { get; }

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

            string qBKey = $"{prefix}.self_attn.q.bias";
            string kBKey = $"{prefix}.self_attn.k.bias";
            string vBKey = $"{prefix}.self_attn.v.bias";
            if (weights.Contains(qBKey) && weights.Contains(kBKey) && weights.Contains(vBKey))
            {
                var qB = weights.ReadF32(qBKey);
                var kB = weights.ReadF32(kBKey);
                var vB = weights.ReadF32(vBKey);
                var qkvB = new float[dim * 3];
                qB.CopyTo(qkvB.AsSpan(0, dim));
                kB.CopyTo(qkvB.AsSpan(dim, dim));
                vB.CopyTo(qkvB.AsSpan(dim * 2, dim));
                SelfAttnQkvBias = backend.Upload(qkvB, TensorShape.D1(dim * 3), exact: true);
            }

            SelfAttnQ = UploadWeight(backend, qW, TensorShape.D2(dim, dim));
            SelfAttnK = UploadWeight(backend, kW, TensorShape.D2(dim, dim));
            SelfAttnV = UploadWeight(backend, vW, TensorShape.D2(dim, dim));
            SelfAttnO = UploadWeight(backend, weights.ReadF32($"{prefix}.self_attn.o.weight"), TensorShape.D2(dim, dim));

            string selfOBKey = $"{prefix}.self_attn.o.bias";
            if (weights.Contains(selfOBKey))
                SelfAttnOBias = backend.Upload(weights.ReadF32(selfOBKey), TensorShape.D1(dim), exact: true);

            string normQKey = $"{prefix}.self_attn.norm_q.weight";
            if (weights.Contains(normQKey))
                SelfAttnNormQ = backend.Upload(weights.ReadF32(normQKey), TensorShape.D1(dim), exact: true);

            string normKKey = $"{prefix}.self_attn.norm_k.weight";
            if (weights.Contains(normKKey))
                SelfAttnNormK = backend.Upload(weights.ReadF32(normKKey), TensorShape.D1(dim), exact: true);

            CrossAttnQ = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.q.weight"), TensorShape.D2(dim, dim));
            string crossQBKey = $"{prefix}.cross_attn.q.bias";
            if (weights.Contains(crossQBKey))
                CrossAttnQBias = backend.Upload(weights.ReadF32(crossQBKey), TensorShape.D1(dim), exact: true);

            CrossAttnK = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.k.weight"), TensorShape.D2(dim, dim));
            string crossKBKey = $"{prefix}.cross_attn.k.bias";
            if (weights.Contains(crossKBKey))
                CrossAttnKBias = backend.Upload(weights.ReadF32(crossKBKey), TensorShape.D1(dim), exact: true);

            CrossAttnV = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.v.weight"), TensorShape.D2(dim, dim));
            string crossVBKey = $"{prefix}.cross_attn.v.bias";
            if (weights.Contains(crossVBKey))
                CrossAttnVBias = backend.Upload(weights.ReadF32(crossVBKey), TensorShape.D1(dim), exact: true);

            CrossAttnO = UploadWeight(backend, weights.ReadF32($"{prefix}.cross_attn.o.weight"), TensorShape.D2(dim, dim));
            string crossOBKey = $"{prefix}.cross_attn.o.bias";
            if (weights.Contains(crossOBKey))
                CrossAttnOBias = backend.Upload(weights.ReadF32(crossOBKey), TensorShape.D1(dim), exact: true);

            string crossNormQKey = $"{prefix}.cross_attn.norm_q.weight";
            if (weights.Contains(crossNormQKey))
                CrossAttnNormQ = backend.Upload(weights.ReadF32(crossNormQKey), TensorShape.D1(dim), exact: true);

            string crossNormKKey = $"{prefix}.cross_attn.norm_k.weight";
            if (weights.Contains(crossNormKKey))
                CrossAttnNormK = backend.Upload(weights.ReadF32(crossNormKKey), TensorShape.D1(dim), exact: true);

            Ffn0 = UploadWeight(backend, weights.ReadF32($"{prefix}.ffn.0.weight"), TensorShape.D2(ffnDim, dim));
            string ffn0BKey = $"{prefix}.ffn.0.bias";
            if (weights.Contains(ffn0BKey))
                Ffn0Bias = backend.Upload(weights.ReadF32(ffn0BKey), TensorShape.D1(ffnDim), exact: true);

            Ffn2 = UploadWeight(backend, weights.ReadF32($"{prefix}.ffn.2.weight"), TensorShape.D2(dim, ffnDim));
            string ffn2BKey = $"{prefix}.ffn.2.bias";
            if (weights.Contains(ffn2BKey))
                Ffn2Bias = backend.Upload(weights.ReadF32(ffn2BKey), TensorShape.D1(dim), exact: true);
        }

        public void Dispose()
        {
            _backend.Free(Modulation);
            _backend.Free(Norm3Weight);
            _backend.Free(Norm3Bias);
            _backend.Free(SelfAttnQkv);
            if (SelfAttnQkvBias is not null) _backend.Free(SelfAttnQkvBias);
            _backend.Free(SelfAttnQ);
            _backend.Free(SelfAttnK);
            _backend.Free(SelfAttnV);
            _backend.Free(SelfAttnO);
            if (SelfAttnOBias is not null) _backend.Free(SelfAttnOBias);
            if (SelfAttnNormQ is not null) _backend.Free(SelfAttnNormQ);
            if (SelfAttnNormK is not null) _backend.Free(SelfAttnNormK);
            _backend.Free(CrossAttnQ);
            if (CrossAttnQBias is not null) _backend.Free(CrossAttnQBias);
            _backend.Free(CrossAttnK);
            if (CrossAttnKBias is not null) _backend.Free(CrossAttnKBias);
            _backend.Free(CrossAttnV);
            if (CrossAttnVBias is not null) _backend.Free(CrossAttnVBias);
            _backend.Free(CrossAttnO);
            if (CrossAttnOBias is not null) _backend.Free(CrossAttnOBias);
            if (CrossAttnNormQ is not null) _backend.Free(CrossAttnNormQ);
            if (CrossAttnNormK is not null) _backend.Free(CrossAttnNormK);
            _backend.Free(Ffn0);
            if (Ffn0Bias is not null) _backend.Free(Ffn0Bias);
            _backend.Free(Ffn2);
            if (Ffn2Bias is not null) _backend.Free(Ffn2Bias);
        }
    }

    public BlockWeights[] Blocks { get; }
    public float[] HostHeadModulation { get; }
    public CoreTensor HeadModulation { get; }
    public CoreTensor HeadWeight { get; }
    public CoreTensor? HeadBias { get; }

    public WanGpuWeights(IComputeBackend backend, IWeightLoader weights, string prefix, int numLayers, int dim, int ffnDim)
    {
        _backend = backend;

        PatchEmbedding = UploadWeight(backend, weights.ReadF32($"{prefix}patch_embedding.weight"), TensorShape.D2(dim, WanModel.InChannels));
        string patchBiasKey = $"{prefix}patch_embedding.bias";
        if (weights.Contains(patchBiasKey))
            PatchEmbeddingBias = backend.Upload(weights.ReadF32(patchBiasKey), TensorShape.D1(dim), exact: true);

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
        string headBKey = $"{prefix}head.head.bias";
        if (weights.Contains(headBKey))
            HeadBias = backend.Upload(weights.ReadF32(headBKey), TensorShape.D1(WanModel.InChannels), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(PatchEmbedding);
        if (PatchEmbeddingBias is not null) _backend.Free(PatchEmbeddingBias);
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
        if (HeadBias is not null) _backend.Free(HeadBias);
    }
}
