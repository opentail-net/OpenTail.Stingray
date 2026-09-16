using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// GPU-resident VRAM representation of LTX-Video-2B DiT weights.
/// Holds all 28 transformer blocks, linear projections, AdaLN tables, and normalizations
/// in device VRAM, eliminating CPU-GPU transfer bottlenecks during sampling loops.
/// </summary>
public sealed class LtxVideoGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor PatchifyProjWeight { get; }
    public CoreTensor? PatchifyProjBias { get; }

    public CoreTensor CapProj1Weight { get; }
    public CoreTensor? CapProj1Bias { get; }
    public CoreTensor CapProj2Weight { get; }
    public CoreTensor? CapProj2Bias { get; }

    public CoreTensor TimeEmb1Weight { get; }
    public CoreTensor? TimeEmb1Bias { get; }
    public CoreTensor TimeEmb2Weight { get; }
    public CoreTensor? TimeEmb2Bias { get; }
    public CoreTensor TimeProjWeight { get; }
    public CoreTensor? TimeProjBias { get; }

    public LtxBlockGpuWeights[] Blocks { get; }

    public CoreTensor TopScaleShiftTable { get; }
    public CoreTensor ProjOutWeight { get; }
    public CoreTensor? ProjOutBias { get; }

    public sealed class LtxBlockGpuWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor ScaleShiftTable { get; }

        public CoreTensor Attn1QWeight { get; }
        public CoreTensor? Attn1QBias { get; }
        public CoreTensor Attn1KWeight { get; }
        public CoreTensor? Attn1KBias { get; }
        public CoreTensor Attn1VWeight { get; }
        public CoreTensor? Attn1VBias { get; }
        public CoreTensor Attn1QNorm { get; }
        public CoreTensor Attn1KNorm { get; }
        public CoreTensor Attn1OutWeight { get; }
        public CoreTensor? Attn1OutBias { get; }

        public CoreTensor Attn2QWeight { get; }
        public CoreTensor? Attn2QBias { get; }
        public CoreTensor Attn2KWeight { get; }
        public CoreTensor? Attn2KBias { get; }
        public CoreTensor Attn2VWeight { get; }
        public CoreTensor? Attn2VBias { get; }
        public CoreTensor Attn2QNorm { get; }
        public CoreTensor Attn2KNorm { get; }
        public CoreTensor Attn2OutWeight { get; }
        public CoreTensor? Attn2OutBias { get; }

        public CoreTensor Ffn0Weight { get; }
        public CoreTensor? Ffn0Bias { get; }
        public CoreTensor Ffn2Weight { get; }
        public CoreTensor? Ffn2Bias { get; }

        public LtxBlockGpuWeights(
            IComputeBackend backend,
            Func<string, float[]> getWeight,
            Func<string, float[]?> tryGetWeight,
            int layerIdx,
            int d)
        {
            _backend = backend;
            string prefix = $"transformer_blocks.{layerIdx}";

            ScaleShiftTable = backend.Upload(getWeight($"{prefix}.scale_shift_table"), TensorShape.D1(6 * d), exact: true);

            // Self-Attention (attn1)
            Attn1QWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn1.to_q.weight"), d, d);
            var b = tryGetWeight($"{prefix}.attn1.to_q.bias");
            if (b is not null) Attn1QBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            Attn1KWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn1.to_k.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn1.to_k.bias");
            if (b is not null) Attn1KBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            Attn1VWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn1.to_v.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn1.to_v.bias");
            if (b is not null) Attn1VBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            Attn1QNorm = backend.Upload(getWeight($"{prefix}.attn1.q_norm.weight"), TensorShape.D1(d), exact: true);
            Attn1KNorm = backend.Upload(getWeight($"{prefix}.attn1.k_norm.weight"), TensorShape.D1(d), exact: true);

            Attn1OutWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn1.to_out.0.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn1.to_out.0.bias");
            if (b is not null) Attn1OutBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            // Cross-Attention (attn2)
            Attn2QWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn2.to_q.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn2.to_q.bias");
            if (b is not null) Attn2QBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            Attn2KWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn2.to_k.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn2.to_k.bias");
            if (b is not null) Attn2KBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            Attn2VWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn2.to_v.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn2.to_v.bias");
            if (b is not null) Attn2VBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            Attn2QNorm = backend.Upload(getWeight($"{prefix}.attn2.q_norm.weight"), TensorShape.D1(d), exact: true);
            Attn2KNorm = backend.Upload(getWeight($"{prefix}.attn2.k_norm.weight"), TensorShape.D1(d), exact: true);

            Attn2OutWeight = UploadTransposedLinear(backend, getWeight($"{prefix}.attn2.to_out.0.weight"), d, d);
            b = tryGetWeight($"{prefix}.attn2.to_out.0.bias");
            if (b is not null) Attn2OutBias = backend.Upload(b, TensorShape.D1(d), exact: true);

            // FeedForward
            int ffnDim = d * 4;
            Ffn0Weight = UploadTransposedLinear(backend, getWeight($"{prefix}.ff.net.0.proj.weight"), ffnDim, d);
            b = tryGetWeight($"{prefix}.ff.net.0.proj.bias");
            if (b is not null) Ffn0Bias = backend.Upload(b, TensorShape.D1(ffnDim), exact: true);

            Ffn2Weight = UploadTransposedLinear(backend, getWeight($"{prefix}.ff.net.2.weight"), d, ffnDim);
            b = tryGetWeight($"{prefix}.ff.net.2.bias");
            if (b is not null) Ffn2Bias = backend.Upload(b, TensorShape.D1(d), exact: true);
        }

        public void Dispose()
        {
            _backend.Free(ScaleShiftTable);
            _backend.Free(Attn1QWeight);
            if (Attn1QBias is not null) _backend.Free(Attn1QBias);
            _backend.Free(Attn1KWeight);
            if (Attn1KBias is not null) _backend.Free(Attn1KBias);
            _backend.Free(Attn1VWeight);
            if (Attn1VBias is not null) _backend.Free(Attn1VBias);
            _backend.Free(Attn1QNorm);
            _backend.Free(Attn1KNorm);
            _backend.Free(Attn1OutWeight);
            if (Attn1OutBias is not null) _backend.Free(Attn1OutBias);

            _backend.Free(Attn2QWeight);
            if (Attn2QBias is not null) _backend.Free(Attn2QBias);
            _backend.Free(Attn2KWeight);
            if (Attn2KBias is not null) _backend.Free(Attn2KBias);
            _backend.Free(Attn2VWeight);
            if (Attn2VBias is not null) _backend.Free(Attn2VBias);
            _backend.Free(Attn2QNorm);
            _backend.Free(Attn2KNorm);
            _backend.Free(Attn2OutWeight);
            if (Attn2OutBias is not null) _backend.Free(Attn2OutBias);

            _backend.Free(Ffn0Weight);
            if (Ffn0Bias is not null) _backend.Free(Ffn0Bias);
            _backend.Free(Ffn2Weight);
            if (Ffn2Bias is not null) _backend.Free(Ffn2Bias);
        }
    }

    public LtxVideoGpuWeights(
        IComputeBackend backend,
        Func<string, float[]> getWeight,
        Func<string, float[]?> tryGetWeight,
        int inChannels,
        int hiddenSize,
        int captionChannels,
        int numLayers)
    {
        _backend = backend;
        int d = hiddenSize;

        PatchifyProjWeight = UploadTransposedLinear(backend, getWeight("patchify_proj.weight"), d, inChannels);
        var pb = tryGetWeight("patchify_proj.bias");
        if (pb is not null) PatchifyProjBias = backend.Upload(pb, TensorShape.D1(d), exact: true);

        CapProj1Weight = UploadTransposedLinear(backend, getWeight("caption_projection.linear_1.weight"), d, captionChannels);
        var cb1 = tryGetWeight("caption_projection.linear_1.bias");
        if (cb1 is not null) CapProj1Bias = backend.Upload(cb1, TensorShape.D1(d), exact: true);

        CapProj2Weight = UploadTransposedLinear(backend, getWeight("caption_projection.linear_2.weight"), d, d);
        var cb2 = tryGetWeight("caption_projection.linear_2.bias");
        if (cb2 is not null) CapProj2Bias = backend.Upload(cb2, TensorShape.D1(d), exact: true);

        TimeEmb1Weight = UploadTransposedLinear(backend, getWeight("adaln_single.emb.timestep_embedder.linear_1.weight"), d, 256);
        var tb1 = tryGetWeight("adaln_single.emb.timestep_embedder.linear_1.bias");
        if (tb1 is not null) TimeEmb1Bias = backend.Upload(tb1, TensorShape.D1(d), exact: true);

        TimeEmb2Weight = UploadTransposedLinear(backend, getWeight("adaln_single.emb.timestep_embedder.linear_2.weight"), d, d);
        var tb2 = tryGetWeight("adaln_single.emb.timestep_embedder.linear_2.bias");
        if (tb2 is not null) TimeEmb2Bias = backend.Upload(tb2, TensorShape.D1(d), exact: true);

        TimeProjWeight = UploadTransposedLinear(backend, getWeight("adaln_single.linear.weight"), d * 6, d);
        var tpb = tryGetWeight("adaln_single.linear.bias");
        if (tpb is not null) TimeProjBias = backend.Upload(tpb, TensorShape.D1(d * 6), exact: true);

        Blocks = new LtxBlockGpuWeights[numLayers];
        for (int l = 0; l < numLayers; l++)
        {
            Blocks[l] = new LtxBlockGpuWeights(backend, getWeight, tryGetWeight, l, d);
        }

        TopScaleShiftTable = backend.Upload(getWeight("scale_shift_table"), TensorShape.D1(2 * d), exact: true);

        ProjOutWeight = UploadTransposedLinear(backend, getWeight("proj_out.weight"), inChannels, d);
        var pob = tryGetWeight("proj_out.bias");
        if (pob is not null) ProjOutBias = backend.Upload(pob, TensorShape.D1(inChannels), exact: true);

        GC.Collect();
    }

    /// <summary>
    /// Uploads PyTorch Safetensors weight [outDim, inDim] directly to device VRAM.
    /// In Vulkan matrix multiplication Sgemm(C, A, B, M, K, N), B is loaded as [N, K] = [outDim, inDim].
    /// </summary>
    public static CoreTensor UploadTransposedLinear(IComputeBackend backend, float[] rawWeight, int outDim, int inDim)
    {
        if (backend.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[rawWeight.Length];
            TensorPrimitives.ConvertToHalf(rawWeight, half);
            return backend.UploadHalf(half, TensorShape.D2(outDim, inDim));
        }
        if (backend.BestSgemmPrecision == SgemmPrecision.Bf16)
        {
            var bf16 = new ushort[rawWeight.Length];
            for (int i = 0; i < rawWeight.Length; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits(rawWeight[i]);
                bf16[i] = (ushort)(bits >> 16);
            }
            return backend.UploadBf16(bf16, TensorShape.D2(outDim, inDim));
        }
        return backend.Upload(rawWeight, TensorShape.D2(outDim, inDim), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(PatchifyProjWeight);
        if (PatchifyProjBias is not null) _backend.Free(PatchifyProjBias);
        _backend.Free(CapProj1Weight);
        if (CapProj1Bias is not null) _backend.Free(CapProj1Bias);
        _backend.Free(CapProj2Weight);
        if (CapProj2Bias is not null) _backend.Free(CapProj2Bias);
        _backend.Free(TimeEmb1Weight);
        if (TimeEmb1Bias is not null) _backend.Free(TimeEmb1Bias);
        _backend.Free(TimeEmb2Weight);
        if (TimeEmb2Bias is not null) _backend.Free(TimeEmb2Bias);
        _backend.Free(TimeProjWeight);
        if (TimeProjBias is not null) _backend.Free(TimeProjBias);

        for (int i = 0; i < Blocks.Length; i++)
            Blocks[i]?.Dispose();

        _backend.Free(TopScaleShiftTable);
        _backend.Free(ProjOutWeight);
        if (ProjOutBias is not null) _backend.Free(ProjOutBias);
    }
}
