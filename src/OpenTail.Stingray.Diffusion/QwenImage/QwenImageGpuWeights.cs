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

        /// <summary>
        /// Chunked image-stream MLP (see <see cref="MlpChunkCount"/>): when non-null, the up projection is stored as
        /// <see cref="MlpChunkCount"/> row slices [hidden x hidden] (with matching bias slices) and the down projection as
        /// <see cref="MlpChunkCount"/> K slices [hidden x hidden], and <see cref="ImgMlpUpWeight"/> / <see cref="ImgMlpDownWeight"/>
        /// are not uploaded. This keeps every activation buffer 3072 wide, which the quantized GEMM handles much faster than the
        /// 12288-wide hidden activation (measured: whole MLP 188.6 ms -> 156.3 ms per block on the Vega 8 iGPU).
        /// </summary>
        public CoreTensor[]? ImgMlpUpChunks { get; }
        public CoreTensor?[]? ImgMlpUpBiasChunks { get; }
        public CoreTensor[]? ImgMlpDownChunks { get; }
        public CoreTensor TxtMlpUpWeight { get; }
        public CoreTensor? TxtMlpUpBias { get; }
        public CoreTensor TxtMlpDownWeight { get; }
        public CoreTensor? TxtMlpDownBias { get; }

        public BlockGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Func<string, float[]?> tryGetWeight, int idx, int d, int headDim, bool forceFp32 = false, RawWeightSource? rawSource = null)
        {
            _backend = backend;
            try
            {
            string p = $"transformer_blocks.{idx}.";

            ImgModWeight = UploadWeight(backend, rawSource, $"{p}img_mod.1.weight", getWeight, TensorShape.D2(6 * d, d), forceFp32);
            ImgModBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mod.1.bias"));
            TxtModWeight = UploadWeight(backend, rawSource, $"{p}txt_mod.1.weight", getWeight, TensorShape.D2(6 * d, d), forceFp32);
            TxtModBias = UploadOptionalBias(backend, tryGetWeight($"{p}txt_mod.1.bias"));

            ImgToQWeight = UploadWeight(backend, rawSource, $"{p}attn.to_q.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            ImgToKWeight = UploadWeight(backend, rawSource, $"{p}attn.to_k.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            ImgToVWeight = UploadWeight(backend, rawSource, $"{p}attn.to_v.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            TxtAddQWeight = UploadWeight(backend, rawSource, $"{p}attn.add_q_proj.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            TxtAddKWeight = UploadWeight(backend, rawSource, $"{p}attn.add_k_proj.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            TxtAddVWeight = UploadWeight(backend, rawSource, $"{p}attn.add_v_proj.weight", getWeight, TensorShape.D2(d, d), forceFp32);

            ImgNormQScale = backend.Upload(getWeight($"{p}attn.norm_q.weight"), TensorShape.D1(headDim), exact: true);
            ImgNormKScale = backend.Upload(getWeight($"{p}attn.norm_k.weight"), TensorShape.D1(headDim), exact: true);
            TxtNormQScale = backend.Upload(getWeight($"{p}attn.norm_added_q.weight"), TensorShape.D1(headDim), exact: true);
            TxtNormKScale = backend.Upload(getWeight($"{p}attn.norm_added_k.weight"), TensorShape.D1(headDim), exact: true);

            ImgToOutWeight = UploadWeight(backend, rawSource, $"{p}attn.to_out.0.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            ImgToOutBias = UploadOptionalBias(backend, tryGetWeight($"{p}attn.to_out.0.bias"));
            TxtToAddOutWeight = UploadWeight(backend, rawSource, $"{p}attn.to_add_out.weight", getWeight, TensorShape.D2(d, d), forceFp32);
            TxtToAddOutBias = UploadOptionalBias(backend, tryGetWeight($"{p}attn.to_add_out.bias"));

            int ffDim = d * 4;
            if (!forceFp32 && ChunkedMlpEnabled
                && TryUploadMlpChunks(backend, rawSource, $"{p}img_mlp.net.0.proj.weight", $"{p}img_mlp.net.2.weight",
                    tryGetWeight($"{p}img_mlp.net.0.proj.bias"), d, ffDim, out var upChunks, out var upBiasChunks, out var downChunks))
            {
                ImgMlpUpChunks = upChunks;
                ImgMlpUpBiasChunks = upBiasChunks;
                ImgMlpDownChunks = downChunks;
                ImgMlpUpWeight = null!;
                ImgMlpDownWeight = null!;
                ImgMlpDownBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mlp.net.2.bias"));
            }
            else
            {
                ImgMlpUpWeight = UploadWeight(backend, rawSource, $"{p}img_mlp.net.0.proj.weight", getWeight, TensorShape.D2(ffDim, d), forceFp32);
                ImgMlpUpBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mlp.net.0.proj.bias"));
                ImgMlpDownWeight = UploadWeight(backend, rawSource, $"{p}img_mlp.net.2.weight", getWeight, TensorShape.D2(d, ffDim), forceFp32);
                ImgMlpDownBias = UploadOptionalBias(backend, tryGetWeight($"{p}img_mlp.net.2.bias"));
            }
            TxtMlpUpWeight = UploadWeight(backend, rawSource, $"{p}txt_mlp.net.0.proj.weight", getWeight, TensorShape.D2(ffDim, d), forceFp32);
            TxtMlpUpBias = UploadOptionalBias(backend, tryGetWeight($"{p}txt_mlp.net.0.proj.bias"));
            TxtMlpDownWeight = UploadWeight(backend, rawSource, $"{p}txt_mlp.net.2.weight", getWeight, TensorShape.D2(d, ffDim), forceFp32);
            TxtMlpDownBias = UploadOptionalBias(backend, tryGetWeight($"{p}txt_mlp.net.2.bias"));
            }
            catch
            {
                // A failed upload (e.g. the device heap is full) must not leak the tensors already
                // uploaded for this block: QwenImageGpuWeights uploads blocks until one fails.
                Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            // Null-tolerant: also called on a partially constructed block.
            foreach (var t in new[] {
                ImgModWeight, ImgModBias, TxtModWeight, TxtModBias,
                ImgToQWeight, ImgToKWeight, ImgToVWeight, TxtAddQWeight, TxtAddKWeight, TxtAddVWeight,
                ImgNormQScale, ImgNormKScale, TxtNormQScale, TxtNormKScale,
                ImgToOutWeight, ImgToOutBias, TxtToAddOutWeight, TxtToAddOutBias,
                ImgMlpUpWeight, ImgMlpUpBias, ImgMlpDownWeight, ImgMlpDownBias,
                TxtMlpUpWeight, TxtMlpUpBias, TxtMlpDownWeight, TxtMlpDownBias })
            {
                if (t is not null) _backend.Free(t);
            }
            if (ImgMlpUpChunks is not null) foreach (var t in ImgMlpUpChunks) _backend.Free(t);
            if (ImgMlpUpBiasChunks is not null) foreach (var t in ImgMlpUpBiasChunks) if (t is not null) _backend.Free(t);
            if (ImgMlpDownChunks is not null) foreach (var t in ImgMlpDownChunks) _backend.Free(t);
        }
    }

    public QwenImageGpuWeights(IComputeBackend backend, Func<string, float[]> getWeight, Func<string, float[]?> tryGetWeight, int numLayers, int hiddenDim, int inChannels, int contextDim, int headDim, int? maxResidentBlocks = null, RawWeightSource? rawSource = null)
    {
        _backend = backend;

        ImgInWeight = UploadWeight(backend, getWeight("img_in.weight"), TensorShape.D2(hiddenDim, inChannels));
        ImgInBias = UploadOptionalBias(backend, tryGetWeight("img_in.bias"));
        TxtInWeight = UploadWeight(backend, getWeight("txt_in.weight"), TensorShape.D2(hiddenDim, contextDim));
        TxtInBias = UploadOptionalBias(backend, tryGetWeight("txt_in.bias"));

        FinalNormLinearWeight = UploadWeight(backend, getWeight("norm_out.linear.weight"), TensorShape.D2(2 * hiddenDim, hiddenDim));
        FinalNormLinearBias = UploadOptionalBias(backend, tryGetWeight("norm_out.linear.bias"));
        FinalProjOutWeight = UploadWeight(backend, getWeight("proj_out.weight"), TensorShape.D2(inChannels, hiddenDim));
        FinalProjOutBias = UploadOptionalBias(backend, tryGetWeight("proj_out.bias"));

        // Partial residency: the full 20.8B-parameter model is ~41 GB at FP16, more than e.g. a
        // shared-memory iGPU's 31.4 GiB host-visible heap. Upload blocks in order until the device
        // refuses one (or maxResidentBlocks is reached), then give back HeadroomBlocks for runtime
        // allocations. The caller runs blocks [ResidentBlockCount, numLayers) on the CPU.
        int cap = maxResidentBlocks is int m ? Math.Clamp(m, 0, numLayers) : numLayers;
        var blocks = new List<BlockGpuWeights>(cap);
        bool hitLimit = false;
        for (int i = 0; i < cap; i++)
        {
            try
            {
                blocks.Add(new BlockGpuWeights(backend, getWeight, tryGetWeight, i, hiddenDim, headDim, rawSource: rawSource));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[QwenImage] GPU heap full after {blocks.Count} blocks ({ex.GetType().Name}); the rest run on CPU.");
                hitLimit = true;
                break;
            }
        }
        if (hitLimit)
        {
            for (int k = 0; k < HeadroomBlocks && blocks.Count > 0; k++)
            {
                blocks[^1].Dispose();
                blocks.RemoveAt(blocks.Count - 1);
            }
        }
        Blocks = blocks.ToArray();

        GC.Collect();
    }

    /// <summary>Blocks freed again after the heap-full point, for workspace/attention scratch
    /// and other runtime allocations.</summary>
    private const int HeadroomBlocks = 2;

    public int ResidentBlockCount => Blocks.Length;

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

    /// <summary>Number of 3072-wide slices the image-stream MLP hidden dimension (4 x hidden) is split into.</summary>
    internal const int MlpChunkCount = 4;

    /// <summary>STINGRAY_QWEN_MLP_CHUNKS=0 keeps the original single 12288-wide MLP (rollback switch).</summary>
    private static readonly bool ChunkedMlpEnabled = Environment.GetEnvironmentVariable("STINGRAY_QWEN_MLP_CHUNKS") != "0";

    private static int QuantBlockBytes(DType t) => t switch
    {
        DType.Q3_K => 110,
        DType.Q4_K => 144,
        DType.Q5_K => 176,
        _ => throw new ArgumentOutOfRangeException(nameof(t)),
    };

    /// <summary>
    /// Uploads one image-stream MLP as <see cref="MlpChunkCount"/> up-projection row slices (contiguous in the GGUF) and
    /// <see cref="MlpChunkCount"/> down-projection K slices (re-gathered per row, because a K slice is not contiguous), all as raw
    /// block-quantized tensors. Returns false, uploading nothing, when the weights are not raw K-quants of the expected shape
    /// (the caller then takes the ordinary single-tensor path).
    /// </summary>
    private static unsafe bool TryUploadMlpChunks(IComputeBackend backend, RawWeightSource? raw, string upName, string downName,
        float[]? upBias, int d, int ffDim, out CoreTensor[]? upChunks, out CoreTensor?[]? upBiasChunks, out CoreTensor[]? downChunks)
    {
        upChunks = null; upBiasChunks = null; downChunks = null;
        if (raw is null || !backend.SupportsQuantizedSgemm || ffDim != MlpChunkCount * d || d % 256 != 0) return false;
        if (!raw(upName, out nint upData, out long upLen, out DType upType, out int upRows, out int upCols)
            || !raw(downName, out nint downData, out long downLen, out DType downType, out int downRows, out int downCols))
            return false;
        if (!QuantizedGpuUpload.IsSupported(upType) || !QuantizedGpuUpload.IsSupported(downType)) return false;
        if (upRows != ffDim || upCols != d || downRows != d || downCols != ffDim) return false;

        long upRowBytes = (long)(d / 256) * QuantBlockBytes(upType);
        long downRowBytes = (long)(ffDim / 256) * QuantBlockBytes(downType);
        long downChunkRowBytes = (long)(d / 256) * QuantBlockBytes(downType);
        if (upLen != upRowBytes * ffDim || downLen != downRowBytes * d) return false;

        var ups = new List<CoreTensor>(MlpChunkCount);
        var biases = new List<CoreTensor?>(MlpChunkCount);
        var downs = new List<CoreTensor>(MlpChunkCount);
        try
        {
            for (int c = 0; c < MlpChunkCount; c++)
            {
                // Up projection rows [c*d, (c+1)*d) are one contiguous byte range.
                ups.Add(backend.UploadRaw(
                    new ReadOnlySpan<byte>((byte*)upData + c * d * upRowBytes, checked((int)(d * upRowBytes))),
                    TensorShape.D2(d, d), upType));
                biases.Add(upBias is null ? null : UploadOptionalBias(backend, upBias.AsSpan(c * d, d).ToArray()));

                // Down projection columns [c*d, (c+1)*d) of every row: gather that block range row by row.
                var tmp = new byte[checked((int)(d * downChunkRowBytes))];
                for (int r = 0; r < d; r++)
                    new ReadOnlySpan<byte>((byte*)downData + r * downRowBytes + c * downChunkRowBytes, (int)downChunkRowBytes)
                        .CopyTo(tmp.AsSpan((int)(r * downChunkRowBytes), (int)downChunkRowBytes));
                downs.Add(backend.UploadRaw(tmp, TensorShape.D2(d, d), downType));
            }
        }
        catch
        {
            foreach (var t in ups) backend.Free(t);
            foreach (var t in biases) if (t is not null) backend.Free(t);
            foreach (var t in downs) backend.Free(t);
            throw;
        }
        upChunks = ups.ToArray();
        upBiasChunks = biases.ToArray();
        downChunks = downs.ToArray();
        return true;
    }

    /// <summary>
    /// Q3_K/Q4_K weights are uploaded as their raw GGUF blocks when the backend's Sgemm can consume
    /// them (Vulkan: dequant-in-shader SgemmQ3K/SgemmQ4K + MatVecDq*): ~3.4-4.5 bits per weight
    /// instead of 16, so the whole 20.8B-parameter model is ~9 GB on the GPU rather than ~41 GB,
    /// and no host-side dequantize+convert is needed. Anything else (or
    /// STINGRAY_QWEN_GPU_FP16=1) takes the FP16/BF16/FP32 path below.
    /// </summary>
    private static unsafe CoreTensor UploadWeight(IComputeBackend backend, RawWeightSource? raw, string name,
        Func<string, float[]> getWeight, TensorShape shape, bool forceFp32 = false)
    {
        if (!forceFp32 && QuantizedGpuUpload.TryUpload(backend, raw, name, shape, "STINGRAY_QWEN_GPU_FP16") is { } q)
            return q;
        return UploadWeight(backend, getWeight(name), shape, forceFp32);
    }

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
