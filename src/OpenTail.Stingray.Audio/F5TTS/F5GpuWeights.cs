using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// GPU-resident VRAM representation of F5-TTS's 22 DiT transformer blocks + final norm/proj layer
/// -- the same playbook as <c>WanGpuWeights</c>/<c>FluxGpuWeights</c> (docs/069/072): upload every
/// projection + bias once, reused across every ODE step and every block invocation, eliminating
/// the per-op weight re-dequant this model's existing <see cref="F5Kernels.LinearGpuQ8_0"/> path
/// otherwise avoids anyway (its own weight cache already persists) but never batches around.
/// Input/text/time embedding stay CPU-side deliberately -- they run ONCE per generation (already
/// cached/precomputed outside the ODE loop by <c>F5TtsPipeline</c>, unlike the 22-block loop which
/// runs every single step), so their relative cost is small; the real win here is the per-block
/// loop, not these one-shot embeddings.
/// </summary>
public sealed class F5GpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public sealed class BlockWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public CoreTensor AttnNormLinearW { get; }
        public CoreTensor AttnNormLinearB { get; }
        public CoreTensor ToQW { get; }
        public CoreTensor ToQB { get; }
        public CoreTensor ToKW { get; }
        public CoreTensor ToKB { get; }
        public CoreTensor ToVW { get; }
        public CoreTensor ToVB { get; }
        public CoreTensor ToOutW { get; }
        public CoreTensor ToOutB { get; }
        public CoreTensor FfInW { get; }
        public CoreTensor FfInB { get; }
        public CoreTensor FfOutW { get; }
        public CoreTensor FfOutB { get; }

        public BlockWeights(IComputeBackend backend, F5DiTBlockWeights bw)
        {
            _backend = backend;
            int dim = F5TtsWeights.HiddenDim;
            int ffn = F5TtsWeights.FfnDim;

            // Dequantize from the SAME Q8_0 bytes the CPU reference path (F5DiTBlock.Forward's
            // LinQ8 -- ALWAYS routes through F5Kernels.LinearQ8_0/LinearGpuQ8_0, never raw float32,
            // regardless of backend nullness) uses, rather than the original float32 arrays --
            // found 2026-09-14 via a real single-block parity test showing a ~12% discrepancy that
            // turned out to be this precision mismatch (Q8_0 int8 quantization vs full float32),
            // not a logic bug in any of the GPU ops themselves (all independently re-verified
            // against their own shader source first). Using the identical dequantized values on
            // both sides isolates GPU-vs-CPU compute correctness from quantization-scheme choice.
            AttnNormLinearW = F5GpuWeights.UploadWeightQ8(backend, bw.AttnNormLinearQ8, dim * 6, dim);
            AttnNormLinearB = backend.Upload(bw.AttnNormLinearBias, TensorShape.D1(dim * 6), exact: true);

            ToQW = F5GpuWeights.UploadWeightQ8(backend, bw.ToQQ8, dim, dim);
            ToQB = backend.Upload(bw.ToQBias, TensorShape.D1(dim), exact: true);
            ToKW = F5GpuWeights.UploadWeightQ8(backend, bw.ToKQ8, dim, dim);
            ToKB = backend.Upload(bw.ToKBias, TensorShape.D1(dim), exact: true);
            ToVW = F5GpuWeights.UploadWeightQ8(backend, bw.ToVQ8, dim, dim);
            ToVB = backend.Upload(bw.ToVBias, TensorShape.D1(dim), exact: true);
            ToOutW = F5GpuWeights.UploadWeightQ8(backend, bw.ToOutQ8, dim, dim);
            ToOutB = backend.Upload(bw.ToOutBias, TensorShape.D1(dim), exact: true);

            FfInW = F5GpuWeights.UploadWeightQ8(backend, bw.FfInQ8, ffn, dim);
            FfInB = backend.Upload(bw.FfInBias, TensorShape.D1(ffn), exact: true);
            FfOutW = F5GpuWeights.UploadWeightQ8(backend, bw.FfOutQ8, dim, ffn);
            FfOutB = backend.Upload(bw.FfOutBias, TensorShape.D1(dim), exact: true);
        }

        /// <summary>Plain-float32 constructor for architecturally-identical models with no Q8_0
        /// quantization of their own (e.g. CosyVoice3's DiT, tensor-for-tensor identical to F5-TTS
        /// per <c>CosyVoice3DiTModel.cs</c>'s own doc comment -- its CPU reference path always uses
        /// raw float32 <c>F5Kernels.Linear</c>, never Q8_0, so this constructor keeps GPU-vs-CPU
        /// precision matched exactly, the same lesson the Q8_0 constructor above exists to apply
        /// for F5 itself).</summary>
        public BlockWeights(IComputeBackend backend, int dim, int ffn,
            float[] attnNormW, float[] attnNormB,
            float[] qW, float[] qB, float[] kW, float[] kB, float[] vW, float[] vB, float[] outW, float[] outB,
            float[] ffInW, float[] ffInB, float[] ffOutW, float[] ffOutB)
        {
            _backend = backend;
            AttnNormLinearW = F5GpuWeights.UploadWeight(backend, attnNormW, TensorShape.D2(dim * 6, dim));
            AttnNormLinearB = backend.Upload(attnNormB, TensorShape.D1(dim * 6), exact: true);

            ToQW = F5GpuWeights.UploadWeight(backend, qW, TensorShape.D2(dim, dim));
            ToQB = backend.Upload(qB, TensorShape.D1(dim), exact: true);
            ToKW = F5GpuWeights.UploadWeight(backend, kW, TensorShape.D2(dim, dim));
            ToKB = backend.Upload(kB, TensorShape.D1(dim), exact: true);
            ToVW = F5GpuWeights.UploadWeight(backend, vW, TensorShape.D2(dim, dim));
            ToVB = backend.Upload(vB, TensorShape.D1(dim), exact: true);
            ToOutW = F5GpuWeights.UploadWeight(backend, outW, TensorShape.D2(dim, dim));
            ToOutB = backend.Upload(outB, TensorShape.D1(dim), exact: true);

            FfInW = F5GpuWeights.UploadWeight(backend, ffInW, TensorShape.D2(ffn, dim));
            FfInB = backend.Upload(ffInB, TensorShape.D1(ffn), exact: true);
            FfOutW = F5GpuWeights.UploadWeight(backend, ffOutW, TensorShape.D2(dim, ffn));
            FfOutB = backend.Upload(ffOutB, TensorShape.D1(dim), exact: true);
        }

        public void Dispose()
        {
            _backend.Free(AttnNormLinearW); _backend.Free(AttnNormLinearB);
            _backend.Free(ToQW); _backend.Free(ToQB);
            _backend.Free(ToKW); _backend.Free(ToKB);
            _backend.Free(ToVW); _backend.Free(ToVB);
            _backend.Free(ToOutW); _backend.Free(ToOutB);
            _backend.Free(FfInW); _backend.Free(FfInB);
            _backend.Free(FfOutW); _backend.Free(FfOutB);
        }
    }

    public BlockWeights[] Blocks { get; }
    public CoreTensor NormOutLinearW { get; }
    public CoreTensor NormOutLinearB { get; }
    public CoreTensor ProjOutW { get; }
    public CoreTensor ProjOutB { get; }

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

    /// <summary>Dequantize a Q8_0 weight blob to float32 then upload (fp16/bf16-converted per the
    /// backend's own preference, same as <see cref="UploadWeight"/>) -- matches the CPU reference
    /// path's own precision exactly (see this class's <see cref="BlockWeights"/> constructor doc
    /// comment for why this matters).</summary>
    private static CoreTensor UploadWeightQ8(IComputeBackend backend, byte[] q8Weight, int outDim, int inDim)
    {
        int count = outDim * inDim;
        var f32 = new float[count];
        Cpu.Dequantize.ToFloat32(q8Weight, f32, DType.Q8_0, count);
        return UploadWeight(backend, f32, TensorShape.D2(outDim, inDim));
    }

    public F5GpuWeights(IComputeBackend backend, F5TtsWeights w)
    {
        _backend = backend;
        int dim = F5TtsWeights.HiddenDim;

        Blocks = new BlockWeights[F5TtsWeights.NumLayers];
        for (int i = 0; i < F5TtsWeights.NumLayers; i++)
            Blocks[i] = new BlockWeights(backend, w.Blocks[i]);

        NormOutLinearW = UploadWeightQ8(backend, w.NormOutLinearQ8, dim * 2, dim);
        NormOutLinearB = backend.Upload(w.NormOutLinearBias, TensorShape.D1(dim * 2), exact: true);
        ProjOutW = UploadWeightQ8(backend, w.ProjOutQ8, F5TtsWeights.MelDim, dim);
        ProjOutB = backend.Upload(w.ProjOutBias, TensorShape.D1(F5TtsWeights.MelDim), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in Blocks) b.Dispose();
        _backend.Free(NormOutLinearW);
        _backend.Free(NormOutLinearB);
        _backend.Free(ProjOutW);
        _backend.Free(ProjOutB);
    }
}
