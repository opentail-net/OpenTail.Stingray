using System.Buffers;
using CoreTensor = OpenTail.Stingray.Core.Tensor;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// GPU-resident weights for <see cref="MiniMaxMusic3Transformer"/>, uploaded to device memory
/// (Vulkan/CUDA) on first use and reused across all denoising steps.
/// </summary>
public sealed class MiniMaxMusic3GpuTransformerWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public CoreTensor PreprocessConvWeight { get; }
    public CoreTensor ProjInWeight { get; }
    public CoreTensor ProjOutWeight { get; }
    public CoreTensor PostprocessConvWeight { get; }
    public MiniMaxMusic3GpuTransformerBlockWeights[] Blocks { get; }

    public MiniMaxMusic3GpuTransformerWeights(MiniMaxMusic3TransformerWeights cpuWeights, IComputeBackend backend)
    {
        _backend = backend;
        bool useFp16 = backend.BestSgemmPrecision == SgemmPrecision.Fp16;

        PreprocessConvWeight = UploadWeight(cpuWeights.PreprocessConvWeight, useFp16);
        ProjInWeight = UploadWeight(cpuWeights.ProjInWeight, useFp16);
        ProjOutWeight = UploadWeight(cpuWeights.ProjOutWeight, useFp16);
        PostprocessConvWeight = UploadWeight(cpuWeights.PostprocessConvWeight, useFp16);

        Blocks = new MiniMaxMusic3GpuTransformerBlockWeights[cpuWeights.Blocks.Length];
        for (int i = 0; i < Blocks.Length; i++)
        {
            var b = cpuWeights.Blocks[i];
            Blocks[i] = new MiniMaxMusic3GpuTransformerBlockWeights
            {
                QWeight = UploadWeight(b.QWeight, useFp16),
                KWeight = UploadWeight(b.KWeight, useFp16),
                VWeight = UploadWeight(b.VWeight, useFp16),
                OWeight = UploadWeight(b.OWeight, useFp16),
                FfInWeight = UploadWeight(b.FfInWeight, useFp16),
                FfOutWeight = UploadWeight(b.FfOutWeight, useFp16),
            };
        }
    }

    private CoreTensor UploadWeight(float[] w, bool useFp16)
    {
        if (useFp16)
        {
            var half = ArrayPool<Half>.Shared.Rent(w.Length);
            try
            {
                for (int i = 0; i < w.Length; i++) half[i] = (Half)w[i];
                return _backend.UploadHalf(half.AsSpan(0, w.Length), TensorShape.D1(w.Length));
            }
            finally
            {
                ArrayPool<Half>.Shared.Return(half);
            }
        }
        return _backend.Upload(w, TensorShape.D1(w.Length), exact: true);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(PreprocessConvWeight);
        _backend.Free(ProjInWeight);
        _backend.Free(ProjOutWeight);
        _backend.Free(PostprocessConvWeight);

        if (Blocks is not null)
        {
            for (int i = 0; i < Blocks.Length; i++)
            {
                var b = Blocks[i];
                if (b is null) continue;
                _backend.Free(b.QWeight);
                _backend.Free(b.KWeight);
                _backend.Free(b.VWeight);
                _backend.Free(b.OWeight);
                _backend.Free(b.FfInWeight);
                _backend.Free(b.FfOutWeight);
            }
        }
    }
}

public sealed class MiniMaxMusic3GpuTransformerBlockWeights
{
    public required CoreTensor QWeight { get; init; }
    public required CoreTensor KWeight { get; init; }
    public required CoreTensor VWeight { get; init; }
    public required CoreTensor OWeight { get; init; }
    public required CoreTensor FfInWeight { get; init; }
    public required CoreTensor FfOutWeight { get; init; }
}
