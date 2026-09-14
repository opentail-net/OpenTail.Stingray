using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// GPU-resident VRAM representation of CosyVoice3's DiT transformer blocks + final norm/proj
/// layer. Reuses <see cref="F5GpuWeights.BlockWeights"/> directly (its plain-float32 constructor
/// overload) since CosyVoice3's DiT is tensor-for-tensor architecturally identical to F5-TTS's own
/// (same HiddenDim=1024/NumHeads=16/HeadDim=64/FfnDim=2048, confirmed via
/// <c>CosyVoice3DiTModel.cs</c>'s own doc comment) -- this class only handles CosyVoice3's own
/// weight object (<see cref="CosyVoice3DiTBlockWeights"/>, plain float32, no Q8_0 quantization
/// unlike F5's) and layer count (real, metadata-derived, not hardcoded to 22).
/// </summary>
public sealed class CosyVoice3GpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public F5GpuWeights.BlockWeights[] Blocks { get; }
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

    public CosyVoice3GpuWeights(IComputeBackend backend, CosyVoice3DiTWeights w)
    {
        _backend = backend;
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int ffn = CosyVoice3DiTWeights.FfnDim;

        Blocks = new F5GpuWeights.BlockWeights[w.NumLayers];
        for (int i = 0; i < w.NumLayers; i++)
        {
            var bw = w.Blocks[i];
            Blocks[i] = new F5GpuWeights.BlockWeights(backend, dim, ffn,
                bw.AttnNormLinearWeight, bw.AttnNormLinearBias,
                bw.ToQWeight, bw.ToQBias, bw.ToKWeight, bw.ToKBias, bw.ToVWeight, bw.ToVBias, bw.ToOutWeight, bw.ToOutBias,
                bw.FfInWeight, bw.FfInBias, bw.FfOutWeight, bw.FfOutBias);
        }

        NormOutLinearW = UploadWeight(backend, w.NormOutLinearWeight, TensorShape.D2(dim * 2, dim));
        NormOutLinearB = backend.Upload(w.NormOutLinearBias, TensorShape.D1(dim * 2), exact: true);
        ProjOutW = UploadWeight(backend, w.ProjOutWeight, TensorShape.D2(CosyVoice3DiTWeights.MelDim, dim));
        ProjOutB = backend.Upload(w.ProjOutBias, TensorShape.D1(CosyVoice3DiTWeights.MelDim), exact: true);
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
