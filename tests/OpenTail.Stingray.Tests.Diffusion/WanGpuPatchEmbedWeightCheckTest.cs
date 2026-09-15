using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Isolation test (docs/081, 2026-09-14): checks whether the GPU-uploaded patch_embedding weight
/// (via WanGpuWeights.UploadWeight, which converts to FP16/BF16) round-trips to values close to the
/// real checkpoint's raw F32 data, after a real cross-reference bisection found GPU patch_embedding
/// output diverges from the C++ reference at cosine=0.961 -- too large to plausibly be pure FP16
/// rounding noise on a 64-term dot product. This isolates whether the weight itself is wrong
/// (upload/shape bug) vs. the GEMM/shader being wrong given a correct weight.
/// </summary>
public sealed class WanGpuPatchEmbedWeightCheckTest
{
    [Fact]
    public void PatchEmbeddingGpuWeight_DownloadedValues_MatchRealCheckpoint()
    {
        string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\wan2.1\wan2.1-t2v-1.3b-dit.safetensors";
        if (!File.Exists(modelPath)) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var realWeight = loader.ReadF32("patch_embedding.weight"); // real shape [1536, 16, 1, 2, 2] flattened -> [1536, 64]

        VulkanBackend? backend = null;
        try { backend = new VulkanBackend(); } catch { return; }
        using var _ = backend;

        Console.WriteLine($"[WanGpuPatchEmbedWeightCheck] backend.BestSgemmPrecision={backend.BestSgemmPrecision}");
        const int dim = 1536, inChannels = 64;

        Tensor uploaded;
        if (backend.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[realWeight.Length];
            for (int i = 0; i < realWeight.Length; i++) half[i] = (Half)realWeight[i];
            uploaded = backend.UploadHalf(half, TensorShape.D2(dim, inChannels));
        }
        else if (backend.BestSgemmPrecision == SgemmPrecision.Bf16)
        {
            var bf16 = new ushort[realWeight.Length];
            for (int i = 0; i < realWeight.Length; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits(realWeight[i]);
                bf16[i] = (ushort)(bits >> 16);
            }
            uploaded = backend.UploadBf16(bf16, TensorShape.D2(dim, inChannels));
        }
        else
        {
            uploaded = backend.Upload(realWeight, TensorShape.D2(dim, inChannels), exact: true);
        }

        var downloaded = new float[realWeight.Length];
        backend.Download(uploaded, downloaded);
        backend.Free(uploaded);

        double dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < realWeight.Length; i++)
        {
            dot += (double)realWeight[i] * downloaded[i];
            na += (double)realWeight[i] * realWeight[i];
            nb += (double)downloaded[i] * downloaded[i];
            double d = Math.Abs(realWeight[i] - downloaded[i]);
            if (d > maxDiff) maxDiff = d;
        }
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"[WanGpuPatchEmbedWeightCheck] cosine(real, gpu-roundtrip)={cos:F9} realNorm={Math.Sqrt(na):F6} gpuNorm={Math.Sqrt(nb):F6} maxDiff={maxDiff:F6}");
        Console.WriteLine($"[WanGpuPatchEmbedWeightCheck] real[0..4]={realWeight[0]:F8},{realWeight[1]:F8},{realWeight[2]:F8},{realWeight[3]:F8}");
        Console.WriteLine($"[WanGpuPatchEmbedWeightCheck] gpu [0..4]={downloaded[0]:F8},{downloaded[1]:F8},{downloaded[2]:F8},{downloaded[3]:F8}");
    }
}
