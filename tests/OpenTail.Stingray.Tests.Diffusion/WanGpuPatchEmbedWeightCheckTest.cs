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
        if (uploaded.DType == DType.Float16)
        {
            var halfDown = new Half[realWeight.Length];
            backend.DownloadHalf(uploaded, halfDown);
            for (int i = 0; i < realWeight.Length; i++) downloaded[i] = (float)halfDown[i];
        }
        else if (uploaded.DType == DType.BFloat16)
        {
            var bf16Down = new ushort[realWeight.Length];
            backend.DownloadBf16(uploaded, bf16Down);
            for (int i = 0; i < realWeight.Length; i++)
            {
                uint bits = (uint)bf16Down[i] << 16;
                downloaded[i] = BitConverter.UInt32BitsToSingle(bits);
            }
        }
        else
        {
            backend.Download(uploaded, downloaded);
        }
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

    [Fact]
    public unsafe void PatchEmbeddingGpuGemm_MatchesCpuGemm()
    {
        string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\wan2.1\wan2.1-t2v-1.3b-dit.safetensors";
        if (!File.Exists(modelPath)) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var realWeight = loader.ReadF32("patch_embedding.weight");

        VulkanBackend? backend = null;
        try { backend = new VulkanBackend(); } catch { return; }
        using var _ = backend;

        const int dim = 1536, inChannels = 64;
        const int numFrames = 1, latH = 16, latW = 16;
        int patchH = latH / 2, patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW; // 64

        // Load latent
        string latentPath = @"C:\Git-Public\OpenTail.Stingray\examples\stable-diffusion.cpp\wan_cpp_dump_wandbg_input_latent.bin";
        float[] latent;
        if (File.Exists(latentPath))
        {
            var bytes = File.ReadAllBytes(latentPath);
            latent = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, latent, 0, bytes.Length);
        }
        else
        {
            latent = new float[16 * numFrames * latH * latW];
        }

        var packed = WanModel.PackLatents(latent, numFrames, latH, latW);

        // 1. CPU forward computation
        var cpuOut = new float[numTokens * dim];
        fixed (float* pOut = cpuOut, pW = realWeight, pIn = packed)
        {
            OpenTail.Stingray.Cpu.SimdKernels.MatMulBatchedF32(pOut, pW, pIn, numTokens, dim, inChannels, null);
        }

        // 2a. GPU F32 forward computation
        using var uploadedF32 = backend.Upload(realWeight, TensorShape.D2(dim, inChannels), exact: true);
        using var packedGpu = backend.Upload(packed, TensorShape.D2(numTokens, inChannels));
        using var gpuOutF32 = backend.Allocate(TensorShape.D2(numTokens, dim), DType.Float32);

        backend.Sgemm(gpuOutF32, packedGpu, uploadedF32, numTokens, inChannels, dim);

        var gpuF32Downloaded = new float[numTokens * dim];
        backend.Download(gpuOutF32, gpuF32Downloaded);

        // 2b. GPU FP16 forward computation
        var half = new Half[realWeight.Length];
        for (int i = 0; i < realWeight.Length; i++) half[i] = (Half)realWeight[i];
        using var uploadedF16 = backend.UploadHalf(half, TensorShape.D2(dim, inChannels));
        using var gpuOutF16 = backend.Allocate(TensorShape.D2(numTokens, dim), DType.Float32);

        backend.Sgemm(gpuOutF16, packedGpu, uploadedF16, numTokens, inChannels, dim);

        var gpuF16Downloaded = new float[numTokens * dim];
        backend.Download(gpuOutF16, gpuF16Downloaded);

        // 3. Compare CPU F32 vs GPU F32 vs GPU FP16
        static (double cos, double normA, double normB, double maxDiff) Compare(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0, maxDiff = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
                double d = Math.Abs(a[i] - b[i]);
                if (d > maxDiff) maxDiff = d;
            }
            double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
            return (cos, Math.Sqrt(na), Math.Sqrt(nb), maxDiff);
        }

        var (cosF32, cpuNorm, gpuF32Norm, diffF32) = Compare(cpuOut, gpuF32Downloaded);
        var (cosF16, _, gpuF16Norm, diffF16) = Compare(cpuOut, gpuF16Downloaded);
        var (cosF32_F16, _, _, diffF32_F16) = Compare(gpuF32Downloaded, gpuF16Downloaded);

        Console.WriteLine($"[A/B SGEMM Check] CPU F32 vs GPU F32:  cosine={cosF32:F9} cpuNorm={cpuNorm:F6} gpuF32Norm={gpuF32Norm:F6} maxDiff={diffF32:F6}");
        Console.WriteLine($"[A/B SGEMM Check] CPU F32 vs GPU FP16: cosine={cosF16:F9} cpuNorm={cpuNorm:F6} gpuF16Norm={gpuF16Norm:F6} maxDiff={diffF16:F6}");
        Console.WriteLine($"[A/B SGEMM Check] GPU F32 vs GPU FP16: cosine={cosF32_F16:F9} maxDiff={diffF32_F16:F6}");

        // Also compare against dumped files if they exist
        string csDumpPath = @"C:\Git-Public\OpenTail.Stingray\examples\stable-diffusion.cpp\wan_cs_dump_wandbg_patchembed.bin";
        if (File.Exists(csDumpPath))
        {
            var bytes = File.ReadAllBytes(csDumpPath);
            var dumpCpu = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, dumpCpu, 0, bytes.Length);

            var (cosDump, _, dumpNorm, _) = Compare(cpuOut, dumpCpu);
            Console.WriteLine($"[A/B SGEMM Check] CPU F32 vs cs_dump_patchembed: cosine={cosDump:F9} dumpNorm={dumpNorm:F6}");
        }

        string csGpuDumpPath = @"C:\Git-Public\OpenTail.Stingray\examples\stable-diffusion.cpp\wan_cs_dump_wandbg_patchembed_gpu.bin";
        if (File.Exists(csGpuDumpPath))
        {
            var bytes = File.ReadAllBytes(csGpuDumpPath);
            var dumpGpu = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, dumpGpu, 0, bytes.Length);

            var (cosGpuDump, _, dumpGpuNorm, _) = Compare(gpuF16Downloaded, dumpGpu);
            Console.WriteLine($"[A/B SGEMM Check] GPU FP16 vs cs_dump_patchembed_gpu: cosine={cosGpuDump:F9} dumpGpuNorm={dumpGpuNorm:F6}");
        }

        string cppDumpPath = @"C:\Git-Public\OpenTail.Stingray\examples\stable-diffusion.cpp\wan_cpp_dump_wandbg_patchembed.bin";
        if (File.Exists(cppDumpPath))
        {
            var bytes = File.ReadAllBytes(cppDumpPath);
            var dumpCpp = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, dumpCpp, 0, bytes.Length);

            var (cosCppDump, _, dumpCppNorm, _) = Compare(cpuOut, dumpCpp);
            Console.WriteLine($"[A/B SGEMM Check] CPU F32 vs cpp_dump_patchembed: cosine={cosCppDump:F9} dumpCppNorm={dumpCppNorm:F6}");
        }
    }
}

