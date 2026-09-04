using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Verifies numeric parity and execution of <see cref="MiniMaxMusic3Transformer"/>
/// when offloading GEMM projections to GPU via <see cref="IComputeBackend"/> (Vulkan).
/// Compares GPU output against CPU baseline.
/// </summary>
public sealed class MiniMaxMusic3TransformerGpuParityTests
{
    private static VulkanBackend CreateBackendOrSkip()
    {
        try
        {
            return new VulkanBackend();
        }
        catch (Exception ex)
        {
            Assert.Skip($"Vulkan device could not be created in this environment: {ex.Message}");
            throw;
        }
    }

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadBin(string path, int count)
    {
        var bytes = File.ReadAllBytes(path);
        Assert.Equal(count * 4, bytes.Length);
        var result = new float[count];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    private static float[][] ToRows(float[] flat, int rows, int cols)
    {
        var result = new float[rows][];
        for (int r = 0; r < rows; r++)
        {
            result[r] = new float[cols];
            Array.Copy(flat, r * cols, result[r], 0, cols);
        }
        return result;
    }

    [Fact]
    public void Forward_GpuBackend_MatchesCpuWithinFp16Tolerance()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string latentPath = Path.Combine(scratchDir, "minimax_transformer_latent.bin");
        Assert.SkipUnless(File.Exists(latentPath), "minimax_transformer_*.bin reference dump not found");

        using var backend = CreateBackendOrSkip();

        const int length = 8;
        const int inChannels = 128;
        const int condDim = 2048;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        using var weights = MiniMaxMusic3TransformerWeights.Load(loader);

        var latentFlat = ReadBin(latentPath, length * inChannels);
        var conditionFlat = ReadBin(Path.Combine(scratchDir, "minimax_transformer_condition.bin"), length * condDim);

        var latent = ToRows(latentFlat, length, inChannels);
        var condition = ToRows(conditionFlat, length, condDim);

        // Warmup / upload weights to GPU once
        _ = weights.GetOrCreateGpuWeights(backend);

        // Benchmark CPU reference
        var swCpu = System.Diagnostics.Stopwatch.StartNew();
        var cpuOutput = MiniMaxMusic3Transformer.Forward(weights, latent, condition, timestep, backend: null);
        swCpu.Stop();

        // Benchmark GPU offloaded
        var swGpu = System.Diagnostics.Stopwatch.StartNew();
        var gpuOutput = MiniMaxMusic3Transformer.Forward(weights, latent, condition, timestep, backend: backend);
        swGpu.Stop();

        Console.WriteLine($"\n[BENCHMARK] DiT Forward (36 layers): CPU = {swCpu.ElapsedMilliseconds} ms, GPU = {swGpu.ElapsedMilliseconds} ms");

        Assert.Equal(length, gpuOutput.Length);

        double sumAbsDiff = 0, sumAbsCpu = 0, maxAbsDiff = 0;
        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                double diff = Math.Abs(gpuOutput[t][c] - cpuOutput[t][c]);
                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                sumAbsDiff += diff;
                sumAbsCpu += Math.Abs(cpuOutput[t][c]);
            }
        }
        double relError = sumAbsDiff / sumAbsCpu;

        // FP16 weights vs FP32 accumulation tolerance across 36 transformer layers
        Assert.True(relError < 0.01, $"Relative error {relError:F6} exceeds tolerance (maxAbsDiff={maxAbsDiff:F6})");
    }

    [Fact]
    public void ForwardPair_GpuBackend_MatchesForwardWithinTolerance()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string latentPath = Path.Combine(scratchDir, "minimax_transformer_latent.bin");
        Assert.SkipUnless(File.Exists(latentPath), "minimax_transformer_*.bin reference dump not found");

        using var backend = CreateBackendOrSkip();

        const int length = 8;
        const int inChannels = 128;
        const int condDim = 2048;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        using var weights = MiniMaxMusic3TransformerWeights.Load(loader);

        var latentFlat = ReadBin(latentPath, length * inChannels);
        var conditionFlat = ReadBin(Path.Combine(scratchDir, "minimax_transformer_condition.bin"), length * condDim);

        var latent = ToRows(latentFlat, length, inChannels);
        var condition = ToRows(conditionFlat, length, condDim);
        var zeroCondition = new float[length][];
        for (int t = 0; t < length; t++) zeroCondition[t] = new float[condDim];

        var expectedCond = MiniMaxMusic3Transformer.Forward(weights, latent, condition, timestep, backend: backend);
        var expectedUncond = MiniMaxMusic3Transformer.Forward(weights, latent, zeroCondition, timestep, backend: backend);

        var (actualCond, actualUncond) = MiniMaxMusic3Transformer.ForwardPair(weights, latent, condition, zeroCondition, timestep, backend: backend);

        Assert.Equal(length, actualCond.Length);
        Assert.Equal(length, actualUncond.Length);

        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                Assert.Equal(expectedCond[t][c], actualCond[t][c], 1e-4f);
                Assert.Equal(expectedUncond[t][c], actualUncond[t][c], 1e-4f);
            }
        }
    }
}
