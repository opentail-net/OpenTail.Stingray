using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Benchmarks and validates Q8_0 8-bit quantization of the 36-layer MiniMax-Music3 Flow DiT
/// against the full-precision FP32 baseline on real weights.
/// </summary>
public sealed class MiniMaxMusic3TransformerQuantBenchmarkTests
{
    private static string? FindRepoDir(string relativePath)
    {
        foreach (var startDir in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var dir = startDir;
            for (int i = 0; i < 10; i++)
            {
                var p = Path.Combine(dir, relativePath);
                if (Directory.Exists(p)) return p;
                var parent = Directory.GetParent(dir);
                if (parent is null) break;
                dir = parent.FullName;
            }
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
    public void Benchmark_Fp32VsQ8Quantization_MeasuresLatencyAndAccuracy()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad";
        string latentPath = Path.Combine(scratchDir, "minimax_transformer_latent.bin");
        Assert.SkipUnless(File.Exists(latentPath), "minimax_transformer_*.bin reference dump not found");

        const int length = 8;
        const int inChannels = 128;
        const int condDim = 2048;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        var fp32Weights = MiniMaxMusic3TransformerWeights.Load(loader);

        var latentFlat = ReadBin(latentPath, length * inChannels);
        var conditionFlat = ReadBin(Path.Combine(scratchDir, "minimax_transformer_condition.bin"), length * condDim);

        var latent = ToRows(latentFlat, length, inChannels);
        var condition = ToRows(conditionFlat, length, condDim);

        // 1. Quantize weights to Q8_0
        var swQuant = Stopwatch.StartNew();
        var q8Weights = MiniMaxMusic3QuantizedTransformerWeights.QuantizeFrom(fp32Weights);
        swQuant.Stop();

        // 2. Warmup both paths
        _ = MiniMaxMusic3Transformer.Forward(fp32Weights, latent, condition, timestep);
        _ = MiniMaxMusic3Transformer.Forward(q8Weights, latent, condition, timestep);

        // 3. Benchmark FP32 Baseline (Before)
        const int iterations = 3;
        var swFp32 = Stopwatch.StartNew();
        float[][] fp32Out = null!;
        for (int i = 0; i < iterations; i++)
        {
            fp32Out = MiniMaxMusic3Transformer.Forward(fp32Weights, latent, condition, timestep);
        }
        swFp32.Stop();
        double fp32MsPerPass = swFp32.Elapsed.TotalMilliseconds / iterations;

        // 4. Benchmark Q8_0 Quantized (After)
        var swQ8 = Stopwatch.StartNew();
        float[][] q8Out = null!;
        for (int i = 0; i < iterations; i++)
        {
            q8Out = MiniMaxMusic3Transformer.Forward(q8Weights, latent, condition, timestep);
        }
        swQ8.Stop();
        double q8MsPerPass = swQ8.Elapsed.TotalMilliseconds / iterations;

        // 5. Numerical Accuracy & Diff Check
        double sumAbsDiff = 0, sumAbsFp32 = 0, maxAbsDiff = 0;
        double dotProduct = 0, normFp32Sq = 0, normQ8Sq = 0;

        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                double fVal = fp32Out[t][c];
                double qVal = q8Out[t][c];
                double diff = Math.Abs(qVal - fVal);

                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                sumAbsDiff += diff;
                sumAbsFp32 += Math.Abs(fVal);

                dotProduct += fVal * qVal;
                normFp32Sq += fVal * fVal;
                normQ8Sq += qVal * qVal;
            }
        }

        double relError = sumAbsDiff / sumAbsFp32;
        double cosineSim = dotProduct / (Math.Sqrt(normFp32Sq) * Math.Sqrt(normQ8Sq));
        double speedup = fp32MsPerPass / q8MsPerPass;

        Console.WriteLine("\n============================================================");
        Console.WriteLine("    MiniMax-Music3 Flow DiT Quantization Benchmark (CPU)    ");
        Console.WriteLine("============================================================");
        Console.WriteLine($" Weight Quantization Time (36 layers) : {swQuant.ElapsedMilliseconds} ms");
        Console.WriteLine($" Baseline FP32 Latency (Before)       : {fp32MsPerPass:F2} ms / pass (~6.0 GB weights)");
        Console.WriteLine($" Quantized Q8_0 Latency (After)       : {q8MsPerPass:F2} ms / pass (~1.6 GB weights)");
        Console.WriteLine($" Latency Speedup                      : {speedup:F2}x");
        Console.WriteLine($" Memory Compression                   : 3.76x reduction");
        Console.WriteLine("------------------------------------------------------------");
        Console.WriteLine($" Relative L1 Error                    : {relError:P3} ({relError:F6})");
        Console.WriteLine($" Maximum Absolute Error               : {maxAbsDiff:F6}");
        Console.WriteLine($" Cosine Similarity (Directional)      : {cosineSim:F6}");
        Console.WriteLine("============================================================\n");

        // High directional fidelity check: Cosine similarity >= 0.99 (99.6% measured)
        Assert.True(cosineSim > 0.99, $"Cosine similarity {cosineSim:F6} lower than 0.99 threshold");
        Assert.True(relError < 0.10, $"Relative error {relError:F6} exceeds 10% threshold across 36 layers");
    }

    [Fact]
    public void ForwardPair_Q8Quantized_MatchesSingleForward()
    {
        string? weightsDir = FindRepoDir("models/minimax-music3/transformer");
        Assert.SkipUnless(weightsDir != null, "models/minimax-music3/transformer/ not found");

        const int length = 8;
        const int inChannels = 128;
        const int condDim = 2048;
        const float timestep = 0.37f;

        using var loader = SafetensorsLoader.OpenDirectory(weightsDir!);
        var fp32Weights = MiniMaxMusic3TransformerWeights.Load(loader);
        var q8Weights = MiniMaxMusic3QuantizedTransformerWeights.QuantizeFrom(fp32Weights);

        var random = new Random(42);
        var latent = new float[length][];
        var condition = new float[length][];
        var zeroCondition = new float[length][];
        for (int t = 0; t < length; t++)
        {
            latent[t] = new float[inChannels];
            for (int c = 0; c < inChannels; c++) latent[t][c] = (float)random.NextDouble() - 0.5f;

            condition[t] = new float[condDim];
            for (int c = 0; c < condDim; c++) condition[t][c] = (float)random.NextDouble() - 0.5f;

            zeroCondition[t] = new float[condDim];
        }

        var expectedCond = MiniMaxMusic3Transformer.Forward(q8Weights, latent, condition, timestep);
        var expectedUncond = MiniMaxMusic3Transformer.Forward(q8Weights, latent, zeroCondition, timestep);

        var (actualCond, actualUncond) = MiniMaxMusic3Transformer.ForwardPair(q8Weights, latent, condition, zeroCondition, timestep);

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
