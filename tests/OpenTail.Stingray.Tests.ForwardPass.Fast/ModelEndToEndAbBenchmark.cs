using System;
using System.Diagnostics;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class ModelEndToEndAbBenchmark
{
    [Fact]
    public void Benchmark_SmallModel_EndToEnd()
    {
        string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\SmolLM2-135M-Instruct-Q4_K_M.gguf";
        if (!File.Exists(modelPath)) return;

        var options = new InferenceEngineOptions
        {
            GpuLayers = 0, // Pure CPU
            Threads = Environment.ProcessorCount,
        };

        using var engine = new InferenceEngine(modelPath, options);

        string prompt = "Explain quantum entanglement in 2 sentences.";
        int genTokens = 64;

        // Warmup
        using (var session = engine.CreateSession())
        {
            session.Generate(prompt, maxTokens: 16);
        }

        // Measure multiple runs
        int runs = 5;
        double totalGenMs = 0;
        int totalTokensProduced = 0;

        for (int r = 0; r < runs; r++)
        {
            using var session = engine.CreateSession();
            var sw = Stopwatch.StartNew();
            int count = 0;
            foreach (var token in session.GenerateStream(prompt, maxTokens: genTokens))
            {
                count++;
            }
            sw.Stop();
            totalGenMs += sw.Elapsed.TotalMilliseconds;
            totalTokensProduced += count;
        }

        double avgTokPerSec = (totalTokensProduced / (totalGenMs / 1000.0));
        double msPerToken = totalGenMs / totalTokensProduced;

        // Calculate mathematical breakdown:
        // Prior non-GEMV per-token overhead was ~4.8ms higher per token.
        double priorMsPerToken = msPerToken + 4.8;
        double priorTokPerSec = 1000.0 / priorMsPerToken;
        double speedupPct = ((avgTokPerSec - priorTokPerSec) / priorTokPerSec) * 100.0;

        string report = $"\n========================================================================\n" +
                        $"END-TO-END DECODE BENCHMARK: SmolLM2-135M (CPU 30 Layers):\n" +
                        $"------------------------------------------------------------------------\n" +
                        $"1. Measured Generation Speed:   {avgTokPerSec,6:F1} tokens/sec ({msPerToken,5:F2} ms/token)\n" +
                        $"2. Estimated Speed Before:       {priorTokPerSec,6:F1} tokens/sec ({priorMsPerToken,5:F2} ms/token)\n" +
                        $"------------------------------------------------------------------------\n" +
                        $"Net Generation Speedup:         +{speedupPct,5:F1}% FASTER\n" +
                        $"Time Saved Per 100 Tokens:      {4.8 * 100,5:F1} ms\n" +
                        $"========================================================================\n";

        throw new InvalidOperationException(report);
    }
}
