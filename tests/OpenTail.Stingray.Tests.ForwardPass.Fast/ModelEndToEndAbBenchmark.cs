using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// End-to-end CPU decode throughput gauge for SmolLM2-135M. Rewritten to the current
/// <see cref="InferenceEngine"/>(fwd, tokenizer, name) / <see cref="InferenceEngine.GenerateAsync"/>
/// API (the old <c>InferenceEngineOptions</c>/path-constructor/<c>GenerateStream</c> surface this
/// file used no longer exists) -- see <see cref="SmolLm2RealWeightsTests"/> for the same
/// model-loading pattern. Reports via <see cref="ITestOutputHelper"/> instead of throwing, since a
/// benchmark reporting its own numbers is not a test failure.
/// </summary>
public sealed class ModelEndToEndAbBenchmark(ITestOutputHelper output)
{
    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public async Task Benchmark_SmallModel_EndToEnd()
    {
        string? modelPath = FindModelPath("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        var fwd = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp, maxContextLength: 512);
        using var engine = new InferenceEngine(fwd, tokenizer, "smollm2-135m");

        const string prompt = "Explain quantum entanglement in 2 sentences.";
        const int genTokens = 64;

        // Warmup
        await foreach (var _ in engine.GenerateAsync(prompt, new SamplingParams { Temperature = 0f, MaxNewTokens = 16 })) { }

        const int runs = 5;
        double totalGenMs = 0;
        int totalTokensProduced = 0;

        for (int r = 0; r < runs; r++)
        {
            var sw = Stopwatch.StartNew();
            int count = 0;
            await foreach (var _ in engine.GenerateAsync(prompt, new SamplingParams { Temperature = 0f, MaxNewTokens = genTokens }))
            {
                count++;
            }
            sw.Stop();
            totalGenMs += sw.Elapsed.TotalMilliseconds;
            totalTokensProduced += count;
        }

        Assert.True(totalTokensProduced > 0, "Benchmark produced zero tokens across all runs");

        double avgTokPerSec = totalTokensProduced / (totalGenMs / 1000.0);
        double msPerToken = totalGenMs / totalTokensProduced;

        output.WriteLine(
            $"END-TO-END DECODE BENCHMARK: SmolLM2-135M (CPU): " +
            $"{avgTokPerSec:F1} tokens/sec ({msPerToken:F2} ms/token) over {runs} runs, {totalTokensProduced} tokens total.");
    }
}
