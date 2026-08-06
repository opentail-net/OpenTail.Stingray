using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

public sealed class Flash64SchedulingTests
{
    private const string Flash64Variable = "STINGRAY_PREFILL_ATTN_FLASH64";
    private const string TileJobsVariable = "STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS";

    [Fact]
    public void TileJobs_MatchHeadJobs_BitExactly()
    {
        string? path = FindModelPath();
        if (path is null || !Avx2.IsSupported || !Fma.IsSupported) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        int[] tokens = BuildTokens(320); // Above Flash64's 256-token activation threshold.

        string? previousFlash = Environment.GetEnvironmentVariable(Flash64Variable);
        string? previousTileJobs = Environment.GetEnvironmentVariable(TileJobsVariable);
        try
        {
            Environment.SetEnvironmentVariable(Flash64Variable, "1");
            Environment.SetEnvironmentVariable(TileJobsVariable, "0");
            float[] headJobs = Prefill(model, backend, hp, tokens);

            Environment.SetEnvironmentVariable(TileJobsVariable, "1");
            float[] tileJobs = Prefill(model, backend, hp, tokens);

            Assert.Equal(headJobs, tileJobs);
        }
        finally
        {
            Environment.SetEnvironmentVariable(TileJobsVariable, previousTileJobs);
            Environment.SetEnvironmentVariable(Flash64Variable, previousFlash);
        }
    }

    [Fact]
    public void TileJobs_ChunkedPrefill_MatchesSingleCall()
    {
        string? path = FindModelPath();
        if (path is null || !Avx2.IsSupported || !Fma.IsSupported) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        int[] tokens = BuildTokens(512);

        string? previousFlash = Environment.GetEnvironmentVariable(Flash64Variable);
        string? previousTileJobs = Environment.GetEnvironmentVariable(TileJobsVariable);
        try
        {
            Environment.SetEnvironmentVariable(Flash64Variable, "1");
            Environment.SetEnvironmentVariable(TileJobsVariable, "1");

            float[] singleCall = Prefill(model, backend, hp, tokens);
            float[] chunked = PrefillChunked(model, backend, hp, tokens, chunkSize: 256);

            Assert.Equal(singleCall.Length, chunked.Length);
            float maxAbs = 0f;
            for (int i = 0; i < singleCall.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(singleCall[i] - chunked[i]));

            Assert.InRange(maxAbs, 0f, 0.01f);
            Assert.Equal(Sampler.Greedy(singleCall), Sampler.Greedy(chunked));
        }
        finally
        {
            Environment.SetEnvironmentVariable(TileJobsVariable, previousTileJobs);
            Environment.SetEnvironmentVariable(Flash64Variable, previousFlash);
        }
    }

    private static float[] Prefill(
        GgufModel model, CpuBackend backend, ModelHyperparams hp, int[] tokens)
    {
        using var forward = new Engine.ForwardPass(model, backend, hp);
        using var cache = forward.CreateCache();
        return forward.PrefillWithCache(tokens, cache).ToArray();
    }

    private static float[] PrefillChunked(
        GgufModel model, CpuBackend backend, ModelHyperparams hp, int[] tokens, int chunkSize)
    {
        using var forward = new Engine.ForwardPass(model, backend, hp);
        using var cache = forward.CreateCache();
        float[] logits = [];
        for (int start = 0; start < tokens.Length; start += chunkSize)
        {
            int count = Math.Min(chunkSize, tokens.Length - start);
            logits = forward.PrefillWithCache(
                new ArraySegment<int>(tokens, start, count), cache, start).ToArray();
        }
        return logits;
    }

    private static int[] BuildTokens(int count) =>
        Enumerable.Range(0, count).Select(i => 1 + i * 17 % 997).ToArray();

    private static string? FindModelPath()
    {
        string? directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && directory is not null; i++)
        {
            string candidate = Path.Combine(
                directory, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            directory = Directory.GetParent(directory)?.FullName;
        }
        return null;
    }
}
