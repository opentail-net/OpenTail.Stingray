using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

public sealed class Flash64SchedulingTests : HeavyTestBase
{
    private const string Flash64Variable = "STINGRAY_PREFILL_ATTN_FLASH64";
    private const string TileJobsVariable = "STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS";

    [Fact]
    public void TileJobs_MatchHeadJobs_BitExactly()
    {
        string? path = FindModelPath();
        if (path is null || !Avx2.IsSupported || !Fma.IsSupported) return;

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
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

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
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

    /// <summary>
    /// Production-shape gate for the strided Flash-64 route. Qwen3-8B has 128-wide attention
    /// heads: before the head-width generalisation it always took the materialised-score path.
    /// The flash result is compared with that established fallback, rather than only exercising
    /// the GEMM in isolation.
    ///
    /// <para>SKIPPED, and it must stay skipped rather than simply run: the 128/256 head widths are
    /// currently held back at the Flash-64 activation gate in <c>ForwardPass</c> (see the long
    /// comment there for the full investigation and the route back). With them held back, both
    /// arms of this comparison take the materialised path, so the test would report a confident
    /// PASS while exercising nothing at all — the precise failure mode this suite exists to
    /// prevent. It last ran for real at maxAbs 0.310 against a 0.01 tolerance.</para>
    /// </summary>
    [Fact(Skip = "128/256 head widths are held back at the Flash-64 gate in ForwardPass; " +
                 "un-skip together with re-enabling them, and see that comment for the next step " +
                 "(measure cosine + greedy agreement against the accepted Q8-vs-F32 baseline).")]
    public void Flash128_MatchesMaterialisedAttention()
    {
        string? path = FindModelPath("Qwen3-8B-Q4_K_M.gguf");
        if (path is null || !Avx2.IsSupported || !Fma.IsSupported) return;

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        Assert.Equal(128, hp.HeadDim);
        using var backend = new CpuBackend();
        int[] tokens = BuildTokens(256);

        string? previousFlash = Environment.GetEnvironmentVariable(Flash64Variable);
        try
        {
            Environment.SetEnvironmentVariable(Flash64Variable, "0");
            float[] fallback = Prefill(model, backend, hp, tokens);

            Environment.SetEnvironmentVariable(Flash64Variable, "1");
            float[] flash = Prefill(model, backend, hp, tokens);

            Assert.Equal(fallback.Length, flash.Length);
            float maxAbs = 0f;
            for (int i = 0; i < fallback.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(fallback[i] - flash[i]));

            Assert.InRange(maxAbs, 0f, 0.01f);
            Assert.Equal(Sampler.Greedy(fallback), Sampler.Greedy(flash));
        }
        finally
        {
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

    private static string? FindModelPath(string filename = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    {
        string? directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && directory is not null; i++)
        {
            string candidate = Path.Combine(
                directory, "models", filename);
            if (File.Exists(candidate)) return candidate;
            directory = Directory.GetParent(directory)?.FullName;
        }
        return null;
    }
}
