using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// The KV-outer prefill-attention reorder packs each KV tile once per group of query tiles instead
/// of once per query tile. It is a loop interchange, not a reassociation: every query row still
/// consumes KV tiles in ascending order, so its online-softmax accumulator sees the same sequence.
/// The contract is therefore <b>bit-identical output</b>, and that is what this pins.
///
/// <para>The two schedules are compared inside one process by flipping
/// <c>ForwardPass.Flash64KvOuterEnabled</c>. That is deliberate. The reorder short-circuits ahead of
/// the tile-jobs branch, so configuring it by environment variable and running the existing
/// schedule-comparison test would put both arms on the reordered path and compare it with itself —
/// green, and worthless.</para>
/// </summary>
public sealed class Flash64KvOuterTests : HeavyTestBase
{
    /// <summary>Above Flash-64's 256-token activation threshold, and several query tiles wide.</summary>
    private const int Tokens = 448;

    [Fact]
    public void KvOuter_MatchesDefaultSchedule_BitExactly()
    {
        string? path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2 GGUF not present — model-backed comparison not applicable");
        Assert.SkipUnless(Avx2.IsSupported && Fma.IsSupported, "AVX2/FMA required for the Flash-64 path");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        int[] tokens = BuildTokens(Tokens);

        bool previous = Engine.ForwardPass.Flash64KvOuterEnabled;
        try
        {
            Engine.ForwardPass.Flash64KvOuterEnabled = false;
            float[] baseline = Prefill(model, backend, hp, tokens);

            Engine.ForwardPass.Flash64KvOuterEnabled = true;
            float[] reordered = Prefill(model, backend, hp, tokens);

            Assert.Equal(baseline.Length, reordered.Length);
            for (int i = 0; i < baseline.Length; i++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(baseline[i]) == BitConverter.SingleToInt32Bits(reordered[i]),
                    $"logit {i} differs: default {baseline[i]} vs KV-outer {reordered[i]}");
            }
        }
        finally
        {
            Engine.ForwardPass.Flash64KvOuterEnabled = previous;
        }
    }

    /// <summary>
    /// Group size must not change the answer — it only trades scratch footprint for K-pack reuse.
    /// A group of 1 degenerates to the original per-query-tile packing and a group larger than the
    /// sequence holds every tile live at once; both must still agree bit-for-bit with the default
    /// schedule, which is what makes the tunable safe to tune.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]   // does not divide the tile count evenly — exercises the ragged final group
    [InlineData(64)]  // larger than the sequence: one group holds everything
    public void KvOuter_GroupSizeDoesNotChangeResult(int groupTiles)
    {
        string? path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2 GGUF not present — model-backed comparison not applicable");
        Assert.SkipUnless(Avx2.IsSupported && Fma.IsSupported, "AVX2/FMA required for the Flash-64 path");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        int[] tokens = BuildTokens(Tokens);

        bool previous = Engine.ForwardPass.Flash64KvOuterEnabled;
        string? previousTiles = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER_TILES");
        try
        {
            Engine.ForwardPass.Flash64KvOuterEnabled = false;
            float[] baseline = Prefill(model, backend, hp, tokens);

            Environment.SetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER_TILES",
                groupTiles.ToString(System.Globalization.CultureInfo.InvariantCulture));
            Engine.ForwardPass.Flash64KvOuterEnabled = true;
            float[] reordered = Prefill(model, backend, hp, tokens);

            for (int i = 0; i < baseline.Length; i++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(baseline[i]) == BitConverter.SingleToInt32Bits(reordered[i]),
                    $"groupTiles={groupTiles}: logit {i} differs: {baseline[i]} vs {reordered[i]}");
            }
        }
        finally
        {
            Engine.ForwardPass.Flash64KvOuterEnabled = previous;
            Environment.SetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER_TILES", previousTiles);
        }
    }

    private static float[] Prefill(GgufModel model, CpuBackend backend, ModelHyperparams hp, int[] tokens)
    {
        using var forward = new Engine.ForwardPass(model, backend, hp);
        using var cache = forward.CreateCache();
        return forward.PrefillWithCache(tokens, cache).ToArray();
    }

    private static int[] BuildTokens(int count) =>
        Enumerable.Range(0, count).Select(i => 1 + i * 17 % 997).ToArray();

    private static string? FindModelPath()
    {
        string? directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && directory is not null; i++)
        {
            string candidate = Path.Combine(directory, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            directory = Directory.GetParent(directory)?.FullName;
        }
        return null;
    }
}
