using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// HotSession replay-divergence investigation, second suspect (ruled out, kept as a regression
/// guard): does <see cref="Engine.ForwardPass.BatchForwardMulti"/> -- what
/// <c>ContinuousBatchingEngine</c> uses for EVERY decode step, even a single active sequence --
/// produce different results from the oracle's plain <see cref="Engine.ForwardPass.Forward"/>,
/// the way <see cref="PrefillPathParityTests"/> found for prefill? Measured: no. A batch-of-one
/// <c>BatchForwardMulti</c> call and a plain <c>Forward</c> call are bit-identical
/// (maxAbsDiff=0 across all 49152 vocab lanes) for the same position. Decode is not the source of
/// the remaining <c>HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel</c> divergence.
///
/// <para>Note for anyone re-running this comparison: <see cref="Engine.ForwardPass.Forward"/>
/// operates on the <c>ForwardPass</c> instance's OWN internal cache, not an external
/// <c>PagedKvCache</c> -- an earlier version of this test compared against an unprimed internal
/// cache and got a spurious maxAbsDiff of ~23 with a flipped argmax. Use a second, separately
/// <c>Prefill</c>-ed <c>ForwardPass</c> instance for the oracle arm, as below.</para>
/// </summary>
public sealed class DecodePathParityTests : HeavyTestBase
{
    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void SingleSequenceBatchForwardMulti_VsPlainForward_ForTheSamePosition()
    {
        var path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this parity check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        const string seed = "The capital of France is Paris and the capital of Spain is Madrid and the";
        var tokens = tokenizer.Encode(seed).ToArray();
        Assert.True(tokens.Length >= 15, $"seed tokenised to {tokens.Length} tokens; expected >= 15.");

        int prefixLen = tokens.Length - 1;
        var prefix = tokens[..prefixLen];
        int lastToken = tokens[prefixLen];

        // Arm 1 -- oracle: plain single-token Forward, exactly what GreedyContinuation uses. This
        // operates on ForwardPass's OWN internal cache (not an external PagedKvCache), so it must
        // be populated via Prefill (not PrefillWithCache) -- a separate ForwardPass instance keeps
        // this arm's internal state from ever touching Arm 2's external cache.
        using var fwdOracle = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        fwdOracle.Prefill(prefix);
        var logitsOracle = fwdOracle.Forward(lastToken, prefixLen).ToArray();

        // Arm 2 -- session: a batch-of-one BatchForwardMulti call against an external PagedKvCache,
        // exactly what ContinuousBatchingEngine's decode loop uses even for a lone active sequence.
        var cacheSession = fwd.CreateCache();
        fwd.PrefillWithCache(prefix, cacheSession, startPos: 0);
        var logitsSession = fwd.BatchForwardMulti([lastToken], [prefixLen], [cacheSession])[0];

        Assert.Equal(logitsOracle.Length, logitsSession.Length);

        int argmaxOracle = ArgMax(logitsOracle);
        int argmaxSession = ArgMax(logitsSession);

        double maxAbsDiff = 0;
        int diffCount = 0;
        for (int i = 0; i < logitsOracle.Length; i++)
        {
            double diff = Math.Abs(logitsOracle[i] - logitsSession[i]);
            if (diff > 0) diffCount++;
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }

        Console.WriteLine(
            $"[DecodePathParity] maxAbsDiff={maxAbsDiff}, {diffCount}/{logitsOracle.Length} lanes differ, " +
            $"argmaxOracle={argmaxOracle}, argmaxSession={argmaxSession}");

        // Measured bit-identical (maxAbsDiff=0, 0/49152 lanes differ). If this regresses,
        // BatchForwardMulti and Forward/ForwardCore stopped agreeing for a single sequence --
        // that would reopen decode as a suspect for the remaining HotSession divergence.
        Assert.Equal(0, diffCount);
        Assert.Equal(0.0, maxAbsDiff);
        Assert.Equal(argmaxOracle, argmaxSession);
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}
