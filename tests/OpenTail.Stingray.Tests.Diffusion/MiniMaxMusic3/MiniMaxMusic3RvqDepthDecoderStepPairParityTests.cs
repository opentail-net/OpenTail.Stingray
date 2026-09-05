using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real parity check: <see cref="MiniMaxMusic3RvqDepthDecoder.ForwardStepPair"/> (the CFG-batched,
/// single-weight-stream incremental step added for the AR-loop performance pass, see
/// docs/066-minimax-music3-future-plan.md "AR loop, 2026-09-05") must match two separate
/// <see cref="MiniMaxMusic3RvqDepthDecoder.ForwardStep"/> calls (one per CFG branch) to within tight
/// floating-point tolerance -- NOT literal bit-for-bit, since the new path's
/// <see cref="OpenTail.Stingray.Audio.Primitives.CfmLinearWeight.MatMulPairRowMajor"/> reduces each
/// row with a plain `DotF32` while the original path's <c>SimdKernels.MatVecF32</c> reduces 4 rows
/// together via a different (equally valid) interleaved SIMD accumulation order -- the same kind of
/// unavoidable floating-point non-associativity this project already tolerates at `1e-4f` for the
/// analogous Q8 `ForwardPair`-vs-`Forward` DiT parity check. The whole point of this change is a
/// memory-bandwidth/dispatch-count optimization, not a numeric one, so a divergence bigger than
/// float-reordering noise here is still a real bug.
/// </summary>
public sealed class MiniMaxMusic3RvqDepthDecoderStepPairParityTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ForwardStepPair_MatchesTwoForwardStepCalls_BitForBit()
    {
        string? depthPath = FindRepoFile("models/minimax-music3/rvq_depth_decoder.safetensors");
        Assert.SkipUnless(depthPath != null, "models/minimax-music3/rvq_depth_decoder.safetensors not found");

        using var loader = SafetensorsLoader.Open(depthPath!);
        var w = MiniMaxMusic3RvqDepthDecoderWeights.Load(loader);

        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;
        var random = new Random(123);
        float[] RandomVec()
        {
            var v = new float[hidden];
            for (int i = 0; i < hidden; i++) v[i] = (float)random.NextDouble() - 0.5f;
            return v;
        }

        // Real per-frame shape: step 0 (LM hidden), step 1 (semantic embed), steps 2..7 (residual
        // embeds) -- run all 8 real steps through both call shapes and compare every one.
        var condCacheA = new MiniMaxMusic3RvqDepthKvCache();
        var uncondCacheA = new MiniMaxMusic3RvqDepthKvCache();
        var condCacheB = new MiniMaxMusic3RvqDepthKvCache();
        var uncondCacheB = new MiniMaxMusic3RvqDepthKvCache();

        for (int step = 0; step < 8; step++)
        {
            var condInput = RandomVec();
            var uncondInput = RandomVec();

            var condOut = MiniMaxMusic3RvqDepthDecoder.ForwardStep(w, condInput, step, condCacheA);
            var uncondOut = MiniMaxMusic3RvqDepthDecoder.ForwardStep(w, uncondInput, step, uncondCacheA);

            var (pairCond, pairUncond) = MiniMaxMusic3RvqDepthDecoder.ForwardStepPair(w, condInput, uncondInput, step, condCacheB, uncondCacheB);

            for (int i = 0; i < hidden; i++)
            {
                Assert.Equal(condOut[i], pairCond[i], 1e-4f);
                Assert.Equal(uncondOut[i], pairUncond[i], 1e-4f);
            }
        }
    }
}
