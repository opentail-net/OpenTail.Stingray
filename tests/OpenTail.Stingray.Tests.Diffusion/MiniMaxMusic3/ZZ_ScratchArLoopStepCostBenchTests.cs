using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Scratch: isolates the real per-frame cost of the two components run 200x in the AR loop
/// (Global LM's ForwardIncrementalStepPair, RVQ depth decoder's per-frame 7-step CFG loop) with
/// real weights, to find out which actually dominates the real 200-frame generation's wall clock --
/// the DiT/Flow stage was already measured (ZZ_ScratchDitForwardPairFullScaleBenchTests, ~31.7s per
/// Euler step at T=689) and found to be roughly on par with the real minimaxmusic.cpp C++
/// reference, so the AR loop is the next real suspect for the ~100s+ gap against that reference's
/// 168.3s AR-loop total (docs/066, 2026-09-05). NOT a golden-parity check. Delete once superseded.
/// </summary>
public sealed class ZZ_ScratchArLoopStepCostBenchTests
{
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
    public void ArLoop_RealWeights_MeasuresPerFrameCost()
    {
        string? langDir = FindRepoDir("models/minimax-music3/language_model");
        string? depthPath = FindRepoFile("models/minimax-music3/rvq_depth_decoder.safetensors");
        Assert.SkipUnless(langDir != null && depthPath != null, "minimax-music3 LM/depth weights not found");

        using var langLoader = SafetensorsLoader.OpenDirectory(langDir!);
        using var globalModel = new MiniMaxMusic3GlobalModel(langLoader);
        using var depthLoader = SafetensorsLoader.Open(depthPath!);
        var depthWeights = MiniMaxMusic3RvqDepthDecoderWeights.Load(depthLoader);

        int hidden = MiniMaxMusic3Config.LanguageModelHiddenSize;
        int numLayers = MiniMaxMusic3Config.LanguageModelNumLayers;

        // Real prompt-shaped prefill (23 tokens, matching the actual real prompt length used
        // by ZZ_ScratchMiniMaxMusic3GenerateSampleTests/the real mm3cpp run) to reach steady AR state.
        var promptEncoder = MiniMaxMusic3PromptEncoder.Load(FindRepoDir("models/minimax-music3/tokenizer")!);
        int[] promptTokens = promptEncoder.BuildConditionalPrompt(
            "Intimate acoustic folk, male vocal, fingerpicked guitar",
            "[Verse]\nWalking through the morning rain");
        var unconditionalPromptTokens = (int[])promptTokens.Clone();
        for (int i = 1; i < unconditionalPromptTokens.Length - 2; i++) unconditionalPromptTokens[i] = MiniMaxMusic3Config.AudioCfgTokenId;

        var condCache = new MiniMaxMusic3GlobalKvCache(numLayers);
        var uncondCache = new MiniMaxMusic3GlobalKvCache(numLayers);
        var (condHiddenSeq, _) = globalModel.ForwardIncremental(promptTokens, condCache);
        var (_, _) = globalModel.ForwardIncremental(unconditionalPromptTokens, uncondCache);
        var random = new Random(42);

        var feedback = new float[hidden];
        for (int i = 0; i < hidden; i++) feedback[i] = (float)random.NextDouble() - 0.5f;

        // Warmup, then measure a handful of real steady-state incremental steps.
        globalModel.ForwardIncrementalStepPair(feedback, feedback, condCache, uncondCache);

        const int lmSteps = 5;
        var swLm = Stopwatch.StartNew();
        for (int i = 0; i < lmSteps; i++)
            globalModel.ForwardIncrementalStepPair(feedback, feedback, condCache, uncondCache);
        swLm.Stop();
        double lmMsPerStep = swLm.Elapsed.TotalMilliseconds / lmSteps;

        // Depth decoder: real per-frame cost is 7 residual-codebook CFG steps (2 branches each,
        // sequence growing 2..8) -- matches MiniMaxMusic3AutoregressiveGenerator's real per-frame loop.
        var condDepthCache = new MiniMaxMusic3RvqDepthKvCache();
        var uncondDepthCache = new MiniMaxMusic3RvqDepthKvCache();
        var semanticEmbed = new float[hidden];
        for (int i = 0; i < hidden; i++) semanticEmbed[i] = (float)random.NextDouble() - 0.5f;
        var projectedSemantic = MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, semanticEmbed);
        var condLastHidden = feedback;
        var uncondLastHidden = feedback;

        void RunOneDepthFrame()
        {
            condDepthCache.Reset();
            uncondDepthCache.Reset();
            MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, condLastHidden), 0, condDepthCache);
            MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, uncondLastHidden), 0, uncondDepthCache);
            var condLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, projectedSemantic, 1, condDepthCache);
            var uncondLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, projectedSemantic, 1, uncondDepthCache);
            for (int ci = 0; ci < 6; ci++)
            {
                var embedded = MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, MiniMaxMusic3RvqDepthDecoder.EmbedResidualCode(depthWeights, ci, 0));
                condLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, embedded, ci + 2, condDepthCache);
                uncondLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, embedded, ci + 2, uncondDepthCache);
            }
        }

        RunOneDepthFrame(); // warmup
        const int depthFrames = 10;
        var swDepth = Stopwatch.StartNew();
        for (int i = 0; i < depthFrames; i++) RunOneDepthFrame();
        swDepth.Stop();
        double depthMsPerFrame = swDepth.Elapsed.TotalMilliseconds / depthFrames;

        // Paired (CFG-batched, single-weight-stream) variant -- same real work, new code path.
        var condDepthCacheB = new MiniMaxMusic3RvqDepthKvCache();
        var uncondDepthCacheB = new MiniMaxMusic3RvqDepthKvCache();

        void RunOneDepthFramePaired()
        {
            condDepthCacheB.Reset();
            uncondDepthCacheB.Reset();
            MiniMaxMusic3RvqDepthDecoder.ForwardStepPair(depthWeights,
                MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, condLastHidden),
                MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, uncondLastHidden),
                0, condDepthCacheB, uncondDepthCacheB);
            MiniMaxMusic3RvqDepthDecoder.ForwardStepPair(depthWeights, projectedSemantic, projectedSemantic, 1, condDepthCacheB, uncondDepthCacheB);
            for (int ci = 0; ci < 6; ci++)
            {
                var embedded = MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, MiniMaxMusic3RvqDepthDecoder.EmbedResidualCode(depthWeights, ci, 0));
                MiniMaxMusic3RvqDepthDecoder.ForwardStepPair(depthWeights, embedded, embedded, ci + 2, condDepthCacheB, uncondDepthCacheB);
            }
        }

        RunOneDepthFramePaired(); // warmup
        var swDepthPaired = Stopwatch.StartNew();
        for (int i = 0; i < depthFrames; i++) RunOneDepthFramePaired();
        swDepthPaired.Stop();
        double depthPairedMsPerFrame = swDepthPaired.Elapsed.TotalMilliseconds / depthFrames;

        Console.WriteLine($"[bench] Global LM ForwardIncrementalStepPair: {lmMsPerStep:F1} ms/step");
        Console.WriteLine($"[bench] Depth decoder per-frame (7 separate CFG steps, BEFORE): {depthMsPerFrame:F1} ms/frame");
        Console.WriteLine($"[bench] Depth decoder per-frame (7 CFG-batched ForwardStepPair steps, AFTER): {depthPairedMsPerFrame:F1} ms/frame");
        Console.WriteLine($"[bench] Depth decoder speedup: {(depthMsPerFrame / depthPairedMsPerFrame):F2}x");
        Console.WriteLine($"[bench] Combined AR per-frame estimate: {(lmMsPerStep + depthMsPerFrame):F1} ms/frame -> {(lmMsPerStep + depthMsPerFrame) * 200 / 1000.0:F1} s for 200 frames");
        Console.WriteLine($"[bench] Real minimaxmusic.cpp reference: 841.3 ms/frame -> 168.3 s for 200 frames");
    }
}
