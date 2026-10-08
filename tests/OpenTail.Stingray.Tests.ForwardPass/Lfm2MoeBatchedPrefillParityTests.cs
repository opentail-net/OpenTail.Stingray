using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// LFM2-MoE batched prefill vs token-by-token on the WikiText prefix. The attention-layer drift seen
/// at 256+ tokens is the 64-wide flash attention's online softmax (not bit-identical by design);
/// with <c>STINGRAY_PREFILL_ATTN_FLASH64=0</c> the batched trunk is bit-identical, which the second
/// test pins.
/// </summary>
public sealed class Lfm2MoeBatchedPrefillParityTests : HeavyTestBase
{
    [Fact]
    public void Lfm2Moe_DefaultPrefillMatchesTokenByTokenLogits()
    {
        string? modelPath = FindModel();
        Assert.SkipUnless(modelPath is not null, "LFM2-8B-A1B-Q4_K_M.gguf is required.");

        string repoRoot = FindRepoRoot();
        string corpusPath = Path.Combine(repoRoot, "scripts", "kvarn-gate", "wiki.test.raw");
        Assert.SkipUnless(File.Exists(corpusPath), "scripts/kvarn-gate/wiki.test.raw is required.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(modelPath!);
        var model = modelHandle.Model;
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.True(hp.IsMoE && hp.IsShortConvLayer is not null, "expected an LFM2-MoE short-convolution model");

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var tokens = tokenizer.Encode(File.ReadAllText(corpusPath)).ToList();
        if (tokenizer.AddBosToken && tokenizer.BosTokenId >= 0
            && (tokens.Count == 0 || tokens[0] != tokenizer.BosTokenId))
            tokens.Insert(0, tokenizer.BosTokenId);
        int[] prefix = tokens.Take(256).ToArray();
        Assert.Equal(256, prefix.Length);

        bool prevRecurrent = Engine.ForwardPass.RecurrentBatchedPrefillEnabled;
        bool prevLfm2Batch = Engine.ForwardPass.Lfm2MoeBatchedPrefillEnabled;
        bool prevQ8 = SimdKernels.Q8PrefillEnabled;
        bool prevMoeQ8 = Engine.MoeBatchedExperts.Q8PrefillEnabled;
        int prevBlas = SimdKernels.MinBatchForBlas;
        try
        {
            // Exercise the production-default gate even when the enclosing test process was started
            // with an experimental environment override. The global recurrent switch remains on.
            Engine.ForwardPass.RecurrentBatchedPrefillEnabled = true;
            Engine.ForwardPass.Lfm2MoeBatchedPrefillEnabled = false;
            SimdKernels.Q8PrefillEnabled = false;
            Engine.MoeBatchedExperts.Q8PrefillEnabled = false;
            SimdKernels.MinBatchForBlas = int.MaxValue;

            using var backend = new CpuBackend();
            using var safePrefill = new Engine.ForwardPass(model, backend, hp, maxContextLength: prefix.Length + 1);
            using var tokenByToken = new Engine.ForwardPass(model, backend, hp, maxContextLength: prefix.Length + 1);

            safePrefill.PrefillWithPerPositionLogits(prefix, 0, (position, logits) =>
            {
                var expected = tokenByToken.Forward(prefix[position], position);
                Assert.Equal(expected.Length, logits.Length);
                for (int i = 0; i < logits.Length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(logits[i]))
                        Assert.Fail($"Position {position}, logit {i} differs: token-by-token {expected[i]:R}, prefill {logits[i]:R}.");
                }
            });
        }
        finally
        {
            Engine.ForwardPass.RecurrentBatchedPrefillEnabled = prevRecurrent;
            Engine.ForwardPass.Lfm2MoeBatchedPrefillEnabled = prevLfm2Batch;
            SimdKernels.Q8PrefillEnabled = prevQ8;
            Engine.MoeBatchedExperts.Q8PrefillEnabled = prevMoeQ8;
            SimdKernels.MinBatchForBlas = prevBlas;
        }
    }

    [Fact]
    public void Lfm2Moe_BatchedPrefillWithoutFlash64_MatchesTokenByTokenLogits()
    {
        string? modelPath = FindModel();
        Assert.SkipUnless(modelPath is not null, "LFM2-8B-A1B-Q4_K_M.gguf is required.");

        string repoRoot = FindRepoRoot();
        string corpusPath = Path.Combine(repoRoot, "scripts", "kvarn-gate", "wiki.test.raw");
        Assert.SkipUnless(File.Exists(corpusPath), "scripts/kvarn-gate/wiki.test.raw is required.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(modelPath!);
        var model = modelHandle.Model;
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var tokens = tokenizer.Encode(File.ReadAllText(corpusPath)).ToList();
        if (tokenizer.AddBosToken && tokenizer.BosTokenId >= 0
            && (tokens.Count == 0 || tokens[0] != tokenizer.BosTokenId))
            tokens.Insert(0, tokenizer.BosTokenId);
        int[] prefix = tokens.Take(256).ToArray();

        bool prevRecurrent = Engine.ForwardPass.RecurrentBatchedPrefillEnabled;
        bool prevLfm2Batch = Engine.ForwardPass.Lfm2MoeBatchedPrefillEnabled;
        bool prevQ8 = SimdKernels.Q8PrefillEnabled;
        bool prevMoeQ8 = Engine.MoeBatchedExperts.Q8PrefillEnabled;
        int prevBlas = SimdKernels.MinBatchForBlas;
        string? prevFlash64 = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_FLASH64");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_PREFILL_ATTN_FLASH64", "0");
            Engine.ForwardPass.RecurrentBatchedPrefillEnabled = true;
            Engine.ForwardPass.Lfm2MoeBatchedPrefillEnabled = true;
            SimdKernels.Q8PrefillEnabled = false;
            Engine.MoeBatchedExperts.Q8PrefillEnabled = false;
            SimdKernels.MinBatchForBlas = int.MaxValue;

            using var backend = new CpuBackend();
            using var batchedPrefill = new Engine.ForwardPass(model, backend, hp, maxContextLength: prefix.Length + 1);
            using var tokenByToken = new Engine.ForwardPass(model, backend, hp, maxContextLength: prefix.Length + 1);

            batchedPrefill.PrefillWithPerPositionLogits(prefix, 0, (position, logits) =>
            {
                var expected = tokenByToken.Forward(prefix[position], position);
                for (int i = 0; i < logits.Length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(logits[i]))
                        Assert.Fail($"Position {position}, logit {i} differs: token-by-token {expected[i]:R}, batched {logits[i]:R}.");
                }
            });
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_PREFILL_ATTN_FLASH64", prevFlash64);
            Engine.ForwardPass.RecurrentBatchedPrefillEnabled = prevRecurrent;
            Engine.ForwardPass.Lfm2MoeBatchedPrefillEnabled = prevLfm2Batch;
            SimdKernels.Q8PrefillEnabled = prevQ8;
            Engine.MoeBatchedExperts.Q8PrefillEnabled = prevMoeQ8;
            SimdKernels.MinBatchForBlas = prevBlas;
        }
    }

    private static string? FindModel()
    {
        string path = Path.Combine(FindRepoRoot(), "models", "_models", "LFM2-8B-A1B-Q4_K_M.gguf");
        return File.Exists(path) ? path : null;
    }

    private static string FindRepoRoot()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "src", "OpenTail.Stingray.Engine"))) return dir;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        throw new DirectoryNotFoundException("Could not locate the repository root.");
    }
}
