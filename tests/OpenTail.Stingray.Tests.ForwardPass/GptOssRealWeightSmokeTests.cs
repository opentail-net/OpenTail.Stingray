using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// FIRST real-weight check for the alpha gpt-oss port (GptOssForwardPass.cs,
/// docs/060-gpt-oss-implementation-plan.md Phase 3). This is deliberately a SMOKE test, not a
/// parity receipt: gpt-oss is not admitted to <see cref="ModelCompatibility"/> and this
/// intentionally bypasses that gate by constructing <see cref="GptOssForwardPass"/> directly
/// (the same alpha class the plan doc describes, not the shared <c>Engine.ForwardPass</c> every
/// admitted architecture uses) — a pass here means "loads and produces finite, non-crashing
/// output," NOT "produces correct output." No token-level comparison against any reference
/// exists yet; that needs an actual llama.cpp/llama-eval-callback run against this same
/// checkpoint, not done this session.
/// </summary>
public sealed class GptOssRealWeightSmokeTests : HeavyTestBase
{
    private const string ModelFile = "gpt-oss-20b-MXFP4.gguf";

    [Fact]
    public void GptOss_LoadsAndProducesFiniteLogits_ForOneToken()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this smoke test.");

        using var model = GgufModel.Open(path!);
        var metadata = model.Metadata;

        Assert.Equal("gpt-oss", Convert.ToString(metadata["general.architecture"]));

        const string arch = "gpt-oss";
        int numLayer = Convert.ToInt32(metadata[$"{arch}.block_count"]);
        int embedDim = Convert.ToInt32(metadata[$"{arch}.embedding_length"]);
        int numHeads = Convert.ToInt32(metadata[$"{arch}.attention.head_count"]);
        int numHeadsKv = Convert.ToInt32(metadata[$"{arch}.attention.head_count_kv"]);
        int headDim = Convert.ToInt32(metadata[$"{arch}.attention.key_length"]);
        int numExperts = Convert.ToInt32(metadata[$"{arch}.expert_count"]);
        int numExpertsUsed = Convert.ToInt32(metadata[$"{arch}.expert_used_count"]);
        int vocabSize = (int)model.FindTensor("output.weight")!.Value.Dimensions[1];

        // Record the real checkpoint's shape once, since docs/060-...md's specific numbers (32
        // experts, top-4, hidden 2880, 64/8 heads) came from an external, unverified source --
        // this is the first time they're checked against the actual file.
        Assert.True(numLayer is 24 or 36, $"unexpected layer count {numLayer} (reference only recognizes 24=20B/36=120B)");

        var hp = GptOssHyperparams.FromGgufMetadata(
            metadata, arch, numLayer, embedDim, numHeads, numHeadsKv, headDim,
            numExperts, numExpertsUsed, vocabSize);

        using var fwd = new GptOssForwardPass(model, hp);

        // "The capital of France is" tokenized would need a real tokenizer wired for this arch,
        // not part of this smoke test's scope -- use a fixed low token id instead, which is
        // sufficient to exercise every layer's real weights end to end.
        var logits = fwd.Forward(token: 100, position: 0);

        Assert.Equal(vocabSize, logits.Length);
        bool anyNonZero = false;
        foreach (float v in logits)
        {
            Assert.False(float.IsNaN(v), "logit was NaN");
            Assert.False(float.IsInfinity(v), "logit was infinite");
            if (v != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, "all logits were exactly zero -- suspicious, likely a wiring bug");
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(candidate)) return candidate;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        var external = Path.Combine(@"E:\models", ModelFile);
        return File.Exists(external) ? external : null;
    }
}
