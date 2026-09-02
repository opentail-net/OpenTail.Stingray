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

        Console.WriteLine($"Wq dtype: {model.FindTensor("blk.0.attn_q.weight")?.DType}");
        Console.WriteLine($"Wo dtype: {model.FindTensor("blk.0.attn_output.weight")?.DType}");
        Console.WriteLine($"GateInp dtype: {model.FindTensor("blk.0.ffn_gate_inp.weight")?.DType}");
        Console.WriteLine($"GateExps dtype: {model.FindTensor("blk.0.ffn_gate_exps.weight")?.DType}");
        Console.WriteLine($"DownExps dtype: {model.FindTensor("blk.0.ffn_down_exps.weight")?.DType}");
        Console.WriteLine($"Output dtype: {model.FindTensor("output.weight")?.DType}");

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

    /// <summary>
    /// Diagnostic, not a parity receipt (no known-correct reference completion exists to assert
    /// against yet — see docs/060-...md Phase 4). Tokenizes a real prompt (exercising the new
    /// "gpt-4o" pre-tokenizer cascade added to PreTokenizerPatterns.cs this session, confirmed
    /// against this checkpoint's real tokenizer.ggml.pre value) and greedy-decodes a few tokens,
    /// printing the completion for a human to eyeball. Only weak invariants are asserted (no
    /// crash, no immediate degenerate single-token repetition) since there's nothing stronger to
    /// check against yet.
    /// </summary>
    [Fact]
    public void GptOss_TokenizesAndGreedyDecodes_ForEyeballing()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this smoke test.");

        using var model = GgufModel.Open(path!);
        var metadata = model.Metadata;
        const string arch = "gpt-oss";

        Assert.Equal("gpt-4o", Convert.ToString(metadata["tokenizer.ggml.pre"]));

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var promptTokens = tokenizer.Encode("The capital of France is");
        Assert.True(promptTokens.Count > 0);

        int numLayer = Convert.ToInt32(metadata[$"{arch}.block_count"]);
        int embedDim = Convert.ToInt32(metadata[$"{arch}.embedding_length"]);
        int numHeads = Convert.ToInt32(metadata[$"{arch}.attention.head_count"]);
        int numHeadsKv = Convert.ToInt32(metadata[$"{arch}.attention.head_count_kv"]);
        int headDim = Convert.ToInt32(metadata[$"{arch}.attention.key_length"]);
        int numExperts = Convert.ToInt32(metadata[$"{arch}.expert_count"]);
        int numExpertsUsed = Convert.ToInt32(metadata[$"{arch}.expert_used_count"]);
        int vocabSize = (int)model.FindTensor("output.weight")!.Value.Dimensions[1];
        var hp = GptOssHyperparams.FromGgufMetadata(
            metadata, arch, numLayer, embedDim, numHeads, numHeadsKv, headDim,
            numExperts, numExpertsUsed, vocabSize);

        using var fwd = new GptOssForwardPass(model, hp);

        ReadOnlySpan<float> logits = default;
        int pos = 0;
        foreach (int t in promptTokens)
        {
            logits = fwd.Forward(t, pos++);
        }

        var generated = new List<int>();
        for (int i = 0; i < 12; i++)
        {
            int next = Argmax(logits);
            generated.Add(next);
            logits = fwd.Forward(next, pos++);
        }

        string continuation = tokenizer.Decode(generated.ToArray());
        Console.WriteLine($"Prompt: The capital of France is");
        Console.WriteLine($"Continuation ({generated.Count} tokens): {continuation}");
        Console.WriteLine($"Token ids: {string.Join(", ", generated)}");

        Assert.False(generated.TrueForAll(t => t == generated[0]), "degenerate: every generated token identical");
    }

    [Fact]
    public void GptOss_Benchmark_DecodeSpeed()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this smoke test.");

        using var model = GgufModel.Open(path!);
        var metadata = model.Metadata;
        const string arch = "gpt-oss";

        int numLayer = Convert.ToInt32(metadata[$"{arch}.block_count"]);
        int embedDim = Convert.ToInt32(metadata[$"{arch}.embedding_length"]);
        int numHeads = Convert.ToInt32(metadata[$"{arch}.attention.head_count"]);
        int numHeadsKv = Convert.ToInt32(metadata[$"{arch}.attention.head_count_kv"]);
        int headDim = Convert.ToInt32(metadata[$"{arch}.attention.key_length"]);
        int numExperts = Convert.ToInt32(metadata[$"{arch}.expert_count"]);
        int numExpertsUsed = Convert.ToInt32(metadata[$"{arch}.expert_used_count"]);
        int vocabSize = (int)model.FindTensor("output.weight")!.Value.Dimensions[1];
        var hp = GptOssHyperparams.FromGgufMetadata(
            metadata, arch, numLayer, embedDim, numHeads, numHeadsKv, headDim,
            numExperts, numExpertsUsed, vocabSize);

        using var fwd = new GptOssForwardPass(model, hp);

        // Warmup (1 token)
        var logits = fwd.Forward(100, 0);

        int decodeTokens = 16;
        long bytesBefore = GC.GetAllocatedBytesForCurrentThread();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var tokenSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < decodeTokens; i++)
        {
            tokenSw.Restart();
            int next = Argmax(logits);
            logits = fwd.Forward(next, 1 + i);
            Console.WriteLine($"Token {i + 1}/{decodeTokens} (id={next}): {tokenSw.Elapsed.TotalMilliseconds:F1} ms");
        }
        sw.Stop();
        long bytesAfter = GC.GetAllocatedBytesForCurrentThread();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        double msPerToken = totalMs / decodeTokens;
        double tokPerSec = decodeTokens / sw.Elapsed.TotalSeconds;
        long allocBytesPerToken = (bytesAfter - bytesBefore) / decodeTokens;

        Console.WriteLine($"\n=======================================================");
        Console.WriteLine($"[GPT-OSS 20B MXFP4 CPU SPEED] Sustained Speed: {tokPerSec:F2} T/S ({msPerToken:F2} ms/tok)");
        Console.WriteLine($"[GPT-OSS 20B MXFP4 CPU SPEED] Total: {totalMs:F2} ms for {decodeTokens} tokens | Alloc: {allocBytesPerToken / 1024.0:F1} KB/token");
        Console.WriteLine($"=======================================================\n");
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best;
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
