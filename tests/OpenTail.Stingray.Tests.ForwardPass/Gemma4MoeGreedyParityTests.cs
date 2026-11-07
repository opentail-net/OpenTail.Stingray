using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Plan 6 F1: Gemma 4 26B-A4B (parallel dense FFN plus 128 routed experts with a fused <c>ffn_gate_up_exps</c>) on the CPU forward pass. The expected texts
/// are stock llama.cpp's greedy output for the same chat-formatted prompts (b10306 and b11556 agree; <c>llama-completion -no-cnv --temp 0 -ngl 0</c>,
/// recorded 2026-11-03), so this is a comparison with an independent implementation, not a snapshot of this engine. Only short answers are pinned: the
/// router picks 8 of 128 experts per layer, and a near-tie can flip a token later in a long generation (measured the same way: the engine differs from itself
/// as much under another valid kernel path as it does from llama.cpp). Real weights, 17 GB; skipped when the checkpoint is absent.
/// </summary>
public sealed class Gemma4MoeGreedyParityTests : HeavyTestBase
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("google_gemma-4-26B-A4B-it-Q4_K_M.gguf");

    [Theory]
    [InlineData("What is the capital of France? Answer in one short sentence.", "The capital of France is Paris.")]
    [InlineData("Write a haiku about the sea.", "Blue waves kiss the shore,\nRhythmic songs of salt and sand,\nEndless deep expanse.")]
    public void GreedyChatAnswer_MatchesLlamaCpp(string question, string expected)
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "gemma-4-26B-A4B-it-Q4_K_M.gguf is required");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.True(hp.ParallelDenseMoe);
        // Confirmed against llama.cpp's real src/models/gemma4.cpp (build_moe_ffn(..., norm_w=true, ...))
        // that Gemma 4 DOES renormalize its selected top-k router weights -- see
        // Gemma4ModelSemantics.cs's NormalizeMoeTopKWeights comment.
        Assert.True(hp.NormalizeMoeTopKWeights);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        string prompt = $"<|turn>user\n{question}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        var ids = new List<int>();
        if (tokenizer.AddBosToken) ids.Add(tokenizer.BosTokenId);
        ids.AddRange(tokenizer.Encode(prompt));

        using var backend = new CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp);
        fwd.ResetCache();
        var logits = fwd.Prefill(ids.ToArray());
        int token = 0; float best = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > best) { best = logits[i]; token = i; }

        var generated = new List<int>();
        int pos = ids.Count;
        for (int n = 0; n < 40 && !tokenizer.EogTokenIds.Contains(token); n++)
        {
            generated.Add(token);
            (token, _) = ((IForwardPass)fwd).ForwardArgmax(token, pos++);
        }

        Assert.Equal(expected, tokenizer.Decode(generated).Trim());
    }

    /// <summary>
    /// Plan 6 F3: batched prefill of the parallel dense+MoE layers (experts bucketed per expert, exact F32 batched GEMMs) must agree with the token-by-token
    /// path on the next-token distribution: same argmax and a cosine similarity of the logits above 0.999. They are not bit-identical (the sequential path
    /// quantises activations to Q8_K as ggml does, the batched expert GEMMs are F32), so a threshold, not equality.
    /// </summary>
    [Fact]
    public void BatchedPrefill_AgreesWithTokenByTokenForward()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "gemma-4-26B-A4B-it-Q4_K_M.gguf is required");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var ids = new List<int> { tokenizer.BosTokenId };
        ids.AddRange(tokenizer.Encode(
            "<|turn>user\nExplain in three sentences how a binary search works.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>" +
            "Binary search begins by comparing the target value to the middle element of a sorted list. If the target is not the middle element, the algorithm"));

        using var backend = new CpuBackend();
        using var batched = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp);
        float[] a = batched.Prefill(ids.ToArray()).ToArray();

        using var sequential = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp);
        float[] b = [];
        for (int i = 0; i < ids.Count; i++) b = sequential.Forward(ids[i], i).ToArray();

        Assert.Equal(Array.IndexOf(b, b.Max()), Array.IndexOf(a, a.Max()));
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        Assert.True(dot / Math.Sqrt(na * nb) > 0.999, $"cosine {dot / Math.Sqrt(na * nb):F5}");
    }
}
