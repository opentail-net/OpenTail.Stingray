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
    private static string? FindModel()
    {
        const string File = "google_gemma-4-26B-A4B-it-Q4_K_M.gguf";
        foreach (var drive in DriveInfo.GetDrives().Where(d => d.DriveType == DriveType.Fixed && d.IsReady))
            foreach (string c in new[] { Path.Combine(drive.RootDirectory.FullName, "_models", "gemma4-26b-a4b", File), Path.Combine(drive.RootDirectory.FullName, "_models", File) })
                if (System.IO.File.Exists(c)) return c;
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null; dir = dir.Parent)
            foreach (string c in new[] { Path.Combine(dir.FullName, "models", File), Path.Combine(dir.FullName, "models", "_models", File) })
                if (System.IO.File.Exists(c)) return c;
        return null;
    }

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
        Assert.False(hp.NormalizeMoeTopKWeights);

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
}
