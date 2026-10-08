namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Regression for a NaN found 2026-09-03 while testing ACE-Step's Qwen3 text encoder: on the f16
/// Qwen3-Embedding-0.6B GGUF, this 13-token prompt produced NaN logits at the last position (layer 27), while the
/// Q8_0 file was fine. It no longer reproduces as of 2026-09-27 (docs/103 item 11c): both paths give finite logits,
/// and f16 agrees with Q8_0. This pins it.
/// </summary>
public sealed class Qwen3F16FiniteLogitsTests
{
    private static readonly int[] Prompt = [2, 29051, 198, 14449, 279, 7699, 41733, 6911, 3118, 389, 279, 2661, 4682];

    [Fact]
    public void F16_AceStepPrompt_LogitsFiniteAndMatchQ8_0()
    {
        string? f16 = FindModel("qwen3-embedding-0.6b-f16.gguf");
        string? q8 = FindModel("qwen3-embedding-0.6b-q8_0.gguf");
        if (f16 is null || q8 is null) Assert.Skip("qwen3-embedding-0.6b f16/q8_0 GGUFs not found.");

        float[] prefillF16 = Run(f16!, sequential: false);
        float[] forwardF16 = Run(f16!, sequential: true);
        float[] prefillQ8 = Run(q8!, sequential: false);

        Assert.All(prefillF16, v => Assert.True(float.IsFinite(v)));
        Assert.All(forwardF16, v => Assert.True(float.IsFinite(v)));
        int argF16 = Array.IndexOf(prefillF16, prefillF16.Max());
        Assert.Equal(Array.IndexOf(prefillQ8, prefillQ8.Max()), argF16);
        Assert.Equal(argF16, Array.IndexOf(forwardF16, forwardF16.Max()));
        Assert.InRange(prefillF16.Max() - prefillQ8.Max(), -0.5f, 0.5f);
    }

    private static float[] Run(string path, bool sequential)
    {
        using var model = GgufModel.Open(path);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256);
        if (!sequential) return fwd.Prefill(Prompt).ToArray();
        float[] logits = [];
        for (int i = 0; i < Prompt.Length; i++) logits = fwd.Forward(Prompt[i], i).ToArray();
        return logits;
    }

    private static string? FindModel(string file)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var sub in new[] { Path.Combine("models", "qwen3-embedding-0.6b"), "models", Path.Combine("models", "_models") })
            {
                var candidate = Path.Combine(dir, sub, file);
                if (File.Exists(candidate)) return candidate;
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        return null;
    }
}
