using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Vulkan-vs-CPU parity at the logit level, which is the measurement that actually settles whether
/// the two backends agree.
///
/// <para><b>Why not compare generated text.</b> An end-to-end greedy run diverged from CPU at
/// roughly token 12 ("the operating system" vs "the system"). That is not evidence of a defect:
/// greedy decoding turns one argmax flip into wholly different downstream text, so any two backends
/// with different floating-point orderings will diverge eventually. Nor is it evidence of
/// correctness. Only the logits distinguish the two, so that is what this asserts — the same lesson
/// the Flash 128/256 decision produced, where a plausible-looking output masked a real perplexity
/// shift and a per-prompt check masked a corpus one.</para>
///
/// <para>Tolerances are deliberately asymmetric. The <b>argmax must match exactly</b> — that is the
/// decision the sampler actually makes at greedy, and a mismatch means the backends would produce
/// different text from the very first token. Cosine is held at an EMPIRICAL floor of 0.99 — see
/// the assertion for why that number is a regression guard rather than an endorsement.</para>
/// </summary>
public sealed class VulkanCpuLogitParityTests
{
    private static readonly int[] Prompt =
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

    [Fact]
    public void VulkanPrefillLogits_AgreeWithCpu_OnArgmaxAndDirection()
    {
        string? path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2 GGUF not present");
        using var gpu = TryCreateBackend();
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        float[] cpu;
        using (var backend = new CpuBackend())
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512))
            cpu = fwd.Prefill(Prompt).ToArray();

        float[] vulkan;
        using (var fwd = new GpuForwardPass(model, gpu!, hp, maxContextLength: 512))
            vulkan = fwd.Prefill(Prompt).ToArray();

        Assert.Equal(cpu.Length, vulkan.Length);

        int cpuArgmax = Argmax(cpu), vulkanArgmax = Argmax(vulkan);
        double cos = Cosine(cpu, vulkan);
        float maxAbs = MaxAbs(cpu, vulkan);

        // Reported on failure so a regression says how far apart the backends are, not merely that
        // they differ — the difference between a diagnosable result and "Vulkan is broken".
        Assert.True(cpuArgmax == vulkanArgmax,
            $"argmax differs: CPU {cpuArgmax} vs Vulkan {vulkanArgmax} (cos={cos:F8}, maxAbs={maxAbs:F4}). " +
            "The backends would produce different text from the first generated token.");
        // EMPIRICAL floor, not a principled one. Measured 2026-08-07: cos = 0.99195, maxAbs = 1.476
        // on SmolLM2-1.7B-Q4_K_M. That is roughly 12x looser than the CPU-side approximations this
        // repo already accepts (int8 activation prefill 0.999504, Flash-128 0.999345), and NO cause
        // has been established: the FP16 narrowed-KV store is opt-in via STINGRAY_KV_DTYPE and was
        // not in use here.
        //
        // 0.99 guards the measured baseline against regression without endorsing 0.992 as correct.
        // It was deliberately NOT tightened to whatever makes today's number pass, and should be
        // tightened only once the divergence is explained - if it turns out to be a defect, a
        // threshold fitted to it would have hidden that permanently.
        Assert.True(cos > 0.99,
            $"logit cosine {cos:F8} below the 0.99 empirical floor (maxAbs={maxAbs:F4}, " +
            $"argmax CPU {cpuArgmax} / Vulkan {vulkanArgmax}).");
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static VulkanBackend? TryCreateBackend()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string? dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }
}
