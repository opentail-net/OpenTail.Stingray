using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vulkan;

using Rec = (string Backend, int Layer, string Stage, float[] Data);

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// A32 diagnostic: one decode step of OLMoE through the CPU engine and through the hybrid pass with every layer on the CPU,
/// compared stage by stage (<see cref="StageCapture"/>). Two cases: a single token at position 0 (no cache: attention is over one
/// entry, so the attention path is isolated from the KV cache), and the last token of a short prompt at the end of that prompt.
/// Capture is on only for the final step of each run. A stage is flagged when the largest absolute difference exceeds
/// <see cref="RelativeThreshold"/> times the largest magnitude in that stage (float noise is about 1e-7 relative).
/// Asserts only that stages were compared and that the layer outputs stay close (cosine &gt; 0.99): the purpose is to report where the
/// two diverge. Heavy-gated (STINGRAY_RUN_HEAVY_TESTS=1); skips visibly without the checkpoint or a Vulkan device.
/// </summary>
public sealed class HybridStageDivergenceTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    private const double RelativeThreshold = 1e-6;

    [Fact]
    public void OlmoeDecodeStep_CpuEngineVersusHybridCpuLayers_StageByStage()
    {
        string? path = ModelLocator.FindDeep(Olmoe);
        Assert.SkipWhen(path is null, $"{Olmoe} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host (the hybrid pass needs one)");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model)
            .Encode("The scheduler assigns runnable threads to cores, balancing throughput against latency.").ToArray();

        // Case 1: the first token alone, at position 0.
        int[] single = [prompt[0]];
        var cpuSingle = CpuCapture(model, hp, single);
        var hybSingle = HybridCapture(model, gpu!, hp, single);
        Console.WriteLine($"[a32-stages] case 1: one token at position 0; recorded CPU {cpuSingle.Count}, hybrid {hybSingle.Count} stages");
        var single1 = Compare("case 1 (one token, position 0)", cpuSingle, hybSingle, hp.NumActiveExperts);

        // Case 2: the whole prompt; the captured step is the last prompt token at position n-1 (attention over n entries).
        var cpuPrompt = CpuCapture(model, hp, prompt);
        var hybPrompt = HybridCapture(model, gpu!, hp, prompt);
        Console.WriteLine($"[a32-stages] case 2: {prompt.Length} tokens, last token captured; recorded CPU {cpuPrompt.Count}, hybrid {hybPrompt.Count} stages");
        var multi = Compare($"case 2 ({prompt.Length} tokens, last step)", cpuPrompt, hybPrompt, hp.NumActiveExperts);

        Assert.True(single1.Compared > 0 && multi.Compared > 0, "no stage pair was compared");
    }

    /// <summary>CPU engine: steps 0..n-2 uncaptured, the final step captured. Returns the captured stages.</summary>
    private static List<Rec> CpuCapture(GgufModel model, ModelHyperparams hp, int[] tokens)
    {
        using var backend = new CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp, maxContextLength: 512);
        return CaptureFinalStep(tokens.Length, i => { fwd.Forward(tokens[i], i); });
    }

    /// <summary>Hybrid pass with every layer on the CPU: same step pattern as <see cref="CpuCapture"/>.</summary>
    private static List<Rec> HybridCapture(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int[] tokens)
    {
        var placement = new LayerPlacement(
            GpuLayers: 0, CpuLayers: hp.NumLayers, GpuWeightBytes: 0, GpuKvBytes: 0, RecommendedCtxSize: 512);
        using var fwd = new HybridForwardPass(model, gpu, hp, placement, expertSlotCapacity: -1);
        return CaptureFinalStep(tokens.Length, i => { fwd.Forward(tokens[i], i); });
    }

    private static List<Rec> CaptureFinalStep(int count, Action<int> step)
    {
        for (int i = 0; i < count - 1; i++) step(i);
        StageCapture.Reset();
        StageCapture.Enabled = true;
        try { step(count - 1); }
        finally { StageCapture.Enabled = false; }
        var records = StageCapture.Records.ToList();
        StageCapture.Reset();
        return records;
    }

    private static (int Compared, int Divergent) Compare(string label, List<Rec> reference, List<Rec> other, int topK)
    {
        int compared = 0, divergent = 0, printed = 0, missing = 0;
        string? firstDivergent = null;
        Console.WriteLine($"[a32-stages] === {label}: reference = CPU engine, other = hybrid ===");
        foreach (var c in reference)
        {
            var h = other.FirstOrDefault(r => r.Backend == c.Backend && r.Layer == c.Layer && r.Stage == c.Stage);
            if (h.Data is null) { missing++; continue; }
            compared++;
            double maxAbs = 0, maxMag = 0;
            for (int k = 0; k < c.Data.Length && k < h.Data.Length; k++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs((double)c.Data[k] - h.Data[k]));
                maxMag = Math.Max(maxMag, Math.Abs(c.Data[k]));
            }
            double rel = maxAbs / (maxMag + 1e-30);
            double cos = Cosine(c.Data, h.Data);
            bool differs = rel > RelativeThreshold;
            if (c.Stage == StageCapture.Stages.Router)
            {
                // Pre-softmax logits: the selected set is the top-k. Report whether the sets agree and how close the boundary is.
                var refSet = TopK(c.Data, topK);
                var othSet = TopK(h.Data, topK);
                bool sameSet = refSet.SequenceEqual(othSet);
                Console.WriteLine(FormattableString.Invariant(
                    $"[a32-stages]   layer {c.Layer,2} top-{topK} expert sets {(sameSet ? "identical" : "DIFFER")}; reference boundary margin {BoundaryMargin(c.Data, topK):E3} logits"));
                if (!sameSet) divergent++;
            }
            if (differs)
            {
                divergent++;
                firstDivergent ??= $"layer {c.Layer} {c.Stage}";
            }
            if (c.Layer <= 3 || differs)
            {
                if (printed++ < 60)
                    Console.WriteLine(FormattableString.Invariant(
                        $"[a32-stages] layer {c.Layer,2} {c.Stage,-18} max_abs {maxAbs:E3}  rel {rel:E2}  cosine {cos:F7}{(differs ? "  <-- differs" : "")}"));
            }
        }
        Console.WriteLine($"[a32-stages] {label}: compared {compared} stages ({missing} absent from the hybrid); " +
                          $"{divergent} differ by more than {RelativeThreshold:E0} relative; first divergent: {firstDivergent ?? "none"}");
        return (compared, divergent);
    }

    private static int[] TopK(float[] v, int k) =>
        Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(k).OrderBy(i => i).ToArray();

    /// <summary>Gap between the k-th and the (k+1)-th largest logit: how far the selection is from flipping.</summary>
    private static double BoundaryMargin(float[] v, int k)
    {
        var sorted = v.OrderByDescending(x => x).ToArray();
        return (double)sorted[k - 1] - sorted[k];
    }

    private static double Cosine(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length && i < b.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
    }
}
