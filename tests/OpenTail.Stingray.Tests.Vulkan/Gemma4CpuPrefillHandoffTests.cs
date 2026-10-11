namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Receipt test for the Gemma 4 CPU-prefill KV handoff on the full-offload Vulkan pass (<c>GpuPrefillHandoff.cs</c>): a fresh long prompt is prefilled by the CPU
/// batched pass and each KV-owning layer's K/V (head_dim 256 on sliding-window layers, 512 on global layers; the shared-KV tail layers alias their source) is
/// uploaded to the GPU caches. This is the end-to-end check that every layer's rows landed at the right stride, because attention reads garbage if they did not:
/// teacher-forced decode after the handoff must agree with decode after the sequential GPU prefill (cosine above 0.99, argmax flips only on a near-tie).
/// Real weights and a real Vulkan device; skips visibly without either.
/// </summary>
public sealed class Gemma4CpuPrefillHandoffTests : HeavyTestBase
{
    private const string File = "gemma-4-E4B-it-Q4_K_M.gguf";

    private const string Text =
        "In 1843 Ada Lovelace published notes describing how a machine could manipulate symbols according to rules, " +
        "an idea that reached far beyond arithmetic. A century later wartime codebreakers built electronic devices to " +
        "search enormous spaces of possibilities, and stored-program computers followed within a few years. Transistors " +
        "replaced vacuum tubes, integrated circuits replaced transistors, and the cost of a calculation fell by many " +
        "orders of magnitude. Operating systems learned to share one processor among many programs; networks joined " +
        "machines across continents; databases promised that facts written once could be read correctly forever. Today " +
        "language models trained on vast text collections can summarise, translate and reason about problems that no " +
        "single programmer anticipated, running on hardware whose memory bandwidth matters more than its clock speed.";

    private static string? Find(string file)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                if (System.IO.File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    [Theory]
    [InlineData(null)]              // the default KV dtype
    [InlineData(DType.Float32)]     // exact F32 copy
    public void Handoff_DecodeAgreesWithSequentialPrefill(DType? kv)
    {
        string? path = Find(File);
        Assert.SkipWhen(path is null, $"{File} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu!;
        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Text).ToArray();
        int n = prompt.Length;
        Assert.True(n >= 100, $"prompt too short: {n}");

        var forced = new int[24];
        var decodeFast = new List<float[]>();
        float[] fastLogits;
        using (var fast = new GpuForwardPass(model, gpu!, hp, maxContextLength: 1024, kvDtype: kv))
        {
            fastLogits = fast.Prefill(prompt, 0).ToArray();
            Assert.True(fast.LastPrefillUsedCpuHandoff, "the handoff was not taken: " + fast.LastCpuPrefillRefusal);
            var logits = fastLogits;
            for (int i = 0; i < forced.Length; i++)
            {
                forced[i] = Argmax(logits);
                logits = fast.Forward(forced[i], n + i).ToArray();
                decodeFast.Add(logits);
            }
        }

        var decodeSeq = new List<float[]>();
        float[] seqLogits;
        using (var seq = new GpuForwardPass(model, gpu!, hp, maxContextLength: 1024, kvDtype: kv))
        {
            seq.CpuPrefillEnabled = false;
            seqLogits = seq.Prefill(prompt, 0).ToArray();
            Assert.False(seq.LastPrefillUsedCpuHandoff);
            for (int i = 0; i < forced.Length; i++)
                decodeSeq.Add(seq.Forward(forced[i], n + i).ToArray());
        }

        var failures = new List<string>();
        int softDips = 0;
        void Compare(string stage, float[] reference, float[] actual)
        {
            double cos = Cosine(reference, actual);
            int ra = Argmax(reference), aa = Argmax(actual);
            if (ra != aa)
            {
                float range = reference.Max() - reference.Min(), gap = reference[ra] - reference[aa];
                if (gap / range >= 0.02) failures.Add($"{stage}: argmax {ra} vs {aa}, gap {gap:F3} of range {range:F3}");
            }
            if (cos <= 0.98) failures.Add($"{stage}: cosine {cos:F6}");
            else if (cos <= 0.99 && ++softDips > 2) failures.Add($"{stage}: cosine {cos:F6} (more than two steps below 0.99)");
        }
        Compare("prefill logits", seqLogits, fastLogits);
        for (int i = 0; i < forced.Length; i++) Compare($"decode {i + 1}", decodeSeq[i], decodeFast[i]);
        Console.WriteLine($"[gemma4-handoff] kv {kv}: n={n}; prefill cosine {Cosine(seqLogits, fastLogits):F6}; " +
                          $"decode cosines {string.Join(",", decodeSeq.Select((x, i) => Cosine(x, decodeFast[i]).ToString("F5")))}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }
}
