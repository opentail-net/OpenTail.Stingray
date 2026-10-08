namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Stage B of <c>docs/2-coverage/2026-10-04-moe-handoff-closure-plan.md</c> for the Vulkan layer split
/// (<see cref="VulkanLayerSplitForwardPass"/>, <see cref="HandoffPath.VulkanLayerSplit"/>): a prompt goes through the CPU
/// prefill, the GPU layers receive their K/V rows, and decode continues split. Checked against the all-CPU
/// <see cref="Engine.ForwardPass"/>: GPU K/V rows equal the CPU pass's (exactly for F32, to fp16 rounding for the packed KV),
/// prefill logits equal the CPU pass's, and 40 teacher-forced decode steps agree (cosine above 0.99, an argmax difference only
/// on a near-tie of the CPU's own logits). Skips visibly without the checkpoint.
/// </summary>
public sealed class LayerSplitCpuPrefillHandoffTests : HeavyTestBase
{
    // One Fact per checkpoint (not a Theory) so a single heavy model can be run alone with -method; each loads a 16-45 GB file.
    [Fact] public void GlmAir_Handoff_GpuKvEqualsCpuAndDecodeAgrees() => Handoff("GLM-4.5-Air-Q2_K.gguf", 4);                   // glm4moe: leading dense layer, selection-bias sigmoid routing, partial RoPE, shared expert
    [Fact] public void HunyuanA13B_Handoff_GpuKvEqualsCpuAndDecodeAgrees() => Handoff("tencent.Hunyuan-A13B-Instruct.Q3_K_S.gguf", 4);   // hunyuan-moe: shared expert, QK-norm after RoPE
    [Fact] public void TrinityMini_Handoff_GpuKvEqualsCpuAndDecodeAgrees() => Handoff("Trinity-Mini-Q4_K_M.gguf", 4);                  // afmoe: attention output gate, sliding/global layers, shared expert, leading dense layers

    private static void Handoff(string file, int split)
    {
        string? path = FindModelPath(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        string? wiki = FindWiki();
        Assert.SkipWhen(wiki is null, "scripts/kvarn-gate/wiki.test.raw not found");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        string text = File.ReadAllText(wiki!);
        text = text[..Math.Min(900, text.Length)];
        text = text[..text.LastIndexOf(' ')];
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(text).ToArray();
        Assert.True(prompt.Length >= 64, $"prompt too short for a handoff: {prompt.Length}");
        const int steps = 40;
        int kvDim = hp.NumKvHeads * hp.HeadDim;

        // All-CPU reference: prefill logits, the K/V of the GPU layers, and the greedy decode it will be teacher-forced on.
        var cpuLogits = new List<float[]>();
        var forced = new int[steps];
        var cpuK = new float[split][];
        var cpuV = new float[split][];
        using (var backend = new CpuBackend())
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
        {
            var l = fwd.Prefill(prompt).ToArray();
            cpuLogits.Add(l);
            for (int layer = 0; layer < split; layer++)
            {
                cpuK[layer] = new float[prompt.Length * kvDim];
                cpuV[layer] = new float[prompt.Length * kvDim];
                KvHandoff.ExtractRows(fwd.KvCacheForHandoff, layer, prompt.Length, hp.NumKvHeads, hp.HeadDim, cpuK[layer], cpuV[layer]);
            }
            for (int s = 0; s < steps; s++)
            {
                forced[s] = Argmax(l);
                l = fwd.Forward(forced[s], prompt.Length + s).ToArray();
                cpuLogits.Add(l);
            }
        }

        var splitLogits = new List<float[]>();
        using (var fwd = new VulkanLayerSplitForwardPass(model, gpu!, hp, split, maxContextLength: 1024))
        {
            splitLogits.Add(fwd.Prefill(prompt).ToArray());
            Assert.True(fwd.LastPrefillUsedCpuHandoff, $"handoff did not run: {fwd.LastCpuPrefillRefusal}");

            // GPU K/V rows against the CPU pass's.
            bool exact = fwd.GpuKvDType == DType.Float32;
            var gk = new float[prompt.Length * kvDim];
            var gv = new float[prompt.Length * kvDim];
            for (int layer = 0; layer < split; layer++)
            {
                fwd.ReadGpuKvRows(layer, prompt.Length, gk, gv);
                for (int i = 0; i < gk.Length; i++)
                {
                    float ek = exact ? cpuK[layer][i] : (float)(Half)cpuK[layer][i];
                    float ev = exact ? cpuV[layer][i] : (float)(Half)cpuV[layer][i];
                    if (gk[i] != ek || gv[i] != ev)
                        Assert.Fail($"layer {layer} element {i}: GPU K {gk[i]} V {gv[i]} vs expected K {ek} V {ev} (kv dtype {fwd.GpuKvDType})");
                }
            }

            for (int s = 0; s < steps; s++)
                splitLogits.Add(fwd.Forward(forced[s], prompt.Length + s).ToArray());
        }

        // Prefill logits come straight from the CPU pass.
        Assert.Equal(cpuLogits[0], splitLogits[0]);

        double worst = 1;
        int flips = 0;
        var failures = new List<string>();
        for (int i = 0; i < cpuLogits.Count; i++)
        {
            float[] c = cpuLogits[i], g = splitLogits[i];
            double cos = Cosine(c, g);
            worst = Math.Min(worst, cos);
            int ca = Argmax(c), ga = Argmax(g);
            string stage = i == 0 ? "prefill" : $"decode {i}";
            if (ca != ga)
            {
                flips++;
                float range = c.Max() - c.Min(), gap = c[ca] - c[ga];
                if (gap / range >= 0.02)
                    failures.Add($"{stage}: CPU {ca} vs split {ga}, CPU gap {gap:F3} of range {range:F3}");
            }
            if (cos <= 0.99) failures.Add($"{stage}: cosine {cos:F6}");
        }
        Console.WriteLine($"[split-handoff] {file} -g {split}: n={prompt.Length}; worst cosine {worst:F6}, argmax flips {flips}/{cpuLogits.Count}; " +
            "decode cosines " + string.Join(",", Enumerable.Range(1, steps).Select(i => Cosine(cpuLogits[i], splitLogits[i]).ToString("F5"))));
        Assert.True(failures.Count == 0, string.Join("; ", failures));
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

    private static string? FindWiki()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            string p = Path.Combine(d.FullName, "scripts", "kvarn-gate", "wiki.test.raw");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private static string? FindModelPath(string modelFile)
    {
        string h = Path.Combine(@"H:\_models", modelFile);
        if (File.Exists(h)) return h;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", modelFile), Path.Combine(dir.FullName, "models", "_models", modelFile) })
                if (File.Exists(p)) return p;
        return null;
    }
}
