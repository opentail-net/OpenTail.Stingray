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
    // Varied prose on purpose (a repetitive prompt gives near-tied logits and tells us nothing); about 150 tokens.
    private const string Prose =
        "In 1843 Ada Lovelace published notes describing how a machine could manipulate symbols according to rules, " +
        "an idea that reached far beyond arithmetic. A century later wartime codebreakers built electronic devices to " +
        "search enormous spaces of possibilities, and stored-program computers followed within a few years. Transistors " +
        "replaced vacuum tubes, integrated circuits replaced transistors, and the cost of a calculation fell by many " +
        "orders of magnitude. Operating systems learned to share one processor among many programs; networks joined " +
        "machines across continents; databases promised that facts written once could be read correctly forever. Today " +
        "language models trained on vast text collections can summarise, translate and reason about problems that no " +
        "single programmer anticipated, running on hardware whose memory bandwidth matters more than its clock speed.";

    /// <summary>
    /// A later turn (<c>startPos &gt; 0</c>) takes the handoff on the layer split too: the GPU layers' earlier rows are read back into the
    /// CPU part's cache, the CPU pass prefills only the new tokens against the full history, and the GPU layers receive all rows. Compared
    /// with the all-CPU pass doing the same two prefills: the new-turn logits and 12 teacher-forced decode steps agree (cosine above 0.99;
    /// a MoE router near-tie can move logits, so no byte equality is asserted). Trinity-Mini is the smallest layer-split checkpoint (16 GB).
    /// </summary>
    [Fact]
    public void TrinityMini_LaterTurn_StartPosAboveZero_PrefillLogitsAndDecodeAgreeWithCpu()
    {
        const string file = "Trinity-Mini-Q4_K_M.gguf";
        const int split = 4, firstLen = 48, steps = 12;
        string? path = FindModelPath(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;
        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Prose).ToArray();
        int n = prompt.Length;
        Assert.True(n - firstLen >= 32, $"prompt too short for a second turn past the gate: {n - firstLen} new tokens");
        var first = prompt.AsSpan(0, firstLen).ToArray();
        var second = prompt.AsSpan(firstLen).ToArray();

        float[] cpuTurn2;
        var cpuDecode = new List<float[]>();
        var forced = new int[steps];
        using (var backend = new CpuBackend())
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
        {
            fwd.Prefill(first, 0);
            cpuTurn2 = fwd.Prefill(second, firstLen).ToArray();
            var l = cpuTurn2;
            for (int s = 0; s < steps; s++)
            {
                forced[s] = Argmax(l);
                l = fwd.Forward(forced[s], n + s).ToArray();
                cpuDecode.Add(l);
            }
        }

        using var split_ = new VulkanLayerSplitForwardPass(model, gpu!, hp, split, maxContextLength: 1024);
        split_.Prefill(first, 0);
        Assert.True(split_.LastPrefillUsedCpuHandoff, $"first turn did not take the handoff: {split_.LastCpuPrefillRefusal}");
        var turn2 = split_.Prefill(second, firstLen).ToArray();
        Assert.True(split_.LastPrefillUsedCpuHandoff, $"the later turn did not take the handoff: {split_.LastCpuPrefillRefusal}");
        double c2 = Cosine(cpuTurn2, turn2);
        var cos = new List<double>();
        var logits = turn2;
        for (int s = 0; s < steps; s++)
        {
            logits = split_.Forward(forced[s], n + s).ToArray();
            cos.Add(Cosine(cpuDecode[s], logits));
        }
        cos.Sort();
        Console.WriteLine($"[split-later-turn] {file} -g {split}: turn-2 prefill cosine {c2:F5}; decode min {cos[0]:F5}, median {cos[cos.Count / 2]:F5}");
        Assert.True(c2 > 0.99, $"later-turn prefill logits diverge from the CPU pass: cosine {c2:F5}");
        Assert.True(cos[0] > 0.99, $"decode after the later-turn handoff diverges from the CPU pass: min cosine {cos[0]:F5}");
    }
    /// <summary>What the later-turn handoff buys on the layer split: the second turn (about 100 tokens) after a 48-token first turn, three interleaved repetitions each (machine not idle: read the ratio).</summary>
    [Theory]
    [InlineData("Trinity-Mini-Q4_K_M.gguf")]                          // afmoe: per-token CPU trunk (attention output gate), so the handoff is not faster
    [InlineData("tencent.Hunyuan-A13B-Instruct.Q3_K_S.gguf")]         // hunyuan-moe: batched CPU trunk
    public void LaterTurn_TimingAgainstSequential(string file)
    {
        string? path = FindModelPath(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;
        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Prose).ToArray();
        var first = prompt.AsSpan(0, 48).ToArray();
        var second = prompt.AsSpan(48).ToArray();
        using var f = new VulkanLayerSplitForwardPass(model, gpu!, hp, 4, maxContextLength: 1024);
        f.Prefill(first, 0); f.Prefill(second, 48); f.ResetCache();   // warm both paths once
        var viaHandoff = new List<double>(); var viaSequential = new List<double>();
        for (int rep = 0; rep < 3; rep++)
            foreach (bool handoff in new[] { true, false })
            {
                f.ResetCache();
                f.CpuPrefillEnabled = true;
                f.Prefill(first, 0);
                f.CpuPrefillEnabled = handoff;
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                f.Prefill(second, 48);
                double ms = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                Assert.True(handoff == f.LastPrefillUsedCpuHandoff, "handoff expected " + handoff + ", refusal: " + f.LastCpuPrefillRefusal);
                (handoff ? viaHandoff : viaSequential).Add(ms);
            }
        Console.WriteLine($"[split-later-turn-timing] {file} -g 4, {second.Length}-token second turn: handoff {string.Join(" / ", viaHandoff.Select(x => x.ToString("F0")))} ms, sequential {string.Join(" / ", viaSequential.Select(x => x.ToString("F0")))} ms");
    }
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
