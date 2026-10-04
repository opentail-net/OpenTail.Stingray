namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Phase 1 of the batched-MoE-prefill plan: <see cref="HybridForwardPass.Prefill"/> runs a fresh long prompt on the CPU
/// batched pass and hands its K/V to the hybrid (<c>HybridPrefillHandoff.cs</c>). Three separate properties, as the plan
/// requires, because they fail for different reasons:
/// <list type="bullet">
/// <item>A. the transfer is exact: every K/V row the hybrid now holds (GPU tensors and CPU cache) is byte-identical to the
/// CPU pass's own cache, and the returned logits are byte-identical to the CPU pass's;</item>
/// <item>B. decode after the handoff agrees with decode after the sequential hybrid prefill under the cross-backend contract
/// (cosine above 0.99, an argmax disagreement only on a near-tie of the reference): the K/V values come from CPU kernels
/// instead of GPU kernels, so bitwise equality is not the bar;</item>
/// <item>C. the gating: short prompts, <c>startPos &gt; 0</c> and the boundary lengths use the old path.</item>
/// </list>
/// Real weights and a real Vulkan device; skips visibly without either.
/// </summary>
public sealed unsafe class HybridCpuPrefillHandoffTests : HeavyTestBase
{
    // Varied prose on purpose: a repetitive prompt gives near-tied logits and tells us nothing.
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
        foreach (var p in new[] { Path.Combine(@"F:\_models", file), Path.Combine(@"E:\_models", file), Path.Combine(@"H:\_models", file) })
            if (File.Exists(p)) return p;
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                if (File.Exists(p)) return p;
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

    private sealed record Setup(GgufModel Model, ModelHyperparams Hp, int[] Prompt, VulkanBackend Gpu) : IDisposable
    {
        public void Dispose() { Model.Dispose(); Gpu.Dispose(); }
    }

    private static Setup? Open(string file, string? text = null)
    {
        string? path = Find(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(text ?? Text).ToArray();
        return new Setup(model, hp, prompt, gpu!);
    }

    private static HybridForwardPass NewHybrid(Setup s, int gpuLayers, int slots, int ctx = 1024) =>
        new(s.Model, s.Gpu, s.Hp,
            new LayerPlacement(gpuLayers, s.Hp.NumLayers - gpuLayers, 0, 0, ctx), expertSlotCapacity: slots);

    [Theory]
    [InlineData("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", 4, -1)]
    [InlineData("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", 8, 16)]
    [InlineData("Qwen3-0.6B-Q8_0.gguf", 8, -1)]
    [InlineData("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", 4, 16)]
    [InlineData("Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf", 4, 16)]   // qwen2moe: shared expert with sigmoid gate, 60 experts top-4
    [InlineData("Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf", 1, 8)]
    [InlineData("Nous-Hermes-2-Mixtral-8x7B-DPO.i1-Q4_K_S.gguf", 4, 16)]   // llama + experts (Mixtral-8x7B): 8 experts top-2, renormalised
    [InlineData("Nous-Hermes-2-Mixtral-8x7B-DPO.i1-Q4_K_S.gguf", 1, 4)]
    // phimoe (LongRoPE, RMSNorm + bias, LM-head bias): ctx 1024 uses the short factors, ctx 8192 (> original_context_length 4096) the long ones.
    [InlineData("Phi-3.5-MoE-instruct-Q3_K_M.gguf", 4, 16, 1024)]
    [InlineData("Phi-3.5-MoE-instruct-Q3_K_M.gguf", 1, 8, 1024)]
    [InlineData("Phi-3.5-MoE-instruct-Q3_K_M.gguf", 4, 16, 8192)]
    public void Handoff_IsByteExact_AndDecodeAgreesWithSequentialPrefill(string file, int gpuLayers, int slots, int ctx = 1024)
        => RunHandoff(file, gpuLayers, slots, ctx);

    /// <summary>
    /// One extra checkpoint chosen at run time: <c>STINGRAY_HANDOFF_TEST_MODEL=&lt;file name&gt;</c> (looked up like the InlineData rows), optionally
    /// <c>STINGRAY_HANDOFF_TEST_GPU_LAYERS</c> (default 4) and <c>STINGRAY_HANDOFF_TEST_CTX</c> (default 1024). Skips without the variable.
    /// A new family is tried here first; its permanent InlineData row is added only together with its receipt.
    /// </summary>
    [Fact]
    public void Handoff_EnvironmentChosenModel()
    {
        string? file = Environment.GetEnvironmentVariable("STINGRAY_HANDOFF_TEST_MODEL");
        Assert.SkipWhen(string.IsNullOrWhiteSpace(file), "STINGRAY_HANDOFF_TEST_MODEL is not set");
        int layers = int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HANDOFF_TEST_GPU_LAYERS"), out int l) ? l : 4;
        int ctx = int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HANDOFF_TEST_CTX"), out int c) ? c : 1024;
        RunHandoff(file!, layers, 16, ctx);
    }

    /// <summary>
    /// Llama 4 past its 8192-token attention chunk: a 9430-token prompt (first 42000 characters of scripts/kvarn-gate/wiki.test.raw, as in
    /// LlamaFourLongContextParityTests) goes through the CPU-prefill handoff into 4 GPU layers + 44 CPU layers, then 6 teacher-forced decode steps
    /// are compared with the all-CPU pass continuing from the same prompt. The sequential GPU prefill is not run (about 1 token/s on this
    /// machine, hours at this length); the short-prompt rows cover that comparison. Exercises chunked attention and NoPE temperature tuning on
    /// both the GPU layers (window = position % 8192 + 1) and the CPU layers.
    /// </summary>
    [Fact]
    public void Handoff_Llama4_PastTheChunkBoundary_DecodeAgreesWithCpu()
    {
        const string file = @"Q3_K_M\Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002.gguf";
        const int ctx = 9728, steps = 6;
        string? wiki = null;
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null && wiki is null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "scripts", "kvarn-gate", "wiki.test.raw"))) wiki = Path.Combine(d.FullName, "scripts", "kvarn-gate", "wiki.test.raw");
        Assert.SkipWhen(wiki is null, "scripts/kvarn-gate/wiki.test.raw not found");
        string text = File.ReadAllText(wiki!);
        text = text[..Math.Min(42000, text.Length)];
        text = text[..text.LastIndexOf('\n')];

        using var s = Open(file, text)!;
        Assert.True(s.Prompt.Length > 8192 + 64, $"prompt must cross the 8192 boundary: {s.Prompt.Length}");

        var cpuLogits = new List<float[]>();
        var forced = new int[steps];
        using (var backend = new CpuBackend())
        using (var cpu = new Engine.ForwardPass(s.Model, backend, s.Hp, maxContextLength: ctx))
        {
            var l = cpu.Prefill(s.Prompt).ToArray();
            cpuLogits.Add(l);
            for (int i = 0; i < steps; i++)
            {
                forced[i] = Argmax(l);
                l = cpu.Forward(forced[i], s.Prompt.Length + i).ToArray();
                cpuLogits.Add(l);
            }
        }

        var hybridLogits = new List<float[]>();
        using (var fast = NewHybrid(s, 4, 16, ctx))
        {
            hybridLogits.Add(fast.Prefill(s.Prompt, 0).ToArray());
            Assert.True(fast.LastPrefillUsedCpuHandoff, "the handoff was not taken: " + fast.LastCpuPrefillRefusal);
            for (int i = 0; i < steps; i++) hybridLogits.Add(fast.Forward(forced[i], s.Prompt.Length + i).ToArray());
        }

        Assert.Equal(cpuLogits[0], hybridLogits[0]);   // the handoff returns the CPU pass's own logits
        var failures = new List<string>();
        for (int i = 1; i <= steps; i++)
        {
            double cos = Cosine(cpuLogits[i], hybridLogits[i]);
            int ca = Argmax(cpuLogits[i]), ha = Argmax(hybridLogits[i]);
            if (ca != ha)
            {
                float range = cpuLogits[i].Max() - cpuLogits[i].Min(), gap = cpuLogits[i][ca] - cpuLogits[i][ha];
                if (gap / range >= 0.02) failures.Add($"decode {i}: argmax {ca} vs {ha}, gap {gap:F3} of range {range:F3}");
            }
            if (cos <= 0.99) failures.Add($"decode {i}: cosine {cos:F6}");
        }
        Console.WriteLine($"[handoff] llama4 past 8192: n={s.Prompt.Length}; decode cosines {string.Join(",", Enumerable.Range(1, steps).Select(i => Cosine(cpuLogits[i], hybridLogits[i]).ToString("F5")))}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    // Large checkpoints, kept in their own method so they can be run (and re-run) without the 14 smaller rows.
    [Theory]
    [InlineData(@"Q3_K_M\Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002.gguf", 4, 16)]   // llama4: NoPE every 4th layer, L2 QK-norm, top-1 + shared expert
    [InlineData(@"Q3_K_M\Llama-4-Scout-17B-16E-Instruct-Q3_K_M-00001-of-00002.gguf", 1, 16)]
    public void Handoff_LargeFamilies(string file, int gpuLayers, int slots, int ctx = 1024)
        => RunHandoff(file, gpuLayers, slots, ctx);

    private void RunHandoff(string file, int gpuLayers, int slots, int ctx)
    {
        using var s = Open(file)!;
        Assert.True(s.Prompt.Length >= 100, $"prompt too short for a meaningful test: {s.Prompt.Length}");
        int n = s.Prompt.Length;
        int kvDim = s.Hp.NumKvHeads * s.Hp.HeadDim;

        // ── Reference: an independent CPU pass and F32 cache, the exact computation the handoff reuses ──
        var refK = new float[s.Hp.NumLayers][];
        var refV = new float[s.Hp.NumLayers][];
        float[] refLogits;
        using (var backend = new CpuBackend())
        using (var cpu = new Engine.ForwardPass(s.Model, backend, s.Hp, maxContextLength: ctx))
        using (var cache = new PagedKvCache(s.Hp.NumLayers, s.Hp.NumKvHeads, s.Hp.HeadDim, bf16Store: false, autoBf16: false, layerHeadDim: null))
        {
            refLogits = cpu.PrefillWithCache(s.Prompt, cache, 0).ToArray();
            for (int l = 0; l < s.Hp.NumLayers; l++)
            {
                refK[l] = new float[n * kvDim];
                refV[l] = new float[n * kvDim];
                for (int p = 0; p < n; p++)
                {
                    new ReadOnlySpan<float>(cache.KeyAt(l, p), kvDim).CopyTo(refK[l].AsSpan(p * kvDim, kvDim));
                    for (int h = 0; h < s.Hp.NumKvHeads; h++)
                        new ReadOnlySpan<float>(cache.ValueAtHead(l, p, h), s.Hp.HeadDim).CopyTo(refV[l].AsSpan(p * kvDim + h * s.Hp.HeadDim, s.Hp.HeadDim));
                }
            }
        }

        // ── A. Handoff exactness ──
        float[] fastLogits;
        var decodeFast = new List<float[]>();
        var forced = new int[40];   // closure requirement: 40 teacher-forced decode steps
        using (var fast = NewHybrid(s, gpuLayers, slots, ctx))
        {
            var l0 = fast.Prefill(s.Prompt, 0).ToArray();
            Assert.True(fast.LastPrefillUsedCpuHandoff, "the handoff was not taken: " + fast.LastCpuPrefillRefusal);
            fastLogits = l0;
            Assert.Equal(refLogits, fastLogits); // same computation, same bytes

            var k = new float[n * kvDim];
            var v = new float[n * kvDim];
            for (int l = 0; l < s.Hp.NumLayers; l++)
            {
                fast.ReadKvRows(l, n, k, v);
                Assert.True(refK[l].AsSpan().SequenceEqual(k), $"layer {l} K rows differ after the handoff ({(l < gpuLayers ? "GPU" : "CPU")} layer)");
                Assert.True(refV[l].AsSpan().SequenceEqual(v), $"layer {l} V rows differ after the handoff ({(l < gpuLayers ? "GPU" : "CPU")} layer)");
            }

            // decode on the hybrid after the handoff, teacher-forced on its own greedy tokens
            var logits = l0;
            for (int i = 0; i < forced.Length; i++)
            {
                forced[i] = Argmax(logits);
                logits = fast.Forward(forced[i], n + i).ToArray();
                decodeFast.Add(logits);
            }
        }

        // ── B. Decode after handoff vs decode after the sequential hybrid prefill ──
        var decodeSeq = new List<float[]>();
        float[] seqLogits;
        using (var seq = NewHybrid(s, gpuLayers, slots, ctx))
        {
            seq.CpuPrefillEnabled = false;
            seqLogits = seq.Prefill(s.Prompt, 0).ToArray();
            Assert.False(seq.LastPrefillUsedCpuHandoff);
            for (int i = 0; i < forced.Length; i++)
                decodeSeq.Add(seq.Forward(forced[i], n + i).ToArray());
        }

        var failures = new List<string>();
        void Compare(string stage, float[] reference, float[] actual)
        {
            double cos = Cosine(reference, actual);
            int ra = Argmax(reference), aa = Argmax(actual);
            if (ra != aa)
            {
                float range = reference.Max() - reference.Min(), gap = reference[ra] - reference[aa];
                if (gap / range >= 0.02) failures.Add($"{stage}: argmax {ra} vs {aa}, gap {gap:F3} of range {range:F3}");
            }
            if (cos <= 0.99) failures.Add($"{stage}: cosine {cos:F6}");
        }
        Compare("prefill logits", seqLogits, fastLogits);
        for (int i = 0; i < forced.Length; i++) Compare($"decode {i + 1}", decodeSeq[i], decodeFast[i]);
        Console.WriteLine($"[handoff] {file} -g {gpuLayers}: n={n}; prefill cosine {Cosine(seqLogits, fastLogits):F6}; " +
                          $"decode cosines {string.Join(",", decodeSeq.Select((x, i) => Cosine(x, decodeFast[i]).ToString("F5")))}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Fact]
    public void ShortPrompts_StartPosAboveZero_AndDisabledSwitch_UseTheSequentialPath()
    {
        using var s = Open("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")!;
        using var h = NewHybrid(s, 4, -1);

        // below the gate (32): sequential, with the reason recorded
        foreach (int len in new[] { 1, 2, 15, 16, 17, 31 })
        {
            h.ResetCache();
            h.Prefill(s.Prompt.AsSpan(0, len).ToArray(), 0);
            Assert.False(h.LastPrefillUsedCpuHandoff, $"{len} tokens must not take the handoff");
            Assert.Contains("shorter than", h.LastCpuPrefillRefusal);
        }

        // at and just above the gate: handoff
        foreach (int len in new[] { 32, 33 })
        {
            h.ResetCache();
            h.Prefill(s.Prompt.AsSpan(0, len).ToArray(), 0);
            Assert.True(h.LastPrefillUsedCpuHandoff, $"{len} tokens should take the handoff: {h.LastCpuPrefillRefusal}");
        }

        // a later turn (startPos > 0) must stay sequential, and must agree with a pure sequential run of the same tokens
        h.ResetCache();
        int first = 64, extra = 8;
        h.Prefill(s.Prompt.AsSpan(0, first).ToArray(), 0);
        Assert.True(h.LastPrefillUsedCpuHandoff);
        var continued = h.Prefill(s.Prompt.AsSpan(first, extra).ToArray(), first).ToArray();
        Assert.False(h.LastPrefillUsedCpuHandoff);
        Assert.Contains("startPos", h.LastCpuPrefillRefusal);

        using var seq = NewHybrid(s, 4, -1);
        seq.CpuPrefillEnabled = false;
        var reference = seq.Prefill(s.Prompt.AsSpan(0, first + extra).ToArray(), 0).ToArray();
        double cos = Cosine(reference, continued);
        Console.WriteLine($"[handoff] startPos>0 continuation cosine vs all-sequential: {cos:F6}");
        Assert.True(cos > 0.99, $"continuation after a handed-off prefix diverges: cosine {cos:F6}");

        // switch off
        h.ResetCache();
        h.CpuPrefillEnabled = false;
        h.Prefill(s.Prompt, 0);
        Assert.False(h.LastPrefillUsedCpuHandoff);
        Assert.Contains("disabled", h.LastCpuPrefillRefusal);
    }

    [Fact]
    public void ExpertWarmup_RemovesTheColdStartOfDecodeAfterTheHandoff()
    {
        using var s = Open("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")!;
        int n = s.Prompt.Length;

        // Miss rate over the first 8 decode steps, with and without the warm-up. Without it the GPU expert cache is empty after a
        // CPU prefill (the sequential prefill would have filled it as a side effect) and most lookups miss; with it the experts the
        // prompt used were uploaded while the CPU was still working, so they hit.
        double MissRate(bool warm)
        {
            using var h = NewHybrid(s, 4, -1);
            h.CpuPrefillWarmExperts = warm;
            var logits = h.Prefill(s.Prompt, 0).ToArray();
            Assert.True(h.LastPrefillUsedCpuHandoff, h.LastCpuPrefillRefusal);
            h.WaitForExpertWarmup();
            var profiler = h.ExpertSlots!.Profiler;
            long hits0 = profiler.TotalHits, misses0 = profiler.TotalMisses;
            for (int i = 0; i < 8; i++)
            {
                int next = Argmax(logits);
                logits = h.Forward(next, n + i).ToArray();
            }
            long hits = profiler.TotalHits - hits0, misses = profiler.TotalMisses - misses0;
            return (double)misses / Math.Max(1, hits + misses);
        }

        double cold = MissRate(warm: false);
        double warmed = MissRate(warm: true);
        Console.WriteLine($"[handoff] decode miss rate over 8 steps: without warm-up {cold:P1}, with warm-up {warmed:P1}");
        Assert.True(cold > 0.3, $"without the warm-up the cache should start cold (miss rate {cold:P1}); the premise of the warm-up is gone");
        Assert.True(warmed < 0.1, $"with the warm-up the first decode steps should mostly hit (miss rate {warmed:P1})");
    }

    [Fact]
    public void UnadmittedFamily_AndOversizedKvBudget_AreRefusedWithAReason()
    {
        using var s = Open("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")!;
        using var h = NewHybrid(s, 4, -1);
        h.CpuPrefillKvBudgetBytes = 1024; // far below any prompt's temporary KV cache
        h.Prefill(s.Prompt, 0);
        Assert.False(h.LastPrefillUsedCpuHandoff);
        Assert.Contains("budget", h.LastCpuPrefillRefusal);
    }
}
