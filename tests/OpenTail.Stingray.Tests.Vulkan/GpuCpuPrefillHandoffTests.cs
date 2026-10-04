namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Phase 1b of the batched-MoE-prefill plan: <see cref="GpuForwardPass.Prefill"/> (full offload, <c>-g -1</c>) on a MoE model
/// runs a fresh long prompt on the CPU batched pass and hands its K/V to the GPU caches (<c>GpuPrefillHandoff.cs</c>).
/// The same three properties as <see cref="HybridCpuPrefillHandoffTests"/>, plus the narrowing this pass applies to KV:
/// <list type="bullet">
/// <item>A. the transfer is exact: with F32 KV the downloaded rows equal the CPU pass's cache byte for byte; with packed fp16
/// KV (the default when it can narrow; "BFloat16" in the code, IEEE half in the shaders) they equal an independent
/// <see cref="Half"/> conversion of that cache; the returned logits equal the CPU pass's byte for byte;</item>
/// <item>B. decode after the handoff agrees with decode after the sequential GPU prefill (cosine above 0.99, argmax flips only
/// on a near-tie): this is the end-to-end check that the packing order and layout are right, because attention reads garbage
/// if they are not;</item>
/// <item>C. gating: short prompts, <c>startPos &gt; 0</c> and the off switch keep the old path.</item>
/// </list>
/// Real weights and a real Vulkan device; skips visibly without either.
/// </summary>
public sealed unsafe class GpuCpuPrefillHandoffTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";

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

    private static Setup Open(string file = Olmoe)
    {
        string? path = Find(file);
        Assert.SkipWhen(path is null, $"{file} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        return new Setup(model, hp, GgufTokenizer.FromGgufModel(model).Encode(Text).ToArray(), gpu!);
    }

    [Theory]
    [InlineData(Olmoe, null)]                 // the default KV dtype (packed fp16 when the model can narrow)
    [InlineData(Olmoe, DType.Float32)]        // exact F32 copy
    [InlineData("Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf", null)]   // qwen2moe: shared expert (5632 wide) with sigmoid gate
    [InlineData("Phi-3.5-MoE-instruct-Q3_K_M.gguf", null)]     // phimoe: RMSNorm + bias, LM-head bias
    public void Handoff_IsExact_AndDecodeAgreesWithSequentialPrefill(string file, DType? kv)
    {
        using var s = Open(file);
        int n = s.Prompt.Length;
        Assert.True(n >= 100, $"prompt too short: {n}");
        int kvDim = s.Hp.NumKvHeads * s.Hp.HeadDim;

        // Reference CPU pass and F32 cache
        var refK = new float[s.Hp.NumLayers][];
        var refV = new float[s.Hp.NumLayers][];
        float[] refLogits;
        using (var backend = new CpuBackend())
        using (var cpu = new Engine.ForwardPass(s.Model, backend, s.Hp, maxContextLength: 1024))
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

        // A. exactness
        var forced = new int[6];
        var decodeFast = new List<float[]>();
        float[] fastLogits;
        DType storedKv;
        using (var fast = new GpuForwardPass(s.Model, s.Gpu, s.Hp, maxContextLength: 1024, kvDtype: kv))
        {
            storedKv = fast.KvDTypeForTest;
            fastLogits = fast.Prefill(s.Prompt, 0).ToArray();
            Assert.True(fast.LastPrefillUsedCpuHandoff, "the handoff was not taken: " + fast.LastCpuPrefillRefusal);
            Assert.Equal(refLogits, fastLogits);

            var k = new float[n * kvDim];
            var v = new float[n * kvDim];
            for (int l = 0; l < s.Hp.NumLayers; l++)
            {
                fast.ReadKvRows(l, n, k, v);
                if (storedKv == DType.Float32)
                {
                    Assert.True(refK[l].AsSpan().SequenceEqual(k), $"layer {l}: K rows differ (F32)");
                    Assert.True(refV[l].AsSpan().SequenceEqual(v), $"layer {l}: V rows differ (F32)");
                }
                else
                {
                    for (int i = 0; i < k.Length; i++)
                    {
                        Assert.Equal((float)(Half)refK[l][i], k[i]);
                        Assert.Equal((float)(Half)refV[l][i], v[i]);
                    }
                }
            }

            var logits = fastLogits;
            for (int i = 0; i < forced.Length; i++)
            {
                forced[i] = Argmax(logits);
                logits = fast.Forward(forced[i], n + i).ToArray();
                decodeFast.Add(logits);
            }
        }

        // B. decode after the sequential GPU prefill
        var decodeSeq = new List<float[]>();
        float[] seqLogits;
        using (var seq = new GpuForwardPass(s.Model, s.Gpu, s.Hp, maxContextLength: 1024, kvDtype: kv))
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
        Console.WriteLine($"[gpu-handoff] kv {storedKv}: n={n}; prefill cosine {Cosine(seqLogits, fastLogits):F6}; " +
                          $"decode cosines {string.Join(",", decodeSeq.Select((x, i) => Cosine(x, decodeFast[i]).ToString("F5")))}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Fact]
    public void ShortPrompts_StartPosAboveZero_AndDisabledSwitch_UseTheSequentialPath()
    {
        using var s = Open();
        using var g = new GpuForwardPass(s.Model, s.Gpu, s.Hp, maxContextLength: 1024);

        foreach (int len in new[] { 1, 2, 15, 16, 17, 31 })
        {
            g.ResetCache();
            g.Prefill(s.Prompt.AsSpan(0, len).ToArray(), 0);
            Assert.False(g.LastPrefillUsedCpuHandoff, $"{len} tokens must not take the handoff");
            Assert.Contains("shorter than", g.LastCpuPrefillRefusal);
        }
        foreach (int len in new[] { 32, 33 })
        {
            g.ResetCache();
            g.Prefill(s.Prompt.AsSpan(0, len).ToArray(), 0);
            Assert.True(g.LastPrefillUsedCpuHandoff, $"{len} tokens should take the handoff: {g.LastCpuPrefillRefusal}");
        }

        // later turn: sequential, and it agrees with an all-sequential run of the same tokens
        g.ResetCache();
        int first = 64, extra = 8;
        g.Prefill(s.Prompt.AsSpan(0, first).ToArray(), 0);
        Assert.True(g.LastPrefillUsedCpuHandoff);
        var continued = g.Prefill(s.Prompt.AsSpan(first, extra).ToArray(), first).ToArray();
        Assert.False(g.LastPrefillUsedCpuHandoff);
        Assert.Contains("startPos", g.LastCpuPrefillRefusal);

        using var seq = new GpuForwardPass(s.Model, s.Gpu, s.Hp, maxContextLength: 1024);
        seq.CpuPrefillEnabled = false;
        var reference = seq.Prefill(s.Prompt.AsSpan(0, first + extra).ToArray(), 0).ToArray();
        double cos = Cosine(reference, continued);
        Console.WriteLine($"[gpu-handoff] startPos>0 continuation cosine vs all-sequential: {cos:F6}");
        Assert.True(cos > 0.99, $"continuation after a handed-off prefix diverges: cosine {cos:F6}");

        g.ResetCache();
        g.CpuPrefillEnabled = false;
        g.Prefill(s.Prompt, 0);
        Assert.False(g.LastPrefillUsedCpuHandoff);
        Assert.Contains("disabled", g.LastCpuPrefillRefusal);
    }

    [Fact]
    public void PackHalf_UsesTheShaderLayout_LowHalfFirst()
    {
        float[] src = [1.0f, -2.5f, 0.15625f, 65504f, 1e-8f, -0f];
        var dst = new uint[3];
        GpuForwardPass.PackHalf(src, dst);
        Assert.Equal(0xC1003C00u, dst[0]);   // -2.5h (0xC100) in the high half, 1.0h (0x3C00) low
        Assert.Equal(0x7BFF3100u, dst[1]);   // 65504h (0x7BFF) high, 0.15625h (0x3100) low
        Assert.Equal(0x80000000u, dst[2]);   // -0h high half; 1e-8 is below half the smallest subnormal, so it rounds to 0 in the low half
    }
}
