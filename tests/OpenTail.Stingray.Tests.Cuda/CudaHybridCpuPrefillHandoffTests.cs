namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// CUDA counterpart of the Vulkan <c>HybridCpuPrefillHandoffTests</c> (Phase 1 of the batched-MoE-prefill plan).
/// <b>Written on a machine with no NVIDIA GPU: this file compiles but has never run.</b> It skips visibly without a CUDA
/// device or the checkpoint, and the path it covers (<c>CudaHybridPrefillHandoff.cs</c>, opt-in via
/// <c>STINGRAY_CUDA_HYBRID_CPU_PREFILL=1</c>) must not become a default until these pass on CUDA hardware and the speed
/// is measured there (the CUDA hybrid already has its own batched trunk, so the CPU pass is not assumed to be faster).
/// <list type="bullet">
/// <item>A. the handoff is exact: every K/V row now in the hybrid (GPU tensors and CPU cache) is byte-identical to the CPU
/// pass's cache, and the logits equal the CPU pass's;</item>
/// <item>B. decode after the handoff agrees with decode after the hybrid's own prefill (cosine above 0.99, argmax flips only
/// on a near-tie);</item>
/// <item>C. short prompts and <c>startPos &gt; 0</c> keep the old path.</item>
/// </list>
/// </summary>
[Trait("Category", "Cuda")]
public sealed unsafe class CudaHybridCpuPrefillHandoffTests
{
    private const string Text =
        "In 1843 Ada Lovelace published notes describing how a machine could manipulate symbols according to rules, " +
        "an idea that reached far beyond arithmetic. A century later wartime codebreakers built electronic devices to " +
        "search enormous spaces of possibilities, and stored-program computers followed within a few years. Transistors " +
        "replaced vacuum tubes, integrated circuits replaced transistors, and the cost of a calculation fell by many " +
        "orders of magnitude. Operating systems learned to share one processor among many programs; networks joined " +
        "machines across continents; databases promised that facts written once could be read correctly forever. Today " +
        "language models trained on vast text collections can summarise, translate and reason about problems that no " +
        "single programmer anticipated, running on hardware whose memory bandwidth matters more than its clock speed.";

    private static string? FindModel()
    {
        foreach (var file in new[] { "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" })
        {
            foreach (var p in new[] { Path.Combine(@"F:\_models", file), Path.Combine(@"E:\_models", file) })
                if (File.Exists(p)) return p;
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null)
            {
                foreach (var p in new[] { Path.Combine(dir.FullName, "models", file), Path.Combine(dir.FullName, "models", "_models", file) })
                    if (File.Exists(p)) return p;
                dir = dir.Parent;
            }
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

    [Fact]
    public void Handoff_IsByteExact_AndDecodeAgreesWithTheHybridsOwnPrefill()
    {
        using var gpu = CudaTestGpu.TryCreate();
        Assert.SkipUnless(gpu is not null, "no CUDA device in this environment");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "MoE model fixture not present in this environment");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Text).ToArray();
        int n = prompt.Length;
        int kvDim = hp.NumKvHeads * hp.HeadDim;
        int gpuLayers = 4;
        var placement = new LayerPlacement(gpuLayers, hp.NumLayers - gpuLayers, 0, 0, 1024);

        // Independent CPU reference and F32 cache
        var refK = new float[hp.NumLayers][];
        var refV = new float[hp.NumLayers][];
        float[] refLogits;
        using (var backend = new CpuBackend())
        using (var cpu = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
        using (var cache = new PagedKvCache(hp.NumLayers, hp.NumKvHeads, hp.HeadDim, bf16Store: false, autoBf16: false, layerHeadDim: null))
        {
            refLogits = cpu.PrefillWithCache(prompt, cache, 0).ToArray();
            for (int l = 0; l < hp.NumLayers; l++)
            {
                refK[l] = new float[n * kvDim];
                refV[l] = new float[n * kvDim];
                for (int p = 0; p < n; p++)
                {
                    new ReadOnlySpan<float>(cache.KeyAt(l, p), kvDim).CopyTo(refK[l].AsSpan(p * kvDim, kvDim));
                    for (int h = 0; h < hp.NumKvHeads; h++)
                        new ReadOnlySpan<float>(cache.ValueAtHead(l, p, h), hp.HeadDim).CopyTo(refV[l].AsSpan(p * kvDim + h * hp.HeadDim, hp.HeadDim));
                }
            }
        }

        var forced = new int[6];
        var decodeFast = new List<float[]>();
        float[] fastLogits;
        using (var fast = new CudaHybridForwardPass(model, gpu!, hp, placement))
        {
            Assert.SkipUnless(fast.KvCacheDType == DType.Float32, "GPU KV is narrowed (STINGRAY_KV_DTYPE); the handoff is F32-only");
            fast.CpuPrefillEnabled = true; // opt-in path
            fastLogits = fast.Prefill(prompt, 0).ToArray();
            Assert.True(fast.LastPrefillUsedCpuHandoff, "the handoff was not taken: " + fast.LastCpuPrefillRefusal);
            Assert.Equal(refLogits, fastLogits);

            var k = new float[n * kvDim];
            var v = new float[n * kvDim];
            for (int l = 0; l < hp.NumLayers; l++)
            {
                fast.ReadKvRows(l, n, k, v);
                Assert.True(refK[l].AsSpan().SequenceEqual(k), $"layer {l}: K rows differ ({(l < gpuLayers ? "GPU" : "CPU")} layer)");
                Assert.True(refV[l].AsSpan().SequenceEqual(v), $"layer {l}: V rows differ ({(l < gpuLayers ? "GPU" : "CPU")} layer)");
            }

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
        using (var seq = new CudaHybridForwardPass(model, gpu!, hp, placement))
        {
            seq.CpuPrefillEnabled = false;
            seqLogits = seq.Prefill(prompt, 0).ToArray();
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
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Fact]
    public void ShortPrompts_AndStartPosAboveZero_KeepTheOldPath()
    {
        using var gpu = CudaTestGpu.TryCreate();
        Assert.SkipUnless(gpu is not null, "no CUDA device in this environment");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "MoE model fixture not present in this environment");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] prompt = GgufTokenizer.FromGgufModel(model).Encode(Text).ToArray();
        using var h = new CudaHybridForwardPass(model, gpu!, hp, new LayerPlacement(4, hp.NumLayers - 4, 0, 0, 1024));
        Assert.SkipUnless(h.KvCacheDType == DType.Float32, "GPU KV is narrowed; the handoff is F32-only");

        Assert.False(h.CpuPrefillEnabled, "the CUDA handoff must stay opt-in until it is measured on CUDA hardware");
        h.Prefill(prompt, 0);
        Assert.Contains("disabled", h.LastCpuPrefillRefusal);

        h.CpuPrefillEnabled = true;
        foreach (int len in new[] { 1, 2, 15, 16, 17, 31 })
        {
            h.ResetCache();
            h.Prefill(prompt.AsSpan(0, len).ToArray(), 0);
            Assert.False(h.LastPrefillUsedCpuHandoff, $"{len} tokens must not take the handoff");
        }
        h.ResetCache();
        h.Prefill(prompt.AsSpan(0, 64).ToArray(), 0);
        Assert.True(h.LastPrefillUsedCpuHandoff, h.LastCpuPrefillRefusal);
        h.Prefill(prompt.AsSpan(64, 8).ToArray(), 64);
        Assert.False(h.LastPrefillUsedCpuHandoff);
        Assert.Contains("startPos", h.LastCpuPrefillRefusal);
    }
}
