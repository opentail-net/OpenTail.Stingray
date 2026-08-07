// TEMPORARY diagnostic probe — delete before committing.
// Characterises the Gemma 4 Vulkan prefill divergence (issue: Gemma4Vulkan*E2ETests failures)
// by sweeping prompt prefix lengths for both the passing and failing token streams.
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.ForwardPass;

public sealed class TempPleProbe
{
    private const string ModelFile = "gemma-4-E4B_q4_0-it.gguf";

    private static string? FindModelPath(string filename)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>
    /// Is the CPU/Vulkan argmax disagreement a real defect or Q4_0 near-tie noise? Compare the
    /// LOGIT VECTORS, not the argmax: a tiny max|delta| with a near-zero CPU top1-top2 gap means
    /// noise flipping a coin-toss; a large max|delta| means the Vulkan trunk is genuinely wrong.
    /// </summary>
    [Fact]
    public void Probe_LogitDelta()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        var path = FindModelPath(ModelFile);
        if (path is null) { Console.WriteLine("PROBE: no model"); gpu.Dispose(); return; }

        using (gpu)
        using (var model = GgufModel.Open(path))
        {
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            int bos = 2;
            int[] full = [bos, 818, 5279, 529, 7001, 563];

            foreach (int n in new[] { 1, 6 })
            {
                var toks = full[..n];

                float[] cpu;
                using (var cb = new CpuBackend())
                using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                    cpu = cf.Prefill(toks).ToArray();

                float[] vk;
                using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 2048))
                    vk = vf.Prefill(toks).ToArray();

                double maxAbs = 0, sumAbs = 0, dot = 0, nc = 0, nv = 0;
                for (int i = 0; i < cpu.Length; i++)
                {
                    double d = Math.Abs(cpu[i] - vk[i]);
                    if (d > maxAbs) maxAbs = d;
                    sumAbs += d;
                    dot += (double)cpu[i] * vk[i];
                    nc += (double)cpu[i] * cpu[i];
                    nv += (double)vk[i] * vk[i];
                }

                // CPU top1-top2 gap: how close is the argmax to a coin toss?
                var order = Enumerable.Range(0, cpu.Length).OrderByDescending(i => cpu[i]).Take(5).ToArray();
                double gap = cpu[order[0]] - cpu[order[1]];
                var vkOrder = Enumerable.Range(0, vk.Length).OrderByDescending(i => vk[i]).Take(5).ToArray();

                Console.WriteLine(
                    $"PROBE2 len={n}  max|d|={maxAbs:F4}  mean|d|={sumAbs / cpu.Length:F4}  " +
                    $"cos={dot / (Math.Sqrt(nc) * Math.Sqrt(nv)):F6}  cpuTop1-Top2gap={gap:F4}");
                Console.WriteLine($"PROBE2 len={n}  cpuTop5=[{string.Join(",", order)}]  vkTop5=[{string.Join(",", vkOrder)}]");
                Console.WriteLine($"PROBE2 len={n}  cpuTop5vals=[{string.Join(",", order.Select(i => cpu[i].ToString("F3")))}]");
                Console.WriteLine($"PROBE2 len={n}   vkTop5vals=[{string.Join(",", vkOrder.Select(i => vk[i].ToString("F3")))}]");
            }
        }
    }

    /// <summary>
    /// Scope the defect: is the Vulkan backend wrong in general, or only for Gemma 4 / Q4_0?
    /// Compares CPU vs Vulkan logits for non-Gemma models on the generic trunk.
    /// </summary>
    [Fact]
    public void Probe_OtherModels()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            foreach (var file in new[] { "Qwen3-0.6B-Q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf" })
            {
                var path = FindModelPath(file);
                if (path is null) { Console.WriteLine($"PROBE3 {file}: absent"); continue; }

                using var model = GgufModel.Open(path);
                var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
                int[] toks = [1, 2, 3, 5, 7];

                float[] cpu;
                using (var cb = new CpuBackend())
                using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                    cpu = cf.Prefill(toks).ToArray();

                float[] vk;
                try
                {
                    using var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512);
                    vk = vf.Prefill(toks).ToArray();
                }
                catch (Exception ex) { Console.WriteLine($"PROBE3 {file}: vulkan build failed: {ex.GetType().Name}"); continue; }

                double maxAbs = 0, dot = 0, nc = 0, nv = 0;
                for (int i = 0; i < cpu.Length; i++)
                {
                    maxAbs = Math.Max(maxAbs, Math.Abs(cpu[i] - vk[i]));
                    dot += (double)cpu[i] * vk[i]; nc += (double)cpu[i] * cpu[i]; nv += (double)vk[i] * vk[i];
                }
                int ca = Argmax(cpu), va = Argmax(vk);
                Console.WriteLine($"PROBE3 {file,-40} max|d|={maxAbs,8:F3}  cos={dot / (Math.Sqrt(nc) * Math.Sqrt(nv)):F6}  cpuArg={ca} vkArg={va} {(ca == va ? "ok" : "<<<")}");
            }
        }
    }

    /// <summary>Dump the geometry of every local model so the working/broken split can be read off.</summary>
    [Fact]
    public void Probe_HeadDims()
    {
        foreach (var file in new[]
        {
            "Qwen3-0.6B-Q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf",
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "gemma-4-E4B_q4_0-it.gguf",
            "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
        })
        {
            var path = FindModelPath(file);
            if (path is null) { Console.WriteLine($"PROBE4 {file}: absent"); continue; }
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            string layerDims = hp.LayerHeadDim is null
                ? "-"
                : string.Join("/", hp.LayerHeadDim.Distinct());
            bool freqFactors = model.FindTensor("rope_freqs.weight") is not null;
            Console.WriteLine(
                $"PROBE4 {file,-40} emb={hp.EmbeddingDim,5} L={hp.NumLayers,3} " +
                $"heads={hp.NumHeads,3} kvHeads={hp.NumKvHeads,3} headDim={hp.HeadDim,4} " +
                $"ropeDim={hp.RopeDim,4} theta={hp.RopeTheta,10:F1} freqFactors={freqFactors} " +
                $"neox={hp.IsNeoxRope} layerHeadDim={layerDims}");
        }
    }

    /// <summary>
    /// Localise the divergence by prompt length. At position 0 RoPE is the identity and attention
    /// over a single key is degenerate, so len=1 bypasses RoPE / attention / KV-append. A clean
    /// len=1 that breaks at len>=2 indicts those; a broken len=1 indicts matmul / norm / FFN.
    /// </summary>
    [Fact]
    public void Probe_LenSweepCos()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            foreach (var file in new[]
            {
                "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",   // headDim 64  — broken
                "Qwen3-8B-Q4_K_M.gguf",                // headDim 128 — control
            })
            {
                var path = FindModelPath(file);
                if (path is null) { Console.WriteLine($"PROBE5 {file}: absent"); continue; }

                using var model = GgufModel.Open(path);
                var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

                int[] allToks = [1, 2, 3, 5];
                for (int n = 1; n <= allToks.Length; n++)
                {
                    int[] toks = allToks[..n];

                    float[] cpu;
                    using (var cb = new CpuBackend())
                    using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                        cpu = cf.Prefill(toks).ToArray();

                    float[] vk;
                    using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
                        vk = vf.Prefill(toks).ToArray();

                    double maxAbs = 0, dot = 0, nc = 0, nv = 0;
                    for (int i = 0; i < cpu.Length; i++)
                    {
                        maxAbs = Math.Max(maxAbs, Math.Abs(cpu[i] - vk[i]));
                        dot += (double)cpu[i] * vk[i]; nc += (double)cpu[i] * cpu[i]; nv += (double)vk[i] * vk[i];
                    }
                    Console.WriteLine($"PROBE5 {file,-38} len={n}  max|d|={maxAbs,8:F3}  cos={dot / (Math.Sqrt(nc) * Math.Sqrt(nv)):F6}");
                }
            }
        }
    }

    /// <summary>
    /// SmolLM2 (headDim 64) is clean at len=1 and collapses at len=2, which implicates the KV
    /// round-trip. Both it and the working Qwen3 store KV as fp16, so if fp32 KV recovers len=2
    /// the defect is in the narrowed-KV append/attention path at headDim 64 specifically.
    /// </summary>
    [Fact]
    public void Probe_KvDtype()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            var path = FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (path is null) { Console.WriteLine("PROBE6: absent"); return; }

            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            int[] allToks = [1, 2, 3, 5];

            foreach (var kv in new DType?[] { null, DType.Float32, DType.BFloat16 })
            {
                for (int n = 1; n <= 2; n++)
                {
                    int[] toks = allToks[..n];

                    float[] cpu;
                    using (var cb = new CpuBackend())
                    using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                        cpu = cf.Prefill(toks).ToArray();

                    float[] vk;
                    try
                    {
                        using var vf = kv is null
                            ? new GpuForwardPass(model, gpu, hp, maxContextLength: 512)
                            : new GpuForwardPass(model, gpu, hp, maxContextLength: 512, kvDtype: kv.Value);
                        vk = vf.Prefill(toks).ToArray();
                    }
                    catch (Exception ex) { Console.WriteLine($"PROBE6 kv={kv?.ToString() ?? "default"} len={n}: {ex.GetType().Name}"); continue; }

                    double dot = 0, nc = 0, nv = 0;
                    for (int i = 0; i < cpu.Length; i++)
                    { dot += (double)cpu[i] * vk[i]; nc += (double)cpu[i] * cpu[i]; nv += (double)vk[i] * vk[i]; }
                    Console.WriteLine($"PROBE6 kv={kv?.ToString() ?? "default",-10} len={n}  cos={dot / (Math.Sqrt(nc) * Math.Sqrt(nv)):F6}");
                }
            }
        }
    }

    /// <summary>
    /// Compare CpuBackend.RoPE against VulkanBackend.RoPE directly — the real production kernels,
    /// not the inline reference RoPEMatchesCpu uses. Covers SmolLM2's shape (headDim 64, interleaved,
    /// theta 130k) and Qwen3's control shape (headDim 128, NEOX, theta 1M).
    /// </summary>
    [Fact]
    public void Probe_RoPEDirect()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        using (var cpu = new CpuBackend())
        {
            (int heads, int headDim, float theta, bool neox, string label)[] cases =
            [
                (32, 64, 130000f, false, "SmolLM2 (interleaved)"),
                (32, 128, 1000000f, true, "Qwen3   (neox)"),
                (32, 64, 130000f, true, "headDim64+neox"),
                (32, 128, 1000000f, false, "headDim128+interleaved"),
            ];

            foreach (var (heads, headDim, theta, neox, label) in cases)
            {
                int n = heads * headDim;
                var rng = new Random(7);
                var input = new float[n];
                for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

                foreach (int pos in new[] { 0, 1, 5 })
                {
                    var c = cpu.Upload(input, TensorShape.D1(n));
                    var g = gpu.Upload(input, TensorShape.D1(n));
                    cpu.RoPE(c, pos, headDim, theta, neox);
                    gpu.RoPE(g, pos, headDim, theta, neox);

                    var cr = new float[n];
                    var gr = new float[n];
                    cpu.Download(c, cr);
                    gpu.Download(g, gr);

                    double maxAbs = 0;
                    for (int i = 0; i < n; i++) maxAbs = Math.Max(maxAbs, Math.Abs(cr[i] - gr[i]));
                    Console.WriteLine($"PROBE7 {label,-24} pos={pos}  max|d|={maxAbs:E3}  {(maxAbs < 1e-3 ? "ok" : "<<< MISMATCH")}");

                    cpu.Free(c);
                    gpu.Free(g);
                }
            }
        }
    }

    /// <summary>
    /// Same two tokens, two routes: a 2-token Prefill (batched trunk) vs Prefill(1) + Forward
    /// (per-token path). Both must equal the CPU reference. If only the batched route is wrong,
    /// the defect is in the batched prefill trunk, not in RoPE/attention/KV per se.
    /// </summary>
    [Fact]
    public void Probe_BatchedVsPerToken()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            foreach (var file in new[] { "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf" })
            {
                var path = FindModelPath(file);
                if (path is null) { Console.WriteLine($"PROBE8 {file}: absent"); continue; }

                using var model = GgufModel.Open(path);
                var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
                int[] toks = [1, 2];

                float[] cpu;
                using (var cb = new CpuBackend())
                using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                    cpu = cf.Prefill(toks).ToArray();

                float[] batched;
                using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
                    batched = vf.Prefill(toks).ToArray();

                float[] perToken;
                using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
                {
                    vf.Prefill(toks[..1]);
                    perToken = vf.Forward(toks[1], 1).ToArray();
                }

                Console.WriteLine($"PROBE8 {file,-38} batchedCos={Cos(cpu, batched):F6}  perTokenCos={Cos(cpu, perToken):F6}");
            }
        }

        static double Cos(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }
    }

    /// <summary>
    /// Isolate the remaining suspects at SmolLM2's PRODUCTION shape (32 heads, 32 kv heads — no
    /// GQA — headDim 64), which the existing shader tests never exercise (they use headDim 32,
    /// 2-4 heads). Sweeps shapes so any dependence on headDim / head count / GQA is visible.
    /// </summary>
    [Theory]
    [InlineData(32, 32, 64, "SmolLM2 production shape")]
    [InlineData(2, 2, 32, "existing test shape (known good)")]
    [InlineData(32, 8, 128, "Qwen3 production shape")]
    [InlineData(32, 32, 128, "no-GQA, headDim 128")]
    [InlineData(2, 2, 64, "few heads, headDim 64")]
    public void Probe_AttentionShape(int numHeads, int numKvHeads, int headDim, string label)
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            const int seqLen = 2;
            int maxSeqLen = seqLen + 16;
            int kvDim = numKvHeads * headDim;

            var rng = new Random(11);
            var q = new float[numHeads * headDim];
            var kCache = new float[maxSeqLen * kvDim];
            var vCache = new float[maxSeqLen * kvDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < seqLen * kvDim; i++) kCache[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < seqLen * kvDim; i++) vCache[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: scaled dot-product attention with GQA.
            float scale = 1f / MathF.Sqrt(headDim);
            var expected = new float[numHeads * headDim];
            for (int h = 0; h < numHeads; h++)
            {
                int kvHead = h / (numHeads / numKvHeads);
                var scores = new float[seqLen];
                float max = float.NegativeInfinity;
                for (int t = 0; t < seqLen; t++)
                {
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[h * headDim + d] * kCache[t * kvDim + kvHead * headDim + d];
                    scores[t] = dot * scale;
                    if (scores[t] > max) max = scores[t];
                }
                float sum = 0;
                for (int t = 0; t < seqLen; t++) { scores[t] = MathF.Exp(scores[t] - max); sum += scores[t]; }
                for (int t = 0; t < seqLen; t++) scores[t] /= sum;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0;
                    for (int t = 0; t < seqLen; t++)
                        acc += scores[t] * vCache[t * kvDim + kvHead * headDim + d];
                    expected[h * headDim + d] = acc;
                }
            }

            var gq = gpu.Upload(q, TensorShape.D1(q.Length));
            var gk = gpu.Upload(kCache, TensorShape.D1(kCache.Length));
            var gv = gpu.Upload(vCache, TensorShape.D1(vCache.Length));
            var go = gpu.Allocate(TensorShape.D1(q.Length));
            var gs = gpu.Allocate(TensorShape.D1(1));

            gpu.Attention(gq, gk, gv, go, gs,
                (uint)numHeads, (uint)numKvHeads, (uint)headDim, seqLen, (uint)maxSeqLen);

            var got = new float[q.Length];
            gpu.Download(go, got);

            double maxAbs = 0;
            for (int i = 0; i < got.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - expected[i]));
            Console.WriteLine($"PROBE9 {label,-32} heads={numHeads,3} kv={numKvHeads,3} hd={headDim,4}  max|d|={maxAbs:E3}  {(maxAbs < 1e-3 ? "ok" : "<<< MISMATCH")}");

            gpu.Free(gq); gpu.Free(gk); gpu.Free(gv); gpu.Free(go); gpu.Free(gs);
        }
    }

    /// <summary>
    /// At len=1 softmax over one key is 1.0, so attention returns V and a WRONG K is invisible.
    /// The len=1-clean / len=2-broken signature therefore points at the K path. Append two
    /// positions and verify both slots of both caches land where attention will read them.
    /// </summary>
    [Theory]
    [InlineData(32, 64, "SmolLM2 (kvDim 2048)")]
    [InlineData(8, 128, "Qwen3   (kvDim 1024)")]
    public void Probe_KvAppend(int numKvHeads, int headDim, string label)
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            int kvDim = numKvHeads * headDim;
            const int maxSeqLen = 18;

            var rng = new Random(23);
            var kCache = new float[maxSeqLen * kvDim];
            var vCache = new float[maxSeqLen * kvDim];
            var gk = gpu.Upload(kCache, TensorShape.D1(kCache.Length));
            var gv = gpu.Upload(vCache, TensorShape.D1(vCache.Length));

            var written = new float[2][];
            var writtenV = new float[2][];
            for (uint pos = 0; pos < 2; pos++)
            {
                var kIn = new float[kvDim];
                var vIn = new float[kvDim];
                for (int i = 0; i < kvDim; i++) { kIn[i] = (float)(rng.NextDouble() * 2 - 1); vIn[i] = (float)(rng.NextDouble() * 2 - 1); }
                written[pos] = kIn; writtenV[pos] = vIn;

                var gkIn = gpu.Upload(kIn, TensorShape.D1(kvDim));
                var gvIn = gpu.Upload(vIn, TensorShape.D1(kvDim));
                gpu.KvAppend(gkIn, gvIn, gk, gv, (uint)kvDim, pos, maxSeqLen);
                gpu.Free(gkIn); gpu.Free(gvIn);
            }

            var gotK = new float[kCache.Length];
            var gotV = new float[vCache.Length];
            gpu.Download(gk, gotK);
            gpu.Download(gv, gotV);

            for (int pos = 0; pos < 2; pos++)
            {
                double dk = 0, dv = 0;
                for (int i = 0; i < kvDim; i++)
                {
                    dk = Math.Max(dk, Math.Abs(gotK[pos * kvDim + i] - written[pos][i]));
                    dv = Math.Max(dv, Math.Abs(gotV[pos * kvDim + i] - writtenV[pos][i]));
                }
                Console.WriteLine($"PROBE10 {label,-22} pos={pos}  maxK|d|={dk:E3} maxV|d|={dv:E3}  {(dk < 1e-6 && dv < 1e-6 ? "ok" : "<<< MISMATCH")}");
            }

            gpu.Free(gk); gpu.Free(gv);
        }
    }

    /// <summary>
    /// The decisive one: compare CPU and Vulkan hidden state after EVERY layer at position 1
    /// (where the divergence appears). The first layer whose cosine drops names the culprit.
    /// </summary>
    [Fact]
    public void Probe_LayerTaps()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            var path = FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (path is null) { Console.WriteLine("PROBE11: absent"); return; }

            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            int[] layerIds = Enumerable.Range(0, hp.NumLayers).ToArray();
            int[] toks = [1, 2];

            using var cb = new CpuBackend();
            using var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp);
            cf.EnableHiddenTaps(layerIds);
            cf.Prefill(toks[..1]);
            cf.Forward(toks[1], 1);

            // CPU logits at position 1, for the end-to-end comparison below.
            float[] cpuLogits;
            using (var cb2 = new CpuBackend())
            using (var cf2 = new OpenTail.Stingray.Engine.ForwardPass(model, cb2, hp))
            {
                cf2.Prefill(toks[..1]);
                cpuLogits = cf2.Forward(toks[1], 1).ToArray();
            }

            // Vulkan WITHOUT taps — the normal single-command-buffer path.
            float[] vkPlain;
            using (var vfPlain = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
            {
                vfPlain.Prefill(toks[..1]);
                vkPlain = vfPlain.Forward(toks[1], 1).ToArray();
            }

            using var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512);
            Console.WriteLine($"PROBE11 vulkan SupportsHiddenTaps={vf.SupportsHiddenTaps}");
            vf.EnableHiddenTaps(layerIds);
            vf.Prefill(toks[..1]);
            float[] vkTapped = vf.Forward(toks[1], 1).ToArray();

            // If capture (which inserts a submit + barriers per layer) makes the logits correct,
            // the defect is a missing synchronisation in the single-command-buffer path, not math.
            Console.WriteLine($"PROBE11 logits cos: vulkanPlain={Cos(cpuLogits, vkPlain):F6}  vulkanTapped={Cos(cpuLogits, vkTapped):F6}");

            static double Cos(float[] a, float[] b)
            {
                double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
                return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            }

            var cpuRow = cf.HiddenTapsAt(1);
            var vkRow = vf.HiddenTapsAt(1);
            Console.WriteLine($"PROBE11 tapDim cpu={cf.HiddenTapDim} vk={vf.HiddenTapDim} rows cpu={cpuRow.Length} vk={vkRow.Length}");
            if (cpuRow.Length == 0 || vkRow.Length == 0) { Console.WriteLine("PROBE11: a row was not captured"); return; }

            int embDim = hp.EmbeddingDim;
            for (int L = 0; L < hp.NumLayers; L++)
            {
                double dot = 0, nc = 0, nv = 0, maxAbs = 0;
                for (int i = 0; i < embDim; i++)
                {
                    float a = cpuRow[L * embDim + i], b = vkRow[L * embDim + i];
                    dot += (double)a * b; nc += (double)a * a; nv += (double)b * b;
                    maxAbs = Math.Max(maxAbs, Math.Abs(a - b));
                }
                double cos = dot / (Math.Sqrt(nc) * Math.Sqrt(nv));
                Console.WriteLine($"PROBE11 layer={L,3}  cos={cos:F6}  max|d|={maxAbs,10:E3}  {(cos < 0.99 ? "<<< DIVERGED" : "")}");
            }
        }
    }

    /// <summary>
    /// Self-consistency, per backend: a 2-token Prefill must equal Prefill(1) + Forward(pos 1) on
    /// the SAME backend. Whichever backend is internally inconsistent owns the defect — comparing
    /// CPU-prefill against Vulkan-prefill cannot tell you which side is wrong.
    /// </summary>
    [Fact]
    public void Probe_SelfConsistency()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        using (gpu)
        {
            foreach (var file in new[]
            {
                "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
                "Qwen3-8B-Q4_K_M.gguf",
                "gemma-4-E4B_q4_0-it.gguf",   // the two failing tests' model
            })
            {
                var path = FindModelPath(file);
                if (path is null) { Console.WriteLine($"PROBE12 {file}: absent"); continue; }

                using var model = GgufModel.Open(path);
                var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
                int[] toks = [1, 2];

                float[] cpuPrefill, cpuDecode, vkPrefill, vkDecode;

                using (var cb = new CpuBackend())
                using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                    cpuPrefill = cf.Prefill(toks).ToArray();

                using (var cb = new CpuBackend())
                using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                { cf.Prefill(toks[..1]); cpuDecode = cf.Forward(toks[1], 1).ToArray(); }

                using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
                    vkPrefill = vf.Prefill(toks).ToArray();

                using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
                { vf.Prefill(toks[..1]); vkDecode = vf.Forward(toks[1], 1).ToArray(); }

                // Same CPU prefill with int8 activation quantisation OFF. Decode never takes that
                // path, so if this restores self-consistency, Q8 prefill is the defect.
                float[] cpuPrefillF32;
                bool saved = SimdKernels.Q8PrefillEnabled;
                try
                {
                    SimdKernels.Q8PrefillEnabled = false;
                    using var cb = new CpuBackend();
                    using var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp);
                    cpuPrefillF32 = cf.Prefill(toks).ToArray();
                }
                finally { SimdKernels.Q8PrefillEnabled = saved; }

                Console.WriteLine(
                    $"PROBE12 {file,-38} cpuSelf={C(cpuPrefill, cpuDecode):F6}  vkSelf={C(vkPrefill, vkDecode):F6}  " +
                    $"decodeXback={C(cpuDecode, vkDecode):F6}  prefillXback={C(cpuPrefill, vkPrefill):F6}");
                Console.WriteLine(
                    $"PROBE12 {file,-38} Q8off: cpuSelf={C(cpuPrefillF32, cpuDecode):F6}  " +
                    $"vsVulkanPrefill={C(cpuPrefillF32, vkPrefill):F6}");
            }
        }

        static double C(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }
    }

    /// <summary>
    /// Where does Q8 prefill go wrong? Compare CPU per-layer hidden state with Q8 on vs off on the
    /// SAME backend. A smooth cosine decay means accumulating quantisation error (numerics); a
    /// cliff at one layer means a broken kernel.
    /// </summary>
    [Theory]
    [InlineData("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")]
    [InlineData("Qwen3-8B-Q4_K_M.gguf")]
    public void Probe_Q8LayerBisect(string file)
    {
        var path = FindModelPath(file);
        if (path is null) { Console.WriteLine($"PROBE13 {file}: absent"); return; }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] layerIds = Enumerable.Range(0, hp.NumLayers).ToArray();
        int[] toks = [1, 2];
        int embDim = hp.EmbeddingDim;

        float[] Run(bool q8)
        {
            bool saved = SimdKernels.Q8PrefillEnabled;
            try
            {
                SimdKernels.Q8PrefillEnabled = q8;
                using var cb = new CpuBackend();
                using var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp);
                cf.EnableHiddenTaps(layerIds);
                cf.Prefill(toks);
                return cf.HiddenTapsAt(1).ToArray();
            }
            finally { SimdKernels.Q8PrefillEnabled = saved; }
        }

        var on = Run(true);
        var off = Run(false);
        if (on.Length == 0 || off.Length == 0) { Console.WriteLine($"PROBE13 {file}: no taps captured (len on={on.Length} off={off.Length})"); return; }

        for (int L = 0; L < hp.NumLayers; L++)
        {
            double dot = 0, na = 0, nb = 0, maxAbs = 0;
            for (int i = 0; i < embDim; i++)
            {
                float a = off[L * embDim + i], b = on[L * embDim + i];
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
                maxAbs = Math.Max(maxAbs, Math.Abs(a - b));
            }
            double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            Console.WriteLine($"PROBE13 {file,-38} layer={L,3}  cos={cos:F6}  max|d|={maxAbs,10:E3}  {(cos < 0.99 ? "<<<" : "")}");
        }
    }

    /// <summary>
    /// Q8_K/Q8_KS carry one scale per 256-element block, so a single large activation outlier
    /// forces a huge scale and crushes every other value in that block toward zero. Compare the
    /// int8 path against the F32 path on identical inputs, with and without an outlier.
    /// </summary>
    [Theory]
    [InlineData(1f, "uniform")]
    [InlineData(50f, "outlier 50x")]
    [InlineData(500f, "outlier 500x")]
    [InlineData(5000f, "outlier 5000x")]
    public unsafe void Probe_Q8Outlier(float outlierScale, string label)
    {
        const int batchSize = 8, rows = 64, cols = 512;
        var rng = new Random(5);

        // Q4_K weights with sane fp16 block scales (random payload).
        long byteCount = DTypeInfo.ByteSize((long)rows * cols, DType.Q4_K);
        var weights = new byte[byteCount];
        rng.NextBytes(weights);
        for (int off = 0; off + 144 <= weights.Length; off += 144)
        {
            WriteHalf(weights, off, 0.015f);
            WriteHalf(weights, off + 2, 0.004f);
        }

        var input = new float[batchSize * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        // One outlier channel per token, as real transformer activations exhibit.
        if (outlierScale > 1f)
            for (int n = 0; n < batchSize; n++) input[n * cols + 17] = outlierScale;

        var q8 = new float[batchSize * rows];
        var f32 = new float[batchSize * rows];

        bool saved = SimdKernels.Q8PrefillEnabled;
        try
        {
            fixed (byte* w = weights)
            fixed (float* x = input)
            fixed (float* a = q8)
            fixed (float* b = f32)
            {
                SimdKernels.Q8PrefillEnabled = true;
                SimdKernels.MatMulBatched(a, w, x, batchSize, rows, cols, DType.Q4_K, allowQ8: true);
                SimdKernels.Q8PrefillEnabled = false;
                SimdKernels.MatMulBatched(b, w, x, batchSize, rows, cols, DType.Q4_K, allowQ8: true);
            }
        }
        finally { SimdKernels.Q8PrefillEnabled = saved; }

        double dot = 0, na = 0, nb = 0, maxAbs = 0;
        for (int i = 0; i < q8.Length; i++)
        {
            dot += (double)f32[i] * q8[i]; na += (double)f32[i] * f32[i]; nb += (double)q8[i] * q8[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(f32[i] - q8[i]));
        }
        Console.WriteLine($"PROBE14 {label,-14} cos={dot / (Math.Sqrt(na) * Math.Sqrt(nb)):F6}  max|d|={maxAbs:E3}");

        static void WriteHalf(byte[] buffer, int offset, float value)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
            buffer[offset] = (byte)(bits & 0xFF);
            buffer[offset + 1] = (byte)(bits >> 8);
        }
    }

    /// <summary>Dump per-tensor dtype/shape for the layers around the observed cliff.</summary>
    [Fact]
    public void Probe_TensorDtypes()
    {
        var path = FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
        if (path is null) { Console.WriteLine("PROBE15: absent"); return; }

        using var model = GgufModel.Open(path);
        foreach (int L in new[] { 0, 1, 21, 22, 23 })
        {
            foreach (var suffix in new[] { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" })
            {
                var t = model.FindTensor($"blk.{L}.{suffix}.weight");
                if (t is null) { Console.WriteLine($"PROBE15 blk.{L}.{suffix}: MISSING"); continue; }
                Console.WriteLine($"PROBE15 blk.{L,2}.{suffix,-12} dtype={t.Value.DType,-8} dims=[{string.Join(",", t.Value.Dimensions)}]");
            }
        }
        var emb = model.FindTensor("token_embd.weight");
        var outw = model.FindTensor("output.weight");
        Console.WriteLine($"PROBE15 token_embd dtype={emb?.DType.ToString() ?? "-"}  output.weight={(outw is null ? "ABSENT (tied)" : outw.Value.DType.ToString())}");
    }

    /// <summary>
    /// Sweep `cols` for the Q4_K int8 prefill path. In SmolLM2 every Q4_K tensor is cols=2048
    /// except ffn_down at cols=8192, and the per-layer error explodes exactly where ffn_down
    /// becomes Q4_K — so a cols-dependent defect is the live hypothesis.
    /// </summary>
    [Theory]
    [InlineData(DType.Q4_K, 256)]
    [InlineData(DType.Q4_K, 512)]
    [InlineData(DType.Q4_K, 2048)]
    [InlineData(DType.Q4_K, 4096)]
    [InlineData(DType.Q4_K, 8192)]
    [InlineData(DType.Q6_K, 8192)]
    public unsafe void Probe_Q8Cols(DType dtype, int cols)
    {
        const int batchSize = 8, rows = 32;
        var rng = new Random(5);

        long byteCount = DTypeInfo.ByteSize((long)rows * cols, dtype);
        var weights = new byte[byteCount];
        rng.NextBytes(weights);
        if (dtype == DType.Q4_K)
            for (int off = 0; off + 144 <= weights.Length; off += 144)
            { WriteHalf(weights, off, 0.015f); WriteHalf(weights, off + 2, 0.004f); }
        else
            for (int off = 0; off + 210 <= weights.Length; off += 210)
                WriteHalf(weights, off + 208, 0.012f);

        var input = new float[batchSize * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var q8 = new float[batchSize * rows];
        var f32 = new float[batchSize * rows];

        bool saved = SimdKernels.Q8PrefillEnabled;
        try
        {
            fixed (byte* w = weights)
            fixed (float* x = input)
            fixed (float* a = q8)
            fixed (float* b = f32)
            {
                SimdKernels.Q8PrefillEnabled = true;
                SimdKernels.MatMulBatched(a, w, x, batchSize, rows, cols, dtype, allowQ8: true);
                SimdKernels.Q8PrefillEnabled = false;
                SimdKernels.MatMulBatched(b, w, x, batchSize, rows, cols, dtype, allowQ8: true);
            }
        }
        finally { SimdKernels.Q8PrefillEnabled = saved; }

        double dot = 0, na = 0, nb = 0, maxAbs = 0;
        for (int i = 0; i < q8.Length; i++)
        {
            dot += (double)f32[i] * q8[i]; na += (double)f32[i] * f32[i]; nb += (double)q8[i] * q8[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(f32[i] - q8[i]));
        }
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        Console.WriteLine($"PROBE16 {dtype,-6} cols={cols,5}  cos={cos:F6}  max|d|={maxAbs:E3}  {(cos < 0.999 ? "<<< BROKEN" : "")}");

        static void WriteHalf(byte[] buffer, int offset, float value)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
            buffer[offset] = (byte)(bits & 0xFF);
            buffer[offset + 1] = (byte)(bits >> 8);
        }
    }

    /// <summary>
    /// Is the Q8-prefill defect prompt-length dependent? The shipped measurement reported
    /// bit-identical greedy generation on real (long) prompts, while my repro used 2 tokens.
    /// Compare Prefill(N) against Prefill(1)+Forward×(N-1) on the CPU for a range of N.
    /// </summary>
    [Theory]
    [InlineData("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")]
    [InlineData("Qwen3-8B-Q4_K_M.gguf")]
    public void Probe_Q8LengthDependence(string file)
    {
        var path = FindModelPath(file);
        if (path is null) { Console.WriteLine($"PROBE17 {file}: absent"); return; }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        var rng = new Random(3);
        int[] full = new int[64];
        for (int i = 0; i < full.Length; i++) full[i] = 100 + rng.Next(3000);

        foreach (int n in new[] { 2, 4, 8, 16, 32, 64 })
        {
            int[] toks = full[..n];

            float[] Prefill(bool q8)
            {
                bool saved = SimdKernels.Q8PrefillEnabled;
                try
                {
                    SimdKernels.Q8PrefillEnabled = q8;
                    using var cb = new CpuBackend();
                    using var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp);
                    return cf.Prefill(toks).ToArray();
                }
                finally { SimdKernels.Q8PrefillEnabled = saved; }
            }

            // Incremental decode reference — never touches the Q8 path.
            float[] decode;
            {
                using var cb = new CpuBackend();
                using var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp);
                cf.Prefill(toks[..1]);
                decode = null!;
                for (int i = 1; i < n; i++) decode = cf.Forward(toks[i], i).ToArray();
            }

            Console.WriteLine($"PROBE17 {file,-38} n={n,3}  q8On={C(decode, Prefill(true)):F6}  q8Off={C(decode, Prefill(false)):F6}");
        }

        static double C(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }
    }

    /// <summary>
    /// Gemma 4: both backends are self-consistent yet disagree (cos 0.628), and llama.cpp cannot
    /// load gemma4 so there is no third-party oracle. Compare CPU and Vulkan hidden state after
    /// every layer — the first layer that diverges names the stage to read.
    /// </summary>
    [Fact]
    public void Probe_GemmaLayerTaps()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE18: no Vulkan"); return; }

        using (gpu)
        {
            var path = FindModelPath(ModelFile);
            if (path is null) { Console.WriteLine("PROBE18: absent"); return; }

            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            int[] layerIds = Enumerable.Range(0, hp.NumLayers).ToArray();
            // Single token: position 0, where RoPE is the identity. If layer 0 still diverges,
            // RoPE (and therefore rope_freqs) is excluded and the cause is upstream — embedding
            // scale, PLE injection, or the norms.
            int[] toks = [2];
            int embDim = hp.EmbeddingDim;
            int pos = toks.Length - 1;

            using var cb = new CpuBackend();
            using var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp);
            cf.EnableHiddenTaps(layerIds);
            cf.Prefill(toks);

            using var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 2048);
            Console.WriteLine($"PROBE18 vk SupportsHiddenTaps={vf.SupportsHiddenTaps}");
            vf.EnableHiddenTaps(layerIds);
            vf.Prefill(toks);

            var cpuRow = cf.HiddenTapsAt(pos);
            var vkRow = vf.HiddenTapsAt(pos);
            Console.WriteLine($"PROBE18 rows cpu={cpuRow.Length} vk={vkRow.Length} (tapDim cpu={cf.HiddenTapDim} vk={vf.HiddenTapDim})");
            if (cpuRow.Length == 0 || vkRow.Length == 0) return;

            for (int L = 0; L < hp.NumLayers; L++)
            {
                double dot = 0, na = 0, nb = 0, maxAbs = 0;
                for (int i = 0; i < embDim; i++)
                {
                    float a = cpuRow[L * embDim + i], b = vkRow[L * embDim + i];
                    dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
                    maxAbs = Math.Max(maxAbs, Math.Abs(a - b));
                }
                double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
                Console.WriteLine($"PROBE18 layer={L,3}  cos={cos:F6}  max|d|={maxAbs,10:E3}  {(cos < 0.99 ? "<<<" : "")}");
            }
        }
    }

    /// <summary>
    /// Stage-level CPU vs Vulkan diff inside Gemma 4's layer 0 (position 0, so RoPE and attention
    /// are no-ops). The first stage whose cosine drops names the dispatch.
    /// </summary>
    [Fact]
    public void Probe_GemmaStages()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE19: no Vulkan"); return; }

        using (gpu)
        {
            var path = FindModelPath(ModelFile);
            if (path is null) { Console.WriteLine("PROBE19: absent"); return; }

            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            int[] toks = [2];

            StageCapture.Reset();
            StageCapture.Enabled = true;
            try
            {
                using (var cb = new CpuBackend())
                using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                    cf.Prefill(toks);

                using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
                    vf.Prefill(toks);
            }
            finally { StageCapture.Enabled = false; }

            Console.WriteLine($"PROBE19 records={StageCapture.Records.Count}");

            string[] stages =
            [
                StageCapture.Stages.AttnNorm,
                StageCapture.Stages.OProj,
                StageCapture.Stages.PostAttnResidual,
                StageCapture.Stages.PostFfnResidual,
                StageCapture.Stages.PostPle,
                StageCapture.Stages.LayerOutput,
            ];

            Report(-1, StageCapture.Stages.Embed);
            for (int layer = 0; layer <= 1; layer++)
                foreach (var stage in stages)
                    Report(layer, stage);

            StageCapture.Reset();

            static void Report(int layer, string stage)
            {
                var cpu = StageCapture.Find("cpu", layer, stage);
                var vk = StageCapture.Find("vulkan", layer, stage);
                if (cpu is null || vk is null)
                {
                    Console.WriteLine($"PROBE19 layer={layer,2} {stage,-16} MISSING (cpu={cpu is not null} vk={vk is not null})");
                    return;
                }
                double dot = 0, na = 0, nb = 0, maxAbs = 0;
                for (int i = 0; i < cpu.Length; i++)
                {
                    dot += (double)cpu[i] * vk[i]; na += (double)cpu[i] * cpu[i]; nb += (double)vk[i] * vk[i];
                    maxAbs = Math.Max(maxAbs, Math.Abs(cpu[i] - vk[i]));
                }
                double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
                Console.WriteLine($"PROBE19 layer={layer,2} {stage,-16} cos={cos:F6}  max|d|={maxAbs,10:E3}  {(cos < 0.999 ? "<<<" : "")}");
            }
        }
    }

    [Fact]
    public void Probe_PrefixLengthSweep()
    {
        VulkanBackend gpu;
        try { gpu = new VulkanBackend(); }
        catch { Console.WriteLine("PROBE: no Vulkan"); return; }

        var path = FindModelPath(ModelFile);
        if (path is null) { Console.WriteLine("PROBE: no model"); gpu.Dispose(); return; }

        using (gpu)
        using (var model = GgufModel.Open(path))
        {
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            int bos = 2;

            int[] passing = [bos, 651, 6037, 576, 6081, 603, 1234, 4567, 8901];
            int[] failing = [bos, 818, 5279, 529, 7001, 563];

            foreach (var (name, full) in new[] { ("PASSING-stream", passing), ("FAILING-stream", failing) })
            {
                for (int n = 1; n <= full.Length; n++)
                {
                    var toks = full[..n];

                    int cpuArg;
                    using (var cb = new CpuBackend())
                    using (var cf = new OpenTail.Stingray.Engine.ForwardPass(model, cb, hp))
                        cpuArg = Argmax(cf.Prefill(toks));

                    int vkArg;
                    using (var vf = new GpuForwardPass(model, gpu, hp, maxContextLength: 2048))
                        vkArg = Argmax(vf.Prefill(toks));

                    Console.WriteLine($"PROBE {name} len={n,2}  CPU={cpuArg,7}  VK={vkArg,7}  {(cpuArg == vkArg ? "ok" : "<<< DIVERGE")}");
                }
            }
        }
    }
}
