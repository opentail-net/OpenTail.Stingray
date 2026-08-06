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
