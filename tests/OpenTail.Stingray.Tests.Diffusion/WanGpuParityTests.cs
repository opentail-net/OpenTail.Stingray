using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanGpuParityTests
{
    private sealed class SyntheticWeightLoader : IWeightLoader
    {
        private readonly Dictionary<string, float[]> _weights = new(StringComparer.Ordinal);

        public void Add(string name, float[] data) => _weights[name] = data;

        public bool Contains(string name) => _weights.ContainsKey(name);
        public float[] ReadF32(string name) => _weights[name];
        public DType GetDType(string name) => DType.Float32;
        public long[] GetShape(string name) => new long[] { _weights[name].Length };
        public IReadOnlyList<string> TensorNames => _weights.Keys.ToList();
        public bool TryGetRaw(string name, out nint dataPtr, out long byteLen, out DType dtype, out int rows, out int cols)
        {
            dataPtr = 0; byteLen = 0; dtype = DType.Float32; rows = 0; cols = 0;
            return false;
        }
        public void Dispose() { }
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void WanModel_ForwardGpu_MatchesForwardCpu_Numerically()
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return; // Skip cleanly on machines without Vulkan runtime

        const int numLayers = 1;
        const int dim = 1536;
        const int numHeads = 12;
        const int ffnDim = 8960;
        const int numTokens = 32; // Fast test shape
        const int numTxt = 16;

        var loader = new SyntheticWeightLoader();
        var random = new Random(42);

        void AddWeight(string name, int size)
        {
            var arr = new float[size];
            for (int i = 0; i < size; i++) arr[i] = (float)(random.NextDouble() * 0.02 - 0.01);
            loader.Add(name, arr);
        }

        AddWeight("patch_embedding.weight", dim * WanModel.InChannels);
        AddWeight("time_embedding.0.weight", dim * 256);
        AddWeight("time_embedding.2.weight", dim * dim);
        AddWeight("time_projection.1.weight", dim * 6 * dim);
        AddWeight("text_embedding.0.weight", dim * WanModel.TextDim);
        AddWeight("text_embedding.2.weight", dim * dim);

        AddWeight("blocks.0.modulation", dim * 6);
        AddWeight("blocks.0.norm3.weight", dim);
        AddWeight("blocks.0.norm3.bias", dim);
        AddWeight("blocks.0.self_attn.q.weight", dim * dim);
        AddWeight("blocks.0.self_attn.k.weight", dim * dim);
        AddWeight("blocks.0.self_attn.v.weight", dim * dim);
        AddWeight("blocks.0.self_attn.o.weight", dim * dim);
        AddWeight("blocks.0.self_attn.norm_q.weight", dim);
        AddWeight("blocks.0.self_attn.norm_k.weight", dim);

        AddWeight("blocks.0.cross_attn.q.weight", dim * dim);
        AddWeight("blocks.0.cross_attn.k.weight", dim * dim);
        AddWeight("blocks.0.cross_attn.v.weight", dim * dim);
        AddWeight("blocks.0.cross_attn.o.weight", dim * dim);
        AddWeight("blocks.0.cross_attn.norm_q.weight", dim);
        AddWeight("blocks.0.cross_attn.norm_k.weight", dim);

        AddWeight("blocks.0.ffn.0.weight", ffnDim * dim);
        AddWeight("blocks.0.ffn.2.weight", dim * ffnDim);

        AddWeight("head.modulation", dim * 2);
        AddWeight("head.head.weight", WanModel.InChannels * dim);

        using var model = new WanModel(loader, "", numLayers: numLayers, dim: dim, numHeads: numHeads, backend: vulkan);
        using var gpuWeights = model.GetOrCreateGpuWeights();
        var (ropeCos, ropeSin) = WanRoPE.Compute3DRoPECompact(numFrames: 2, patchH: 4, patchW: 4, headDim: dim / numHeads);
        using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, numLayers, numTxt, ropeCos, ropeSin, headDim: dim / numHeads);
        var cpuWs = new WanWorkspace(numTokens, dim, ffnDim, numLayers);

        var latent = new float[16 * 2 * 8 * 8]; // patchH=4, patchW=4 -> 2*4*4 = 32 tokens
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        var textCtx = new float[numTxt * WanModel.TextDim];
        for (int i = 0; i < textCtx.Length; i++) textCtx[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        // Precompute KV cache
        model.PrecomputeCrossKvCache(textCtx, cpuWs);
        model.PrecomputeCrossKvCacheGpu(textCtx, gpuWs, gpuWeights, vulkan);

        // Run CPU Forward
        var expected = model.Forward(latent, 500f, textCtx, numFrames: 2, latH: 8, latW: 8, cpuWs);

        // Run GPU Forward
        var actual = model.ForwardGpu(latent, 500f, textCtx, numFrames: 2, latH: 8, latW: 8, gpuWs, gpuWeights, vulkan);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-2f,
                $"Mismatch at index {i}: expected {expected[i]:F6}, actual {actual[i]:F6}, diff {MathF.Abs(expected[i] - actual[i]):E4}");
        }
    }

    private readonly ITestOutputHelper _output;

    public WanGpuParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Wan_RealWeights_InspectTensorNamesAndShapes()
    {
        string ditPath = @"C:\Git-Public\OpenTail.Stingray\models\wan2.1\wan2.1-t2v-1.3b-dit.safetensors";
        if (!File.Exists(ditPath)) return;

        using var st = SafetensorsLoader.Open(ditPath);
        _output.WriteLine($"Total tensors: {st.TensorCount}");
        foreach (var name in st.TensorNames)
        {
            if ((name.StartsWith("blocks.0.") || !name.StartsWith("blocks.")) && name.EndsWith(".bias"))
            {
                var data = st.ReadF32(name);
                float maxAbs = 0, sumSq = 0;
                for (int i = 0; i < data.Length; i++)
                {
                    float a = MathF.Abs(data[i]);
                    if (a > maxAbs) maxAbs = a;
                    sumSq += data[i] * data[i];
                }
                float norm = MathF.Sqrt(sumSq);
                Console.WriteLine($"[Wan Bias] {name,-32}: len={data.Length,5} maxAbs={maxAbs:F6} norm={norm:F4}");
            }
        }
    }

    [Fact]
    public void WanModel_ForwardGpu_MatchesForwardCpu_RealWeights()
    {
        string ditPath = @"C:\Git-Public\OpenTail.Stingray\models\wan2.1\wan2.1-t2v-1.3b-dit.safetensors";
        if (!File.Exists(ditPath)) return;

        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        using var loader = SafetensorsLoader.Open(ditPath);

        using var model = new WanModel(loader, "", backend: vulkan);
        int numLayers = model.NumLayers;
        int dim = model.Dim;
        int numHeads = model.NumHeads;
        int ffnDim = model.FfnDim;
        const int numFrames = 1, latH = 16, latW = 16;
        int patchH = latH / 2, patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW; // 64
        const int numTxt = 226;

        using var gpuWeights = model.GetOrCreateGpuWeights();
        var (ropeCos, ropeSin) = WanRoPE.Compute3DRoPECompact(numFrames, patchH, patchW, headDim: dim / numHeads);
        using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, numLayers, numTxt, ropeCos, ropeSin, headDim: dim / numHeads);
        var cpuWs = new WanWorkspace(numTokens, dim, ffnDim, numLayers);

        var random = new Random(42);
        var latent = new float[16 * numFrames * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        var textCtx = new float[numTxt * WanModel.TextDim];
        for (int i = 0; i < textCtx.Length; i++) textCtx[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        model.PrecomputeCrossKvCache(textCtx, cpuWs);
        model.PrecomputeCrossKvCacheGpu(textCtx, gpuWs, gpuWeights, vulkan);

        // Compare KV cache between CPU and GPU
        double kDot = 0, kNa = 0, kNb = 0, kMaxDiff = 0;
        var kGpu = new float[numTxt * dim];
        vulkan.Download(gpuWs.CrossKvCache[0].K, kGpu);
        for (int i = 0; i < kGpu.Length; i++)
        {
            float expectedK = cpuWs.CrossKvCache[0].K[i];
            kDot += (double)expectedK * kGpu[i];
            kNa += (double)expectedK * expectedK;
            kNb += (double)kGpu[i] * kGpu[i];
            double d = Math.Abs(expectedK - kGpu[i]);
            if (d > kMaxDiff) kMaxDiff = d;
        }
        double kCos = kDot / (Math.Sqrt(kNa) * Math.Sqrt(kNb) + 1e-12);
        void Log(string msg) { _output.WriteLine(msg); Console.WriteLine(msg); }

        static (double cos, double maxDiff, double normA, double normB) Compare(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0, maxDiff = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
                double d = Math.Abs(a[i] - b[i]);
                if (d > maxDiff) maxDiff = d;
            }
            double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
            return (cos, maxDiff, Math.Sqrt(na), Math.Sqrt(nb));
        }

        var cpuStages = new Dictionary<string, float[]>();
        var gpuStages = new Dictionary<string, float[]>();
        var cpuBlocks = new Dictionary<int, float[]>();
        var gpuBlocks = new Dictionary<int, float[]>();

        model.OnStageCpu = (name, data) => cpuStages[name] = data;
        model.OnStageGpu = (name, data) => gpuStages[name] = data;
        model.OnBlockOutputCpu = (b, data) => cpuBlocks[b] = data;
        model.OnBlockOutputGpu = (b, data) => gpuBlocks[b] = data;

        // Run CPU Forward
        var cpuOut = model.Forward(latent, 500f, textCtx, numFrames, latH, latW, cpuWs);

        // Run GPU Forward
        var gpuOut = model.ForwardGpu(latent, 500f, textCtx, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);

        // Stage-by-stage comparison
        foreach (var (k, cpuArr) in cpuStages)
        {
            if (gpuStages.TryGetValue(k, out var gpuArr))
            {
                var (c, md, na, nb) = Compare(cpuArr, gpuArr);
                Log($"[WanRealParity][Stage] {k,-20}: cosine={c:F9} maxDiff={md:F6} cpuNorm={na:F4} gpuNorm={nb:F4}");
            }
        }

        // Per-block comparison
        for (int b = 0; b < numLayers; b++)
        {
            if (cpuBlocks.TryGetValue(b, out var cpuB) && gpuBlocks.TryGetValue(b, out var gpuB))
            {
                var (c, md, na, nb) = Compare(cpuB, gpuB);
                Log($"[WanRealParity][Block {b,2}]: cosine={c:F9} maxDiff={md:F6} cpuNorm={na:F4} gpuNorm={nb:F4}");
            }
        }

        var (fCos, fMd, fNa, fNb) = Compare(cpuOut, gpuOut);
        Log($"[WanRealParity] 30-layer Forward CPU vs GPU: cosine={fCos:F9} cpuNorm={fNa:F6} gpuNorm={fNb:F6} maxDiff={fMd:F6}");

        // Profile real production fused batch (batchBlocks = 30)
        model.OnStageGpu = null;
        model.OnBlockOutputGpu = null;
        model.OnProfileLog = Log;
        Log("[WanRealParity] Running production fused forward pass (batchBlocks=30):");
        var gpuOutProd = model.ForwardGpu(latent, 500f, textCtx, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);
        var (pCos, pMd, _, _) = Compare(cpuOut, gpuOutProd);
        Log($"[WanRealParity] Production fused pass parity vs CPU: cosine={pCos:F9} maxDiff={pMd:F6}");
    }
}

