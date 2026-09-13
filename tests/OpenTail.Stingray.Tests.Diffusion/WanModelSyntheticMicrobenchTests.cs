using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanModelSyntheticMicrobenchTests
{
    private readonly ITestOutputHelper _output;

    public WanModelSyntheticMicrobenchTests(ITestOutputHelper output)
    {
        _output = output;
    }

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

    [Fact]
    public void WanModel_SingleBlock_RunsCleanly_WithWanWorkspace()
    {
        const int numLayers = 1;
        const int dim = 1536;
        const int numHeads = 12;
        const int ffnDim = 8960;
        const int numTokens = 2048;
        const int numTxt = 512;

        var loader = new SyntheticWeightLoader();
        var random = new Random(42);

        // Helper to populate synthetic weights
        void AddWeight(string name, int size)
        {
            var arr = new float[size];
            for (int i = 0; i < size; i++) arr[i] = (float)(random.NextDouble() * 0.02 - 0.01);
            loader.Add(name, arr);
        }

        // Patch embed, time proj, modulation, self-attn, cross-attn, ffn, head
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

        using var model = new WanModel(loader, "", numLayers: numLayers, dim: dim, numHeads: numHeads);
        var ws = new WanWorkspace(numTokens, dim, ffnDim, numLayers);

        var latent = new float[16 * 2 * 64 * 64];
        var textCtx = new float[numTxt * WanModel.TextDim];

        // Warmup & precompute text KV cache
        model.PrecomputeCrossKvCache(textCtx, ws);

        // Warmup runs
        for (int i = 0; i < 2; i++)
        {
            model.Forward(latent, 500f, textCtx, numFrames: 2, latH: 64, latW: 64, ws);
        }

        const int iterations = 5;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            model.Forward(latent, 500f, textCtx, numFrames: 2, latH: 64, latW: 64, ws);
        }
        sw.Stop();

        double msPerBlock = sw.Elapsed.TotalMilliseconds / iterations;
        double projectedFullDiTSeconds = (msPerBlock * 30.0 * 40.0) / 1000.0;
        double projectedFullDiTMinutes = projectedFullDiTSeconds / 60.0;

        _output.WriteLine($"[Wan CPU Benchmark] 1-block forward ({numTokens} visual tokens, D={dim}, H={numHeads}): {msPerBlock:F1} ms");
        _output.WriteLine($"[Wan CPU Benchmark] Projected 30-layer x 40-forward full DiT CPU runtime: {projectedFullDiTSeconds:F1}s ({projectedFullDiTMinutes:F1} min)");
        Console.Error.WriteLine($"[Wan CPU Benchmark] 1-block forward ({numTokens} visual tokens, D={dim}, H={numHeads}): {msPerBlock:F1} ms");
        Console.Error.WriteLine($"[Wan CPU Benchmark] Projected 30-layer x 40-forward full DiT CPU runtime: {projectedFullDiTSeconds:F1}s ({projectedFullDiTMinutes:F1} min)");
    }

    [Fact]
    public void WanModel_SingleBlock_RunsCleanly_WithWanGpuWorkspace()
    {
        Vulkan.VulkanBackend? vulkan = null;
        try { vulkan = new Vulkan.VulkanBackend(); }
        catch { return; }
        using (vulkan)
        {
            const int numLayers = 1;
            const int dim = 1536;
            const int numHeads = 12;
            const int ffnDim = 8960;
            const int numTokens = 2048;
            const int numTxt = 512;

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
            var (ropeCos, ropeSin) = OpenTail.Stingray.Diffusion.Wan.WanRoPE.Compute3DRoPECompact(numFrames: 2, patchH: 32, patchW: 32, headDim: dim / numHeads);
            using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, numLayers, numTxt, ropeCos, ropeSin, headDim: dim / numHeads);

            var latent = new float[16 * 2 * 64 * 64];
            var textCtx = new float[numTxt * WanModel.TextDim];

            // Warmup & precompute text KV cache on GPU
            model.PrecomputeCrossKvCacheGpu(textCtx, gpuWs, gpuWeights, vulkan);

            // Warmup runs
            for (int i = 0; i < 2; i++)
            {
                model.ForwardGpu(latent, 500f, textCtx, numFrames: 2, latH: 64, latW: 64, gpuWs, gpuWeights, vulkan);
            }

            const int iterations = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                model.ForwardGpu(latent, 500f, textCtx, numFrames: 2, latH: 64, latW: 64, gpuWs, gpuWeights, vulkan);
            }
            sw.Stop();

            double msPerBlock = sw.Elapsed.TotalMilliseconds / iterations;
            double projectedFullDiTSeconds = (msPerBlock * 30.0 * 40.0) / 1000.0;
            double projectedFullDiTMinutes = projectedFullDiTSeconds / 60.0;

            _output.WriteLine($"[Wan Vulkan GPU Benchmark] 1-block forward ({numTokens} visual tokens, D={dim}, H={numHeads}): {msPerBlock:F1} ms");
            _output.WriteLine($"[Wan Vulkan GPU Benchmark] Projected 30-layer x 40-forward full DiT Vulkan runtime: {projectedFullDiTSeconds:F1}s ({projectedFullDiTMinutes:F1} min)");
            Console.Error.WriteLine($"[Wan Vulkan GPU Benchmark] 1-block forward ({numTokens} visual tokens, D={dim}, H={numHeads}): {msPerBlock:F1} ms");
            Console.Error.WriteLine($"[Wan Vulkan GPU Benchmark] Projected 30-layer x 40-forward full DiT Vulkan runtime: {projectedFullDiTSeconds:F1}s ({projectedFullDiTMinutes:F1} min)");
        }
    }
}
