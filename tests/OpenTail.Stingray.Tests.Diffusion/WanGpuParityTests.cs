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
        using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, numLayers, numTxt);
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
}
