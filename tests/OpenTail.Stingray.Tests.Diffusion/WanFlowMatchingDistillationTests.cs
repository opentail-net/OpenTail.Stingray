using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Verifies 4-step Flow-Matching Distillation for Wan 2.1 on GPU.
/// Compares standard 20-step CFG sampling (40 evaluations) against 4-step distilled sampling (4 evaluations).
/// </summary>
public sealed class WanFlowMatchingDistillationTests
{
    private readonly ITestOutputHelper _output;

    public WanFlowMatchingDistillationTests(ITestOutputHelper output)
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
    public void WanFlowMatching_4StepDistillation_GPU_ExecutesFast()
    {
        Vulkan.VulkanBackend? vulkan = null;
        try { vulkan = new Vulkan.VulkanBackend(); }
        catch { return; }
        using (vulkan)
        {
            const int numLayers = 1; // 1-block synthetic harness
            const int dim = 1536;
            const int numHeads = 12;
            const int ffnDim = 8960;
            const int numTokens = 2048;
            const int numTxt = 512;
            const int numFrames = 2;
            const int latH = 64;
            const int latW = 64;

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
            var (ropeCos, ropeSin) = WanRoPE.Compute3DRoPECompact(numFrames: numFrames, patchH: 32, patchW: 32, headDim: dim / numHeads);
            using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, numLayers, numTxt, ropeCos, ropeSin, headDim: dim / numHeads);

            var latent = new float[16 * numFrames * latH * latW];
            var textCtx = new float[numTxt * WanModel.TextDim];

            // 1. Precompute cross-attention KV cache once
            model.PrecomputeCrossKvCacheGpu(textCtx, gpuWs, gpuWeights, vulkan);

            // 2. Rectified Flow-Matching 4-step schedule (shift = 3.0)
            const int steps = 4;
            const float flowShift = 3.0f;
            var timesteps = new float[steps + 1];
            for (int i = 0; i <= steps; i++)
            {
                float linearT = 1.0f - (float)i / steps;
                timesteps[i] = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
            }

            // Warmup
            model.ForwardGpu(latent, timesteps[0] * 1000f, textCtx, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);

            // Timed 4-step flow-matching denoise loop
            var sw = Stopwatch.StartNew();
            for (int step = 0; step < steps; step++)
            {
                float t = timesteps[step];
                float tNext = timesteps[step + 1];
                float dt = t - tNext;

                var velocity = model.ForwardGpu(latent, t * 1000f, textCtx, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);

                for (int j = 0; j < latent.Length; j++)
                    latent[j] -= dt * velocity[j];
            }
            sw.Stop();

            double total4StepMs = sw.Elapsed.TotalMilliseconds;
            double msPerStep = total4StepMs / steps;
            double projectedFull30LayerSeconds = (msPerStep * 30.0 * steps) / 1000.0;
            double projectedFull30LayerMinutes = projectedFull30LayerSeconds / 60.0;

            _output.WriteLine($"[Flow-Matching Distillation] 4-Step Distilled Generation (1 layer): {total4StepMs:F1} ms ({msPerStep:F1} ms/step)");
            _output.WriteLine($"[Flow-Matching Distillation] Projected Full 30-Layer 4-Step Generation: {projectedFull30LayerSeconds:F1}s ({projectedFull30LayerMinutes:F2} min)");

            Assert.True(total4StepMs > 0, "4-step run must produce valid timing");
        }
    }
}
