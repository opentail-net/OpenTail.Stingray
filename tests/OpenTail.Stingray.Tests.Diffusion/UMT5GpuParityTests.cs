using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class UMT5GpuParityTests
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
    public void UMT5Encoder_EncodeGpu_MatchesCpuReference_Numerically()
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        const int layers = 24;
        const int dim = 4096;
        const int ffDim = 10240;
        const int seq = 16;
        const int vocabSize = 32128; // Small test vocab

        var rng = new Random(42);
        var loader = new SyntheticWeightLoader();

        // Token embeddings
        var tokEmb = new float[vocabSize * dim];
        for (int i = 0; i < seq * dim; i++) tokEmb[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.05f;
        loader.Add("token_embedding.weight", tokEmb);

        // Rel pos bias weights: [32, 64]
        var rpW = new float[32 * 64];
        for (int i = 0; i < rpW.Length; i++) rpW[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;

        // 24 Layers weights
        var ln0 = new float[dim];
        Array.Fill(ln0, 1.0f);
        var qW = new float[dim * dim];
        var kW = new float[dim * dim];
        var vW = new float[dim * dim];
        var oW = new float[dim * dim];
        for (int i = 0; i < dim * dim; i += 100)
        {
            qW[i] = 0.01f;
            kW[i] = 0.01f;
            vW[i] = 0.01f;
            oW[i] = 0.01f;
        }

        var ln1 = new float[dim];
        Array.Fill(ln1, 1.0f);
        var wi0 = new float[ffDim * dim];
        var wi1 = new float[ffDim * dim];
        var wo = new float[dim * ffDim];
        for (int i = 0; i < 1000; i++)
        {
            wi0[i] = 0.01f;
            wi1[i] = 0.01f;
            wo[i] = 0.01f;
        }

        for (int l = 0; l < layers; l++)
        {
            string p = $"blocks.{l}";
            loader.Add($"{p}.pos_embedding.embedding.weight", rpW);
            loader.Add($"{p}.norm1.weight", ln0);
            loader.Add($"{p}.attn.q.weight", qW);
            loader.Add($"{p}.attn.k.weight", kW);
            loader.Add($"{p}.attn.v.weight", vW);
            loader.Add($"{p}.attn.o.weight", oW);
            loader.Add($"{p}.norm2.weight", ln1);
            loader.Add($"{p}.ffn.gate.0.weight", wi0);
            loader.Add($"{p}.ffn.fc1.weight", wi1);
            loader.Add($"{p}.ffn.fc2.weight", wo);
        }

        var fnW = new float[dim];
        Array.Fill(fnW, 1.0f);
        loader.Add("norm.weight", fnW);

        using var encoder = UMT5Encoder.FromLoader(loader);
        var tokens = new int[seq];
        for (int i = 0; i < seq; i++) tokens[i] = i;

        var cpuOut = encoder.Encode(tokens);
        var gpuOut = encoder.EncodeGpu(tokens, vulkan);

        Assert.Equal(cpuOut.Length, gpuOut.Length);
        float maxDiff = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 0.08f, $"Max difference {maxDiff} exceeded tolerance 0.08");
    }
}
